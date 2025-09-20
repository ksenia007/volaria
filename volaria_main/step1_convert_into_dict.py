"""
Convert VCF into variant to patients dict with FIXED key: CHROM_POS_REF_ALT

Buckets: '11' (hom-alt), '10' (het), 'dot' (any '.' in GT)
Default: require record-level FILTER == PASS
Optional: require substring in SAMPLE field (e.g., 'PASS') per-sample (KO quirk)

Use EITHER:
  --vcf /full/path/to/chr10.20k.vcf.recode.vcf          (single-file mode; recommended for weird basenames)
OR
  --chrom 10 --file-subname 20k --loc /data             (chrom mode ⇒ /data/chr10.20k.vcf.recode.vcf)

Sample selection is PREFIX-ONLY (e.g., --sample-prefix KO or --sample-prefix GTEX).
"""
from __future__ import annotations
import argparse, os, gzip, pickle
from typing import List, Tuple, Optional
import numpy as np

def open_maybe_gzip(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")

def resolve_paths(args) -> Tuple[str, str]:
    if args.vcf:
        in_path = args.vcf
        out_prefix = args.output_prefix or (args.vcf + ".processed")
        return in_path, out_prefix
    if not (args.chrom and args.file_subname and args.loc):
        raise SystemExit("Chrom mode requires --chrom --file-subname --loc (else use --vcf).")
    in_path = os.path.join(args.loc, f"chr{args.chrom}.{args.file_subname}.vcf.recode.vcf")
    out_dir = os.path.join(args.loc, "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = args.output_prefix or os.path.join(out_dir, f"chr{args.chrom}.{args.file_subname}")
    return in_path, out_prefix

def parse_header_get_samples(header_line: str, sample_prefix: Optional[str]) -> Tuple[List[str], List[int]]:
    cols = header_line.strip().split()
    try:
        fmt_idx = cols.index("FORMAT")
    except ValueError:
        raise SystemExit("VCF header missing FORMAT column.")
    all_ids = cols[fmt_idx+1:]
    if not all_ids:
        raise SystemExit("No samples found after FORMAT.")
    def keep(s: str) -> bool:
        return s.startswith(sample_prefix) if sample_prefix is not None else True
    kept = [s for s in all_ids if keep(s)]
    if not kept:
        raise SystemExit("No samples matched --sample-prefix.")
    kept_idx = [cols.index(s) for s in kept]  # indices in the line
    return kept, kept_idx

def read_header_samples(vcf_path: str, sample_prefix: Optional[str]):
    with open_maybe_gzip(vcf_path) as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM") or line.startswith("#CHR"):
                return parse_header_get_samples(line, sample_prefix)
            if not line.startswith("#"):
                raise SystemExit("Malformed VCF: data before #CHROM header.")
    raise SystemExit("No #CHROM header found.")

def make_variant_id(fields: List[str]) -> str:
    # 0=CHROM, 1=POS, 3=REF, 4=ALT
    return f"{fields[0]}_{fields[1]}_{fields[3]}_{fields[4]}"

def classify_gt(sample_field: str) -> str:
    gt = sample_field.split(":", 1)[0]
    if "." in gt:
        return "dot"
    has1 = "1" in gt
    has0 = "0" in gt
    if has1 and has0:
        return "10"
    if has1 and not has0:
        return "11"
    return "skip"  # e.g., 0/0

def process_vcf(in_path: str,
                out_path: str,
                out_prefix: str,
                sample_ids: List[str],
                sample_cols: List[int],
                shard_size: int = 100_000,
                require_record_pass: bool = True,
                require_sample_substr: Optional[str] = None,
                progress_every: int = 30_000):
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    n = len(sample_ids)
    maxlen = max(len(s) for s in sample_ids) + 1
    np_dtype = f"<U{maxlen}"
    variants_patients = {}
    shard_idx = 0
    data_lines = 0

    def save_shard():
        nonlocal variants_patients, shard_idx
        # create an os path to the new file, with folder and file name
        save_file = out_path+'/'+f"{out_prefix}.split_{shard_idx}.pkl"
        with open(save_file, "wb") as f:
            pickle.dump(variants_patients, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[save] {out_path}  ({len(variants_patients)} variants)")
        variants_patients = {}
        shard_idx += 1

    print(f"[info] Input: {in_path}")
    print(f"[info] Samples kept: {n}; example: {sample_ids[:3]}")
    print(f"[info] Output prefix: {out_prefix}")
    print(f"[info] require_record_pass={require_record_pass}  require_sample_substr={require_sample_substr}")
    print("[info] Variant key = CHROM_POS_REF_ALT (fixed across cohorts)")

    with open_maybe_gzip(in_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            data_lines += 1
            if progress_every and (data_lines % progress_every == 0):
                print(f"[progress] processed ~{data_lines} records")

            # VCF is tab-delimited; stick to split("\t") to avoid surprises
            fields = line.rstrip("\n").split("\t")

            # Record-level PASS (FILTER column idx 6)
            if require_record_pass and fields[6] != "PASS":
                continue

            var_id = make_variant_id(fields)

            p11 = np.empty(n, dtype=np_dtype); i11 = 0
            p10 = np.empty(n, dtype=np_dtype); i10 = 0
            pdd = np.empty(n, dtype=np_dtype); idd = 0

            for j, col_idx in enumerate(sample_cols):
                sf = fields[col_idx]
                if require_sample_substr and (require_sample_substr not in sf):
                    continue
                c = classify_gt(sf)
                if c == "11":
                    p11[i11] = sample_ids[j]; i11 += 1
                elif c == "10":
                    p10[i10] = sample_ids[j]; i10 += 1
                elif c == "dot":
                    pdd[idd] = sample_ids[j]; idd += 1

            variants_patients[var_id] = {"11": p11[:i11], "10": p10[:i10], "dot": pdd[:idd]}

            if shard_size and (len(variants_patients) >= shard_size):
                save_shard()

    if variants_patients:
        save_shard()
    print("[done] all shards written.")

def main():
    ap = argparse.ArgumentParser(description="VCF into variant: [patients] dict (fixed key)")
    # Inputs
    ap.add_argument("--vcf", type=str, default=None, help="Path to VCF/VCF.GZ (use for odd basenames like chr10.20k.vcf.recode.vcf).")
    ap.add_argument("--chrom", type=str, default=None, help="Chrom name/number (chrom mode).")
    ap.add_argument("--file-subname", type=str, default=None, help="File subname used in chrom mode (e.g., '20k').")
    ap.add_argument("--loc", type=str, default=None, help="Base directory for chrom mode.")
    # Samples (PREFIX ONLY)
    ap.add_argument("--sample-prefix", type=str, default=None, help="Keep samples whose IDs START WITH this (e.g., KO, GTEX).")
    # Filtering / behavior
    ap.add_argument("--no-record-pass", action="store_true", help="Do not require FILTER==PASS at record level.")
    ap.add_argument("--require-sample-substr", type=str, default=None, help="Substring required inside SAMPLE field (e.g., 'PASS').")
    # Output / perf
    ap.add_argument("--output-folder", type=str, default=None, help="Output folder (chrom mode). Created if needed.")
    ap.add_argument("--output-prefix", type=str, default=None, help="Prefix for {prefix}.split_k.pkl.")
    ap.add_argument("--shard-size", type=int, default=100_000, help="Variants per shard (default 100k).")
    ap.add_argument("--progress-every", type=int, default=30_000, help="Progress stride (default 30k).")
    args = ap.parse_args()

    in_path, out_prefix = resolve_paths(args)
    sample_ids, sample_cols = read_header_samples(in_path, args.sample_prefix)
    process_vcf(
        in_path=in_path,
        out_path=args.output_folder,
        out_prefix=out_prefix,
        sample_ids=sample_ids,
        sample_cols=sample_cols,
        shard_size=args.shard_size,
        require_record_pass=not args.no_record_pass,
        require_sample_substr=args.require_sample_substr,
        progress_every=args.progress_every,
    )

if __name__ == "__main__":
    main()
