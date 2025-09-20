"""
GTEx Volaria pipeline (Steps 1–5) — streaming demo

This script *demonstrates* how one would run GTEx Volaria embedding creation
"""

import os
import shlex
import subprocess
from pathlib import Path

# configure neutral roots 
ROOT_GTEXT = Path("/ROOT/GTEx/output/<user>")
ROOT_VOLARIA = ROOT_GTEXT / "volaria"
ROOT_RESOURCES = Path("/ROOT/curegn/resources")

# inputs
VCF_EXONS = ROOT_GTEXT / "exons" / "GTEx_filtered_exons_all.vcf"
VCF_NC20K = ROOT_GTEXT / "nc_20kb_variants" / "GTEx_filtered_20kbp_all.vcf"
EXPECTOSC_CSV = ROOT_GTEXT / "nc_20kb_expectosc" / "all_kidney.csv"
VEP_TABLE = ROOT_GTEXT / "exons" / "vep_output_exons.txt"
GENE_MAP_CSV = ROOT_RESOURCES / "compiled_CAGE_Gencode_tss_renamed.csv"

# outputs
STEP1_EXONS_DIR = ROOT_VOLARIA / "step1_output" / "exons"
STEP1_NC20_DIR = ROOT_VOLARIA / "step1_output" / "nc_20kbp"
STEP2_DIR = ROOT_VOLARIA / "step2_output"
STEP3_DIR = ROOT_VOLARIA / "step3_output"
STEP4_DIR = ROOT_VOLARIA / "step4_output"
STEP5_DIR = ROOT_VOLARIA / "step5_output"

EXONS_PREFIX = "gtex_exons_all"
NC20_PREFIX = "gtex_nc_20kb"
PAT_DICT_EXONS = STEP2_DIR / "exons_dict.pkl"
PAT_DICT_NC20 = STEP2_DIR / "nc_20kbp_dict.pkl"
EXOSC_FULL_PKL = STEP3_DIR / "expectosc_full.pkl"
EXOSC_MAIN_PKL = STEP3_DIR / "expectosc_main.pkl"
AM_EXPANDED_CSV = STEP4_DIR / "vep_output_exons.AM_expanded.csv"
AM_MAIN_PKL = STEP4_DIR / "AM_main.pkl"
COMBINED_FULL = STEP5_DIR / "combined_embeddings_full.pkl"
COMBINED_MAIN = STEP5_DIR / "combined_embeddings_main.csv"  
LONG_KEYS = "podocyte,cd4tcell,cd8tcell,glomerularendothelium,myofibroblast,weighted_AM_mean"

def ensure_dirs():
    for d in [STEP1_EXONS_DIR, STEP1_NC20_DIR, STEP2_DIR, STEP3_DIR, STEP4_DIR, STEP5_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def run_stream(cmd: str, env=None) -> int:
    print(f"[run] {cmd}")
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env or os.environ.copy(),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    proc.wait()
    print(f"[exit] code={proc.returncode}\n")
    return proc.returncode

def main():
    ensure_dirs()

    cmds = [
        f"python step1_convert_into_dict.py --vcf {VCF_EXONS} --sample-prefix GTEX "
        f"--output-folder {STEP1_EXONS_DIR} --output-prefix {EXONS_PREFIX}",

        f"python step1_convert_into_dict.py --vcf {VCF_NC20K} --sample-prefix GTEX "
        f"--output-folder {STEP1_NC20_DIR} --output-prefix {NC20_PREFIX}",

        f"python step2_flip_dict_patient_variant.py {STEP1_EXONS_DIR} {PAT_DICT_EXONS} {VCF_EXONS}",
        f"python step2_flip_dict_patient_variant.py {STEP1_NC20_DIR} {PAT_DICT_NC20} {VCF_NC20K}",

        f"python step3_collect_expectosc.py --scores_file {EXPECTOSC_CSV} "
        f"--pat_dict_file {PAT_DICT_NC20} --save_file_full {EXOSC_FULL_PKL} "
        f"--save_file_embedding {EXOSC_MAIN_PKL}",

        f"python step_AM_collect.py --vep_table {VEP_TABLE} --am_expanded_csv {AM_EXPANDED_CSV} "
        f"--gene_map_csv {GENE_MAP_CSV} --patient_dict_pkl {PAT_DICT_EXONS} --out_pickle {AM_MAIN_PKL}",

        f"python step5_combine_all_predictions.py --am_pickle {AM_MAIN_PKL} --sc_pickle {EXOSC_MAIN_PKL} "
        f"--out_pickle {COMBINED_FULL} --long_keys {LONG_KEYS} --long_out_csv {COMBINED_MAIN}",
    ]

    failed = []
    for cmd in cmds:
        if run_stream(cmd) != 0:
            failed.append(cmd)

    if failed:
        print("[summary] failures:")
        for c in failed:
            print(f"  - {c}")
    else:
        print("[summary] all steps completed successfully.")

    print("[artifact] primary output:", COMBINED_MAIN)

if __name__ == "__main__":
    main()
