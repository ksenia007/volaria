# Volaria Demo Pipeline

This demo illustrates how to process a sample VCF file through the **Volaria** genomic embedding and prediction pipeline - starting from raw variants and ending with disease-outcome predictions.

---

## Environment Setup

Before running the pipeline, create a clean Python environment with the required dependencies:

```bash
pyenv install 3.10.9
pyenv local 3.10.9  
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes about the sample file

Note that for privacy reasons the provided .vcf is a sample generated file that does not represent a real person. It is provided as a test file to try running the model 

## File setup 

All the required files are provided in the repo. Global variables are

```bash
VCF="KOSAMPLE.vcf.gz"
REG="regulatory_predictions.csv"
EXONIC="exonic_predictions.tsv"
WORKDIR="./demo_run"
GENE_MAP="gene_map_names.csv"
SCALE_LOC="../manuscript/models/models_regress.pkl"
MODELS_BASE="../manuscript/models/outcomes/"
```
They correspond to: 

| Variable      | Description                                                     |
| ------------- | --------------------------------------------------------------- |
| `VCF`         | Input VCF file containing variant calls for one or more samples |
| `REG`         | Regulatory variant predictions (Expecto-SC or similar)          |
| `EXONIC`      | Exonic variant predictions (AlphaMissense or similar)           |
| `GENE_MAP`    | Gene ID <-> name mapping table                                    |
| `SCALE_LOC`   | Path to saved PCA/scaler/model pack (`models_regress.pkl`)      |
| `MODELS_BASE` | Directory containing trained outcome models                     |
| `WORKDIR`     | Working directory for intermediate and final outputs            |

---

## Step-by-Step Pipeline

### 1. Convert Variants -> Dict

Each variant in the VCF is assigned a unified key (`CHROM_POS_REF_ALT`) and grouped by sample.

```bash
python ../volaria_main/step1_convert_into_dict.py \
  --vcf "$VCF" \
  --sample-prefix "KO" \
  --output-folder "$WORKDIR/variants_to_person" \
  --output-prefix "variants_to_person"
```

**Expected log**
```
[info] Input: KOSAMPLE.vcf.gz
[info] Samples kept: 1; example: ['KOSAMPLE']
[info] Output prefix: variants_to_person
[info] require_record_pass=True  require_sample_substr=None
[info] Variant key = CHROM_POS_REF_ALT (fixed across cohorts)
[progress] processed ~30000 records
[save] ./demo_run/variants_to_person  (40000 variants)
[done] all shards written.
```

Saves sharded pickles under `./demo_run/variants_to_person`

---

### 2. Flip to Patient -> Variants

Combine the per-variant dictionaries into a single dictionary keyed by individual.

```bash
python ../volaria_main/step2_flip_dict_patient_variant.py \
  "$WORKDIR/variants_to_person" \
  "$WORKDIR/persons_to_variants.pkl" \
  "$VCF"
```

**Expected log**
```
loc_dicts ./demo_run/variants_to_person
save_file ./demo_run/persons_to_variants.pkl
original_file KOSAMPLE.vcf.gz
patient_list ['KOSAMPLE']
N patients: 1
example keys: ['KOSAMPLE']
variants_to_person.split_0.pkl
```
---

### 3. Collect Regulatory Predictions

Extract per-variant regulatory scores (e.g., from ExpectoSC) and aggregate them for each patient.

```bash
python ../volaria_main/step3_collect_expectosc.py \
  --scores_file "$EXPECTO_SUB" \
  --pat_dict_file "$WORKDIR/persons_to_variants.pkl" \
  --save_file_full "$WORKDIR/expectosc_by_person.full.pkl" \
  --save_file_embedding "$WORKDIR/expectosc_by_person.embedding.pkl"
```
---

### 4. Collect Exonic Predictions

Integrate per-variant exonic effect scores (e.g., AlphaMissense) and map to gene names.

```bash
python ../volaria_main/step4_collect_alphamissense.py \
  --scores-csv "$AM_SUB" \
  --patient-dict "$WORKDIR/persons_to_variants.pkl" \
  --out-pkl "$WORKDIR/am_by_person.full.pkl" \
  --id-format curegn \
  --gene-map "$GENE_MAP"
```

### 5. Combine into Patient Embeddings

Merge regulatory and exonic features

```bash
python ../volaria_main/step5_combine_all_predictions.py \
  --am_pickle "$WORKDIR/am_by_person.full.pkl" \
  --sc_pickle "$WORKDIR/expectosc_by_person.embedding.pkl" \
  --out_pickle "$WORKDIR/combined_mean_counts_weighted.pkl" \
  --long_keys "podocyte,bcell,cd4tcell,cd8tcell,glomerularendothelium,myofibroblast,weighted_AM_mean" \
  --long_out_csv "$WORKDIR/combined_features_long.tsv"
```

**Expected log**

```
[ok] Saved combined dict → ./demo_run/combined_mean_counts_weighted.pkl
[ok] Saved long matrix (transpose) → ./demo_run/combined_features_long.tsv
```

---

### 6. Predict Outcomes

Generate per-outcome risk scores using the fitted scalers/PCA and trained models.

```bash
python ../volaria_main/get_predictions.py \
  --features       "$WORKDIR/combined_features_long.tsv" \
  --regressors-pkl "$SCALE_LOC" \
  --models-dir     "$MODELS_BASE" \
  --outdir         "$WORKDIR/preds"
```

**Expected log**

```
[info] residual ABS features: 1
[info] ESRD: need=947; X.shape=(1, 947)
[info] Steroid_resistant: need=947; X.shape=(1, 947)
[info] eGFR40: need=947; X.shape=(1, 947)
[ok] saved ./demo_run/preds/predictions.csv (rows=3)
```

Output:

* `./demo_run/preds/predictions.csv`  -> final per-endpoint predictions

---

Since this is a sample demo, runtime is expected to be short — under 5 minutes total.
Full-files runtime depends on the number of individuals, the number of variants per person, and the availability of a GPU to obtain regulatory predictions.
