VCF="KOSAMPLE.vcf.gz"
REG="regulatory_predictions.csv"
EXONIC="exonic_predicitons.tsv"
WORKDIR="./demo_run"
GENE_MAP="gene_map_names.csv"
SCALE_LOC="../manuscript/models/models_regress.pkl"
MODELS_BASE="../manuscript/models/outcomes/"

mkdir -p "$WORKDIR"

# 1) Variants -> dict 
python ../volaria_main/step1_convert_into_dict.py \
  --vcf "$VCF" \
  --sample-prefix "KO" \
  --output-folder "$WORKDIR/variants_to_person" \
  --output-prefix "variants_to_person"

# OUTPUT
# [info] Input: KOSAMPLE.vcf.gz
# [info] Samples kept: 1; example: ['KOSAMPLE']
# [info] Output prefix: variants_to_person
# [info] require_record_pass=True  require_sample_substr=None
# [info] Variant key = CHROM_POS_REF_ALT (fixed across cohorts)
# [progress] processed ~30000 records
# [save] ./demo_run/variants_to_person  (40000 variants)
# [done] all shards written.

# 2) Flip to patient -> variants 
python ../volaria_main/step2_flip_dict_patient_variant.py \
  "$WORKDIR/variants_to_person" \
  "$WORKDIR/persons_to_variants.pkl" \
  "$VCF"

# OUTPUT
# loc_dicts ./demo_run/variants_to_person
# save_file ./demo_run/persons_to_variants.pkl
# original_file KOSAMPLE.vcf.gz
# patient_list ['KOSAMPLE']
# N patients: 1
# example keys: ['KOSAMPLE']
# variants_to_person.split_0.pkl
# len file 1

# 3) Collect regulatory predictions
python ../volaria_main/step3_collect_expectosc.py \
  --scores_file "$EXPECTO_SUB" \
  --pat_dict_file "$WORKDIR/persons_to_variants.pkl" \
  --save_file_full "$WORKDIR/expectosc_by_person.full.pkl" \
  --save_file_embedding "$WORKDIR/expectosc_by_person.embedding.pkl"
  # OUTPUT
  # long output, ending with 
  # Saved embeddings: ./demo_run/expectosc_by_person.embedding.pkl
# Time taken: 0.49 seconds

# 4) Collect exonic predictions
python ../volaria_main/step4_collect_alphamissense.py \
  --scores-csv "$AM_SUB" \
  --patient-dict "$WORKDIR/persons_to_variants.pkl" \
  --out-pkl "$WORKDIR/am_by_person.full.pkl" \
  --id-format curegn \
  --gene-map "$GENE_MAP"
  # OUTPUT
  # Saved: ./demo_run/am_by_person.full.pkl

# 5) Combine into raw embeddings
python ../volaria_main/step5_combine_all_predictions.py \
  --am_pickle "$WORKDIR/am_by_person.full.pkl" \
  --sc_pickle "$WORKDIR/expectosc_by_person.embedding.pkl" \
  --out_pickle "$WORKDIR/combined_mean_counts_weighted.pkl" \
  --long_keys "podocyte,bcell,cd4tcell,cd8tcell,glomerularendothelium,myofibroblast,weighted_AM_mean" \
  --long_out_csv "$WORKDIR/combined_features_long.tsv"

  # OUTPUT
#   [ok] Saved combined dict → ./demo_run/combined_mean_counts_weighted.pkl
# [ok] Saved long matrix (transpose) → ./demo_run/combined_features_long.tsv

# 6) Predict 
python ../volaria_main/get_predictions.py \
  --features       "$WORKDIR/combined_features_long.tsv" \
  --regressors-pkl "$SCALE_LOC" \
  --models-dir     "$MODELS_BASE" \
  --outdir         "$WORKDIR/preds" 
# OUTPUT
# [info] residual ABS features: 1
# [info] ESRD: need=947; X.shape=(1, 947)
# [info] Steroid_resistant: need=947; X.shape=(1, 947)
# [info] eGFR40: need=947; X.shape=(1, 947)
# [ok] saved ./demo_run/preds/predictions.csv (rows=3)
