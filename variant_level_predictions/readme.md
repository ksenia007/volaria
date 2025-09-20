# Prepare variant effects (exonic, VEP + AlphaMissense)

This folder provides an example on how to:
1) select variants in specified regions from a source VCF, and  
2) annotate those variants with coding and non-coding variant effects. 

Note that this part is often specific to the file-formatting and thus can vary. 
---

## Requirements

- Bgzipped + indexed VCF (`.vcf.gz` with `.tbi`)
- BED file of regions to keep (e.g., exon windows or 20kb windows around TSS - provided in this repository)
- Ensembl VEP installed with matching cache and reference FASTA for your genome build
- AlphaMissense predictions table for the same build 

> **Build consistency:** VCF, VEP cache, FASTA, and AlphaMissense file must all be GRCh37 **or** all be GRCh38.

---

## Subset variants to regions 

It is recommended to subset to the bed regions of interest (e.g. `exonic_regions.bed`), otherwise the files become too large

Example with vcftools
```bash
vcftools \
  --gzvcf /path/to/source/ALL.chr22.genotypes.vcf.gz \
  --bed /path/to/regions/select_regions.bed \
  --recode \
  --out /path/to/out/chr22.filtered
```

## Get coding variant effect predictions

Then, assuming VEP is installed, use AM plugin:

```bash
vep \
  -i /path/to/out/chr22.filtered.vcf \
  -o /path/to/out/chr22.vep.txt \
  --vcf \
  --offline \
  --dir_cache /path/to/vep/cache \
  --fasta /path/to/reference/GRCh37.fa.gz \
  --lookup_ref \
  --force_overwrite \
  --plugin AlphaMissense,file=/path/to/AlphaMissense_hg19.tsv.gz,cols=all
```


## Non-coding effects (ExpectoSC)

For non-coding/regulatory variant effect scoring (e.g., 20kb windows, using `nc_20kb_regions.bed`), follow the instructions in the [ExpectoSC](https://github.com/ksenia007/ExPectoSC) repository (installation, model weights, and scoring commands). Use the same VCF region-filtering approach above to produce the variant lists you’ll score with ExpectoSC. **Note that this step requires GPU access.**
