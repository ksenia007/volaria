# Volaria

Code and workflows to recreate the results of the manuscript 📄 

- `variant_effects/` – region selection (vcftools) and how-to info on getting variant effects. See `readme` for more information
- `volaria_main/` – stepwise pipelines for embeddings and analyses:
  - **Steps 1–5:** build variant-effect embeddings (GTEx example provided).
  - **Step 6:** cohort integration (CureGN / GTEx, as in manuscript).
  - **Step 7:** outcome model training.
- `manuscript/` - code needed to reproduce manuscript Figures and Tables
- `requirements.txt` - Python 3.10.9 environment 
- `demo` - sample workflow to run the model end-to-end on example VCF


### Data access 
The raw whole genome sequence data used in this study are available through dbGaP under accession numbers phs000424.v10.p2 (GTEx) and phs002480.v3.p3 (CureGN). GENCODE Release 19, GRCh37.p13 was used to identify TSS locations. 

Scripts expect local paths to these datasets and annotations; see comments in volaria_main/ and manuscript/.


---
### Example usage notes:

Please see `demo` for sample walkthrough, getting from example VCF to predictions 

In summary, the pipeline is as follows:

- Build embeddings (example in volaria_main/ Steps 1–5)
- [option 1] Train new outcome models (Step 6-7).
- [option 2] Use pre-trained models

Code in `manuscript` can be used to generate figures/tables via scripts in manuscript/ (see file comment on the top of the file for inputs).

---
### Environment Setup

Before running the pipeline, create a clean Python environment with the required dependencies:

```bash
pyenv install 3.10.9
pyenv local 3.10.9  
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Environment time installation should be negligible; sample run is also quick. Full run on custom cohorts would depend on the # individuals and # variants.
