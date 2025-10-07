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
`GTEx WGS`: dbGaP accession phs000424.v10.p2 (controlled access).

`CureGN WGS + clinical`: request via the Cure Glomerulonephropathy (CureGN) Consortium under applicable data-use agreements.
Scripts expect local paths to these datasets and annotations; see comments in volaria_main/ and manuscript/.


---
### Example usage notes:

Please see `demo` for sample walkthrough, getting from example VCF to predictions 

In summary, the pipeline is as follows:

- Build embeddings (example in volaria_main/ Steps 1–5)

Then, either:
- Train outcome models (Step 6-7).
- Generate figures/tables via scripts in manuscript/ (see file headers for inputs).

Or, use pre-trained models
- Use embeddings to get predictions



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
