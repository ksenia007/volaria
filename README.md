# Volaria

Code and workflows to recreate the results of the manuscript 📄 

- `variant_effects/` – region selection (vcftools) and how-to info on getting variant effects. See `readme` for more information
- `volaria_main/` – stepwise pipelines for embeddings and analyses:
  - **Steps 1–5:** build variant-effect embeddings (GTEx example provided).
  - **Step 6:** cohort integration (CureGN / GTEx, as in manuscript).
  - **Step 7:** outcome model training.
- `manuscript/` - code needed to reproduce manuscript Figures and Tables
- `requirements.txt` - Python environment


### Data access 
`GTEx WGS`: dbGaP accession phs000424.v10.p2 (controlled access).
`CureGN WGS + clinical`: request via the Cure Glomerulonephropathy (CureGN) Consortium under applicable data-use agreements.
Scripts expect local paths to these datasets and annotations; see comments in volaria_main/ and manuscript/.


### Example usage notes:

- Build embeddings (example in volaria_main/ Steps 1–5).
- Integrate cohorts & adjust for PCs (Step 6).
- Train outcome models (Step 7).
- Generate figures/tables via scripts in manuscript/ (see file headers for inputs).