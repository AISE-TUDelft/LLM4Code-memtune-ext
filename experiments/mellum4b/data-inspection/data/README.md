Expected Mellum-4b data-inspection inputs:

- `pre-train_tags.csv`
- `fine-tune_tags.csv`

If these files are missing, `mellum-rq4.ipynb` can auto-generate them from:

- Mellum parquet outputs in `experiments/mellum4b/memorization/*`
- tag templates from `full_precision/quantization/lora` data-inspection folders

Manual placement is still supported.

Required columns:

- `Cat` (category labels)
- `agree` (optional; if present, rows with `agree == 1` are used)
- Mellum EM columns such as:
  - `me4b_ep0_p100_em`, `me4b_ep3_p100_em` for pre-train tags
  - `me4b_ep0_d3_em`, `me4b_ep3_d3_em` for fine-tune tags

Supported model prefixes in the notebook:

- `me4b_` (preferred)
- `mellum4b_` (fallback)
