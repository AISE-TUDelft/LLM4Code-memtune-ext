# Experiment setup

Output file: a single `attack_mem_plen.parquet` file.

(number of .py files a file for model)
Model: StarCoder2-15B
Epochs: 0, 3 (zero is the baseline to be used for comparison)
Precision: bfloat16, int8, int4

For each model I need to perform the attack with
- p100
- p150
- p200
- p250

The duplication rate is fixed on d-g3 (since is fixed is not explicit in the naming)

bf16_ep0_p100
bf16_ep3_p100

i8_ep0_p100
i8_ep3_p100

i4_ep0_p100
i4_ep3_p100

The fp32 corresponds to data that we already have:
15b_ep0_p100
15b_ep3_p100

The idea is to launch for each single folder an executable that runs all the experiments.