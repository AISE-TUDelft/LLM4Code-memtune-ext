# Experiment setup

Output file: 
- `attack_mem_dedup_1.parquet`
- `attack_mem_dedup_2.parquet`
- `attack_mem_dedup_3.parquet`
- `attack_mem_dedup_g3.parquet`

(number of .py files a file for model)
Model: StarCoder2-15B
Epochs: 0, 3 (zero is the baseline to be used for comparison)
Precision: bfloat16, int8, int4

For each model I need to perform the attack with
- p100 (since is fixed is not explicit in the naming)

For each model we perform 4 different attacks, each one with its specific duplication sample set. 
Each file is an attack on a sample set with its specific duplication rate.
- d1
- d2
- d3
- dg3

bf16_ep0_d1
bf16_ep0_d2
bf16_ep0_d3
bf16_ep0_dg3

bf16_ep3_d1
bf16_ep3_d2
bf16_ep3_d3
bf16_ep3_dg3

i8_ep0_d1
i8_ep0_d2
i8_ep0_d3
i8_ep0_dg3

i8_ep3_d1
i8_ep3_d2
i8_ep3_d3
i8_ep3_dg3

i4_ep0_d1
i4_ep0_d2
i4_ep0_d3
i4_ep0_dg3

i4_ep3_d1
i4_ep3_d2
i4_ep3_d3
i4_ep3_dg3

The fp32 corresponds to data that we already have:
15b_ep0_d1
15b_ep0_d2
15b_ep0_d3
15b_ep0_dg3

15b_ep3_d1
15b_ep3_d2
15b_ep3_d3
15b_ep3_dg3

The idea is to launch for each single folder an executable that runs all the experiments.

Fine tuned models from hugging face:
- AISE-TUDelft/StarCoder2Java-15b_ep1
- AISE-TUDelft/StarCoder2Java-15b_ep2
- AISE-TUDelft/StarCoder2Java-15b_ep3