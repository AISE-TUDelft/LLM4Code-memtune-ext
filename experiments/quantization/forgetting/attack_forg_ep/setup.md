# Experiment setup
The only thing that changes compared to the memorization attack is the attack dataset (the samples comes now from the pre-training set). 
The rest remains unchanged

Output file: a single `attack_forg_ep.parquet` file.

(number of .py files a file for model)
Model: StarCoder2-15B
Epochs: 0, 1, 2, 3 (zero is the baseline to be used for comparison)
Precision: bfloat16, int8, int4

For each model I need to perform the attack with
- p100 (since is fixed is not explicit in the naming)

The duplication rate is fixed on d-g3 (since is fixed is not explicit in the naming)

bf16_ep0
bf16_ep1
bf16_ep2
bf16_ep3

i8_ep0
i8_ep1
i8_ep2
i8_ep3

i4_ep0
i4_ep1
i4_ep2
i4_ep3

The fp32 corresponds to data that we already have:
15b_ep0
15b_ep1
15b_ep2
15b_ep3

The idea is to launch for each single folder an executable that runs all the experiments.


Fine tuned models from hugging face:
- AISE-TUDelft/StarCoder2Java-15b_ep1
- AISE-TUDelft/StarCoder2Java-15b_ep2
- AISE-TUDelft/StarCoder2Java-15b_ep3