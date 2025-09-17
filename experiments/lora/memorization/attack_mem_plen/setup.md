# Experiment setup

Output file: a single `attack_mem_plen.parquet` file.

(number of .py files a file for model)
Model: StarCoder2-7B
Epochs: 0, 3 (zero is the baseline to be used for comparison)

For each model I need to perform the attack with
- p100
- p150
- p200
- p250

The duplication rate is fixed on d-g3 (since is fixed is not explicit in the naming)

The experiment is:
7b_lo_ep3_p100

The baseline is this:
7b_ep0_p100

The classic fine-tuning corresponds to data that we already have:
7b_ep3_p100
...

The idea is to launch for each single folder an executable that runs all the experiments.
aalkaswan/StarCoder2Java-7b_LoRA_ep3