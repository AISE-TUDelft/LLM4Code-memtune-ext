# Experiment setup

Output file: a single `attack_mem_ep.parquet` file.

(number of .py files a file for model)
Model: StarCoder2-7B
Epochs: 0, 1, 2, 3 (zero is the baseline to be used for comparison)

For each model I need to perform the attack with
- p100 (since is fixed is not explicit in the naming)

The duplication rate is fixed on d-g3 (since is fixed is not explicit in the naming)
7b_lo_ep1
7b_lo_ep2
7b_lo_ep3

The othe finetuning is already done
7b_ep1
7b_ep2
7b_ep3

Epoch zero is already done

The idea is to launch for each single folder an executable that runs all the experiments.

aalkaswan/StarCoder2Java-7b_LoRA_ep1
aalkaswan/StarCoder2Java-7b_LoRA_ep2
aalkaswan/StarCoder2Java-7b_LoRA_ep3