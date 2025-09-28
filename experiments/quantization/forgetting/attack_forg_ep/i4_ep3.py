# import
import pandas as pd
import torch

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, set_seed

from experiments.utils.eval_metrics import em_compute, meteor_compute, bleu_compute, rouge_compute

# parallel processing
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=16)
from tqdm import tqdm
tqdm.pandas()

import os

# config
set_seed(42)

checkpoint = "bigcode/starcoder2-15b"
fine_tuned_model = "AISE-TUDelft/StarCoder2Java-15b_ep3"

p_names = ["bf16", "i8", "i4"]
p_name = p_names[2]

epochs = ['_ep0', '_ep1', '_ep2', '_ep3']
epoch = epochs[3]

p_length = 'prefix_100'

print(f"----Starting attack with model: {p_name} and {epoch} epochs----")

# load the data needed
df = load_dataset("AISE-TUDelft/memtune-data_attack", name = "pre-train", split = "dg3")
df = df.select_columns(['prefix_250', 'prefix_200', 'prefix_150', 'prefix_100', 'suffix'])

# load the model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(fine_tuned_model, device_map="auto", quantization_config=quantization_config)
model.eval()

# set the generation pipeline
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer, framework='pt', max_new_tokens=50)

# IMPORTANT
gen_name = p_name + epoch

# perform the data extraction attacks
gs_mem = pipe(list(df[p_length]), batch_size=32)
print('attack done')
torch.cuda.empty_cache()

# save the result
df = df.add_column(gen_name, [gs_mem[i][0]['generated_text'][len(list(df[p_length])[i]):]  for i,_ in enumerate(df)])

# compute the evaluation metrics
em = gen_name + '_em'
bleu = gen_name + '_bleu'
met = gen_name + '_meteor'
roug = gen_name + '_rougeL'

# After is saved the first time
df_eval = pd.merge((pd.read_parquet('./attack_forg_ep.parquet')), df.to_pandas()[[gen_name]], left_index =True, right_index=True)

# evaluation
df_eval[em] = df_eval.progress_apply(lambda x: em_compute( pred= [x[gen_name]], suffix = [x['suffix']]), axis=1)
df_eval[bleu] = df_eval.progress_apply(lambda x: bleu_compute( pred= [x[gen_name]], suffix = [x['suffix']]), axis=1)
df_eval[met] = df_eval.progress_apply(lambda x: meteor_compute( pred= [x[gen_name]], suffix = [x['suffix']]), axis=1)
df_eval[roug] = df_eval.progress_apply(lambda x: rouge_compute( pred= [x[gen_name]], suffix = [x['suffix']]), axis=1)
print("Evaluation metrics computation completed")

# Save the results
df_eval.to_parquet('./attack_forg_ep.parquet', index = False)
print("Results saved to attack_forg_ep.parquet")
print(f"----End attack----")