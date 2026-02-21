# import
import pandas as pd
import torch

# parallel processing
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=16)
from tqdm import tqdm
tqdm.pandas()

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed

from experiments.utils.eval_metrics import em_compute, meteor_compute, bleu_compute, rouge_compute

import os

# config
set_seed(42)
checkpoint = "JetBrains/Mellum-4b-base"
p_name = "me4b"

epochs = ['_ep0', '_ep1', '_ep2', '_ep3']
epoch = epochs[0]

plen = ['_p100', '_p150', '_p200', '_p250']

print(f"----Starting attack with model: {p_name} and {epoch} epochs----")

# load the data needed
df = load_dataset("AISE-TUDelft/memtune-data_attack", name = "pre-train", split = "dg3")
df = df.select_columns(['prefix_250', 'prefix_200', 'prefix_150', 'prefix_100', 'suffix'])

# load the model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

# set the generation pipeline
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer, framework='pt', max_new_tokens=50)

gen_name_100 = p_name + epoch + plen[0]
gen_name_150 = p_name + epoch + plen[1]
gen_name_200 = p_name + epoch + plen[2]
gen_name_250 = p_name + epoch + plen[3]

# perform the data extraction attacks
gs_mem_100 = pipe(list(df['prefix_100']), batch_size=32)
print('100 done')
gs_mem_150 = pipe(list(df['prefix_150']), batch_size=32)
print('150 done')
gs_mem_200 = pipe(list(df['prefix_200']), batch_size=32)
print('200 done')
gs_mem_250 = pipe(list(df['prefix_250']), batch_size=32)
print('250 done')
torch.cuda.empty_cache()

# Save the results on each of the datasets
df = df.add_column(gen_name_100, [gs_mem_100[i][0]['generated_text'][len(list(df['prefix_100'])[i]):]  for i,_ in enumerate(df)])
df = df.add_column(gen_name_150, [gs_mem_150[i][0]['generated_text'][len(list(df['prefix_150'])[i]):]  for i,_ in enumerate(df)])
df = df.add_column(gen_name_200, [gs_mem_200[i][0]['generated_text'][len(list(df['prefix_200'])[i]):]  for i,_ in enumerate(df)])
df = df.add_column(gen_name_250, [gs_mem_250[i][0]['generated_text'][len(list(df['prefix_250'])[i]):]  for i,_ in enumerate(df)])

# compute the evaluation metrics
expath = './'

df_eval = df.to_pandas()

em = gen_name_100 + '_em'
bleu = gen_name_100 + '_bleu'
met = gen_name_100 + '_meteor'
roug = gen_name_100 + '_rougeL'
df_eval[em] = df_eval.progress_apply(lambda x: em_compute( pred= [x[gen_name_100]], suffix = [x['suffix']]), axis=1)
df_eval[bleu] = df_eval.progress_apply(lambda x: bleu_compute( pred= [x[gen_name_100]], suffix = [x['suffix']]), axis=1)
df_eval[met] = df_eval.progress_apply(lambda x: meteor_compute( pred= [x[gen_name_100]], suffix = [x['suffix']]), axis=1)
df_eval[roug] = df_eval.progress_apply(lambda x: rouge_compute( pred= [x[gen_name_100]], suffix = [x['suffix']]), axis=1)

em = gen_name_150 + '_em'
bleu = gen_name_150 + '_bleu'
met = gen_name_150 + '_meteor'
roug = gen_name_150 + '_rougeL'
df_eval[em] = df_eval.progress_apply(lambda x: em_compute( pred= [x[gen_name_150]], suffix = [x['suffix']]), axis=1)
df_eval[bleu] = df_eval.progress_apply(lambda x: bleu_compute( pred= [x[gen_name_150]], suffix = [x['suffix']]), axis=1)
df_eval[met] = df_eval.progress_apply(lambda x: meteor_compute( pred= [x[gen_name_150]], suffix = [x['suffix']]), axis=1)
df_eval[roug] = df_eval.progress_apply(lambda x: rouge_compute( pred= [x[gen_name_150]], suffix = [x['suffix']]), axis=1)

em = gen_name_200 + '_em'
bleu = gen_name_200 + '_bleu'
met = gen_name_200 + '_meteor'
roug = gen_name_200 + '_rougeL'
df_eval[em] = df_eval.progress_apply(lambda x: em_compute( pred= [x[gen_name_200]], suffix = [x['suffix']]), axis=1)
df_eval[bleu] = df_eval.progress_apply(lambda x: bleu_compute( pred= [x[gen_name_200]], suffix = [x['suffix']]), axis=1)
df_eval[met] = df_eval.progress_apply(lambda x: meteor_compute( pred= [x[gen_name_200]], suffix = [x['suffix']]), axis=1)
df_eval[roug] = df_eval.progress_apply(lambda x: rouge_compute( pred= [x[gen_name_200]], suffix = [x['suffix']]), axis=1)

em = gen_name_250 + '_em'
bleu = gen_name_250 + '_bleu'
met = gen_name_250 + '_meteor'
roug = gen_name_250 + '_rougeL'
df_eval[em] = df_eval.progress_apply(lambda x: em_compute( pred= [x[gen_name_250]], suffix = [x['suffix']]), axis=1)
df_eval[bleu] = df_eval.progress_apply(lambda x: bleu_compute( pred= [x[gen_name_250]], suffix = [x['suffix']]), axis=1)
df_eval[met] = df_eval.progress_apply(lambda x: meteor_compute( pred= [x[gen_name_250]], suffix = [x['suffix']]), axis=1)
df_eval[roug] = df_eval.progress_apply(lambda x: rouge_compute( pred= [x[gen_name_250]], suffix = [x['suffix']]), axis=1)
print("Evaluation metrics computation completed")

# save the results
df_eval.to_parquet(os.path.join(expath, 'attack_forg_plen.parquet'))
print("Results saved to attack_forg_plen.parquet")
print(f"----End attack----")
