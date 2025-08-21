# import
import pandas as pd
import torch

# parallel processing
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=16)
from tqdm import tqdm
tqdm.pandas()

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

from experiments.utils.eval_metrics import em_compute, meteor_compute, bleu_compute, rouge_compute

import os

# config
expath = './'

checkpoint = "bigcode/starcoder2-15b"
fine_tuned_model = "AISE-TUDelft/StarCoder2Java-15b_ep3"
p_names = ["bf16", "i8", "i4"]
p_name = p_names[2]

epochs = ['_ep0', '_ep1', '_ep2', '_ep3']
epoch = epochs[3]

deduplication = ['_d1', '_d2','_d3','_dg3']

p_length = 'prefix_100'

print(f"----Starting attack with model: {p_name} and {epoch} epochs----")

# Loading the data
df_1 = load_dataset("AISE-TUDelft/memtune-data_attack", name = "fine-tuning", split = "d1")
df_1 = df_1.select_columns(['prefix_250', 'prefix_200', 'prefix_150', 'prefix_100', 'suffix'])

df_2 = load_dataset("AISE-TUDelft/memtune-data_attack", name = "fine-tuning", split = "d2")
df_2 = df_2.select_columns(['prefix_250', 'prefix_200', 'prefix_150', 'prefix_100', 'suffix'])

df_3 = load_dataset("AISE-TUDelft/memtune-data_attack", name = "fine-tuning", split = "d3")
df_3 = df_3.select_columns(['prefix_250', 'prefix_200', 'prefix_150', 'prefix_100', 'suffix'])

df_g3 = load_dataset("AISE-TUDelft/memtune-data_attack", name = "fine-tuning", split = "dg3")
df_g3 = df_g3.select_columns(['prefix_250', 'prefix_200', 'prefix_150', 'prefix_100', 'suffix'])

# load the model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModelForCausalLM.from_pretrained(fine_tuned_model, device_map="auto", quantization_config=quantization_config)
model.eval()

# set the generation pipeline
pipe = pipeline("text-generation", model = model, tokenizer = tokenizer, framework='pt', max_new_tokens=50)

# IMPORTANT
gen_name_1 = p_name + epoch + deduplication[0]
gen_name_2 = p_name + epoch + deduplication[1]
gen_name_3 = p_name + epoch + deduplication[2]
gen_name_g3 = p_name + epoch + deduplication[3]
# perform the data extraction attacks
gs_mem_1 = pipe(df_1[p_length], batch_size=32)
print('1 done')
gs_mem_2 = pipe(df_2[p_length], batch_size=32)
print('2 done')
gs_mem_3 = pipe(df_3[p_length], batch_size=32)
print('3 done')
gs_mem_g3 = pipe(df_g3[p_length], batch_size=32)
print('g3 done')
torch.cuda.empty_cache()

# Save the results on each of the datasets
df_1 = df_1.add_column(gen_name_1, [gs_mem_1[i][0]['generated_text'][len(df_1[p_length][i]):]  for i,_ in enumerate(df_1)])
df_2 = df_2.add_column(gen_name_2, [gs_mem_2[i][0]['generated_text'][len(df_2[p_length][i]):]  for i,_ in enumerate(df_2)])
df_3 = df_3.add_column(gen_name_3, [gs_mem_3[i][0]['generated_text'][len(df_3[p_length][i]):]  for i,_ in enumerate(df_3)])
df_g3 = df_g3.add_column(gen_name_g3, [gs_mem_g3[i][0]['generated_text'][len(df_g3[p_length][i]):]  for i,_ in enumerate(df_g3)])

# After is saved the first time
df_eval_1 = pd.merge((pd.read_parquet(expath + 'attack_mem_dedup_1.parquet')), df_1.to_pandas()[[gen_name_1]], left_index =True, right_index=True)
df_eval_2 = pd.merge((pd.read_parquet(expath + 'attack_mem_dedup_2.parquet')), df_2.to_pandas()[[gen_name_2]], left_index =True, right_index=True)
df_eval_3 = pd.merge((pd.read_parquet(expath + 'attack_mem_dedup_3.parquet')), df_3.to_pandas()[[gen_name_3]], left_index =True, right_index=True)
df_eval_g3 = pd.merge((pd.read_parquet(expath + 'attack_mem_dedup_g3.parquet')), df_g3.to_pandas()[[gen_name_g3]], left_index =True, right_index=True)

# Evaluation
em = gen_name_1 + '_em'
bleu = gen_name_1 + '_bleu'
met = gen_name_1 + '_meteor'
roug = gen_name_1 + '_rougeL'
df_eval_1[em] = df_eval_1.progress_apply(lambda x: em_compute( pred= [x[gen_name_1]], suffix = [x['suffix']]), axis=1)
df_eval_1[bleu] = df_eval_1.progress_apply(lambda x: bleu_compute( pred= [x[gen_name_1]], suffix = [x['suffix']]), axis=1)
df_eval_1[met] = df_eval_1.progress_apply(lambda x: meteor_compute( pred= [x[gen_name_1]], suffix = [x['suffix']]), axis=1)
df_eval_1[roug] = df_eval_1.progress_apply(lambda x: rouge_compute( pred= [x[gen_name_1]], suffix = [x['suffix']]), axis=1)

em = gen_name_2 + '_em'
bleu = gen_name_2 + '_bleu'
met = gen_name_2 + '_meteor'
roug = gen_name_2 + '_rougeL'
df_eval_2[em] = df_eval_2.progress_apply(lambda x: em_compute( pred= [x[gen_name_2]], suffix = [x['suffix']]), axis=1)
df_eval_2[bleu] = df_eval_2.progress_apply(lambda x: bleu_compute( pred= [x[gen_name_2]], suffix = [x['suffix']]), axis=1)
df_eval_2[met] = df_eval_2.progress_apply(lambda x: meteor_compute( pred= [x[gen_name_2]], suffix = [x['suffix']]), axis=1)
df_eval_2[roug] = df_eval_2.progress_apply(lambda x: rouge_compute( pred= [x[gen_name_2]], suffix = [x['suffix']]), axis=1)

em = gen_name_3 + '_em'
bleu = gen_name_3 + '_bleu'
met = gen_name_3 + '_meteor'
roug = gen_name_3 + '_rougeL'
df_eval_3[em] = df_eval_3.progress_apply(lambda x: em_compute( pred= [x[gen_name_3]], suffix = [x['suffix']]), axis=1)
df_eval_3[bleu] = df_eval_3.progress_apply(lambda x: bleu_compute( pred= [x[gen_name_3]], suffix = [x['suffix']]), axis=1)
df_eval_3[met] = df_eval_3.progress_apply(lambda x: meteor_compute( pred= [x[gen_name_3]], suffix = [x['suffix']]), axis=1)
df_eval_3[roug] = df_eval_3.progress_apply(lambda x: rouge_compute( pred= [x[gen_name_3]], suffix = [x['suffix']]), axis=1)

em = gen_name_g3 + '_em'
bleu = gen_name_g3 + '_bleu'
met = gen_name_g3 + '_meteor'
roug = gen_name_g3 + '_rougeL'
df_eval_g3[em] = df_eval_g3.progress_apply(lambda x: em_compute( pred= [x[gen_name_g3]], suffix = [x['suffix']]), axis=1)
df_eval_g3[bleu] = df_eval_g3.progress_apply(lambda x: bleu_compute( pred= [x[gen_name_g3]], suffix = [x['suffix']]), axis=1)
df_eval_g3[met] = df_eval_g3.progress_apply(lambda x: meteor_compute( pred= [x[gen_name_g3]], suffix = [x['suffix']]), axis=1)
df_eval_g3[roug] = df_eval_g3.progress_apply(lambda x: rouge_compute( pred= [x[gen_name_g3]], suffix = [x['suffix']]), axis=1)
print("Evaluation metrics computation completed")

# Save the results
df_eval_1.to_parquet(os.path.join(expath, 'attack_mem_dedup_1.parquet'))
df_eval_2.to_parquet(os.path.join(expath, 'attack_mem_dedup_2.parquet'))
df_eval_3.to_parquet(os.path.join(expath, 'attack_mem_dedup_3.parquet'))
df_eval_g3.to_parquet(os.path.join(expath, 'attack_mem_dedup_g3.parquet'))
print("Results saved to attack_mem_dedup_n.parquet")
print(f"----End attack----")