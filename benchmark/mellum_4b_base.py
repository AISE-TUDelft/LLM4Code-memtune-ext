import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from _utils import load_problems, write_generations

# config
set_seed(42)
checkpoint = "JetBrains/Mellum-4b-base"
model_name = "Mellum-4b-base"

print(f"---- HumanEval-X Java — {model_name} ----")

# load problems
problems = load_problems()
task_ids = sorted(problems.keys(), key=lambda x: int(x.split("/")[1]))

# load model
tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

# generate
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, framework='pt', max_new_tokens=512)

outputs = pipe([problems[tid]["prompt"] for tid in task_ids], batch_size=8)
print("Generation done")
torch.cuda.empty_cache()

# save
write_generations(model_name, task_ids, problems, outputs)
print(f"---- End {model_name} ----")