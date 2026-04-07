# HumanEval-X Java Benchmark

Generation of Java completions for the 8 models used in the paper.

## Structure

```
benchmark/
  generate_humaneval_java.py   # orchestrator — runs all 8 models in sequence
  _utils.py                    # shared helpers (load problems, write JSONL)
  sc2_3b_base.py               # bigcode/starcoder2-3b
  sc2_3b_ep3.py                # AISE-TUDelft/StarCoder2Java-3b_ep3
  sc2_7b_base.py               # bigcode/starcoder2-7b
  sc2_7b_ep3.py                # AISE-TUDelft/StarCoder2Java-7b_ep3
  sc2_15b_base.py              # bigcode/starcoder2-15b
  sc2_15b_ep3.py               # AISE-TUDelft/StarCoder2Java-15b_ep3
  mellum_4b_base.py            # JetBrains/Mellum-4b-base
  mellum_4b_ep3.py             # AISE-TUDelft/MellumJava_ep3
  humaneval_java.jsonl.gz      # benchmark data (164 Java problems)
```

## Run all 8 models

```bash
# from the repo root
python benchmark/generate_humaneval_java.py
```

Each model is launched as a **separate subprocess** so the GPU is fully released between runs.
If a script fails the orchestrator stops and reports which one to restart from.

## Run a single model

```bash
python benchmark/sc2_15b_base.py
python benchmark/sc2_15b_ep3.py
```

## Output format

```
benchmark/
  generations/
    starcoder2-3b/generations.jsonl
    StarCoder2Java-3b_ep3/generations.jsonl
    starcoder2-7b/generations.jsonl
    StarCoder2Java-7b_ep3/generations.jsonl
    starcoder2-15b/generations.jsonl
    StarCoder2Java-15b_ep3/generations.jsonl
    Mellum-4b-base/generations.jsonl
    MellumJava_ep3/generations.jsonl
```

Each line of `generations.jsonl`:

```jsonl
{"task_id": "Java/0", "prompt": "<exact prompt>", "generation": "<model output>"}
```

- `task_id` — `"Java/N"` from the dataset
- `prompt` — unmodified string from `humaneval_java.jsonl.gz`
- `generation` — raw model continuation (prompt not repeated)

The evaluator assembles final code as `prompt + generation + "\n" + test`.

## Generation settings

| Parameter | Value |
|---|---|
| `max_new_tokens` | 512 |
| `batch_size` | 8 |
| `torch_dtype` | `bfloat16` |
| `padding_side` | `left` |
| `seed` | 42 |
| Fine-tuned tokenizer | loaded from base checkpoint (same convention as repo experiments) |