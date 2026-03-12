import gzip
import json
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
DATA_FILE = BENCHMARK_DIR / "humaneval_java.jsonl.gz"


def load_problems(path=DATA_FILE):
    problems = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                problems[p["task_id"]] = p
    return problems


def write_generations(model_name, task_ids, problems, outputs):
    out_dir = BENCHMARK_DIR / "generations" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "generations.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for tid, out in zip(task_ids, outputs):
            prompt = problems[tid]["prompt"]
            generation = out[0]["generated_text"][len(prompt):]
            f.write(json.dumps({"task_id": tid, "prompt": prompt, "generation": generation}, ensure_ascii=False) + "\n")

    print(f"Results saved to {out_file}")