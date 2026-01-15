# scripts/pretokenize_jsonl_mp.py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tiktoken
import multiprocessing as mp


def process_one_jsonl(args):
    jsonl_path, out_dir, enc_name, eos_token_id = args

    tokenizer = tiktoken.get_encoding(enc_name)

    out_path = out_dir / (jsonl_path.stem + ".bin")
    if out_path.exists():
        print(f"[skip] {out_path}")
        return

    print(f"[processing] {jsonl_path.name}")
    all_tokens = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
                if not text:
                    continue
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
                all_tokens.append(eos_token_id)
            except Exception:
                continue

    arr = np.array(all_tokens, dtype=np.uint32) ## 这里尤其需要注意dtype, 这个不会影响模型的效果, 是根据tokenizer的vocab_size来定的, 定小了会有溢出风险
    arr.tofile(out_path)

    print(f"[saved] {out_path} | tokens={len(arr):,}")


def pretokenize_jsonl_dir_mp(jsonl_dir, out_dir, enc_name, eos_token_id, num_workers):
    jsonl_dir = Path(jsonl_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(jsonl_dir.glob("*.jsonl"))

    tasks = [
        (path, out_dir, enc_name, eos_token_id)
        for path in jsonl_files
    ]

    with mp.Pool(processes=num_workers) as pool:
        pool.map(process_one_jsonl, tasks)


if __name__ == "__main__":
    enc_name = "cl100k_base"
    tokenizer = tiktoken.get_encoding(enc_name)

    pretokenize_jsonl_dir_mp(
        jsonl_dir="data/CCI3/original_data/val",
        out_dir="data/CCI3/data_bin/val",
        enc_name=enc_name,
        eos_token_id=tokenizer.eot_token,
        num_workers=2,  # 👈 根据 CPU 核数调
    )
