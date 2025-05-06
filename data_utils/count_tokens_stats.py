#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
count_tokens_stats.py
Compute token statistics for ShareGPT/OpenAI-style datasets with descriptive metrics.
==============================================================

Dependencies:
    pip install tiktoken transformers numpy tqdm

Usage examples:
    # Count tokens using GPT‑4o tokenizer, display overview and export CSV
    python count_tokens_stats.py r1_interactive_filter.json --model gpt-4o --csv

    # Count tokens using Qwen‑2 tokenizer
    python count_tokens_stats.py r1_interactive_filter.json --model Qwen/Qwen2-7B

==============================================================
Learn more about tiktoken: https://github.com/openai/tiktoken/blob/main/README.md
"""
import argparse, json, statistics, sys
from pathlib import Path
from typing import Dict, List

import numpy as np                              # Descriptive statistics
from tqdm import tqdm                           # Progress bar
import tiktoken                                 # OpenAI tokenizer
from transformers import AutoTokenizer          # HuggingFace tokenizer


# ---------- Get tokenizer ----------
def get_tokenizer(model_name: str):
    """
    1) Try tiktoken.encoding_for_model first
    2) If it fails, fall back to HuggingFace AutoTokenizer
    Returns a callable: tokenizer(text) -> List[int]
    """
    try:
        enc = tiktoken.encoding_for_model(model_name)  # Prefer GPT-family tokenizer
        return lambda txt: enc.encode(txt)
    except KeyError:
        try:
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            return lambda txt: tok.encode(txt)
        except Exception as e:
            print(f"[Error] Failed to load tokenizer for {model_name}: {e}")
            sys.exit(1)


# ---------- Load data ----------
def load_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        yield from (data if isinstance(data, list) else [data])


# ---------- Compute statistics ----------
def compute_stats(values: List[int]) -> Dict[str, float]:
    """Returns min / max / mean / median / std for a list of numbers"""
    if not values:
        return {}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
    }


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input .json or .jsonl file to process")
    parser.add_argument("--model", default="gpt-4o",
                        help="Tokenizer model name (e.g., gpt-4o, gpt-3.5-turbo, Qwen/Qwen2-7B)")
    parser.add_argument("--field", default="value",
                        help="Field name containing message text, default is 'value'")
    parser.add_argument("--csv", action="store_true",
                        help="Save per-conversation statistics to <file>.token_stats.csv")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)

    conv_token_counts = []
    msg_token_counts  = []
    total_tokens      = 0
    total_messages    = 0

    # Iterate over conversations
    for conv in tqdm(load_records(Path(args.file)), desc="Convs", unit="conv"):
        conv_total = 0
        for msg in conv.get("conversations", []):
            text = msg.get(args.field, "")
            tokens = len(tokenizer(text))
            msg_token_counts.append(tokens)
            conv_total += tokens
            total_tokens += tokens
            total_messages += 1
        conv_token_counts.append(conv_total)

    # ----- Summary -----
    conv_stats = compute_stats(conv_token_counts)
    msg_stats  = compute_stats(msg_token_counts)

    print("\n========== Token Statistics ==========")
    print(f"Number of conversations : {len(conv_token_counts):,}")
    print(f"Number of messages      : {total_messages:,}")
    print(f"Total token count       : {total_tokens:,}\n")

    print("Token stats per conversation:")
    for k, v in conv_stats.items():
        print(f"  {k:<6}: {v:,.2f}")

    print("\nToken stats per message:")
    for k, v in msg_stats.items():
        print(f"  {k:<6}: {v:,.2f}")

    # ----- Optional CSV output -----
    if args.csv:
        import csv
        csv_path = Path(args.file).with_suffix(".token_stats.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["conversation_index", "tokens"])
            for i, tks in enumerate(conv_token_counts):
                writer.writerow([i, tks])
        print(f"\nPer-conversation statistics saved to {csv_path}")


if __name__ == "__main__":
    main()


