#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference_multi.py
Multi-GPU Inference with Progress Bar
==========================================================

Parallelism:
- Data parallel: 🤗 Accelerate
- Model parallel / memory slicing: device_map="auto"

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch inference_multi.py \
  --model /path/to/your_model \
  --input /path/to/input.json \
  --output /path/to/output.json \
  --prompt_max_len 4096 \
  --max_new_tokens 512 \
  --batch_size 4 \
  --bf16
"""
import argparse, json, math
from pathlib import Path
from typing import List

import torch
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedType
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)

# ----------------------- Model Loading --------------------------------------
def load_model(model_name: str, bf16: bool, load_8bit: bool):
    dtype = torch.bfloat16 if bf16 else torch.float16
    q_cfg = BitsAndBytesConfig(load_in_8bit=True) if load_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",          # Layer slicing across GPUs within a process
        low_cpu_mem_usage=True,
        quantization_config=q_cfg,
        trust_remote_code=True,
    ).eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tok

def wrap_prompt(tok, text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True)

# ----------------------- Main Logic -----------------------------------------
def main(args):
    # 0) Initialize Accelerator
    accelerator = Accelerator()                       # Transparent multi-process management
    is_main = accelerator.is_main_process

    # 1) Load data on the main process, broadcast to others
    if is_main:
        recs = json.loads(Path(args.input).read_text(encoding="utf-8"))
        prompts = [r["input"] for r in recs]
    else:
        recs, prompts = None, None
    prompts = accelerator.broadcast_object(prompts, src=0)

    # 2) Load model and tokenizer
    model, tok = load_model(args.model, args.bf16, args.load_8bit)
    model = accelerator.prepare(model)  # Move to the correct device

    # 3) Data parallel slicing: each process handles its own prompts
    world = accelerator.num_processes
    rank  = accelerator.process_index
    local_prompts = prompts[rank :: world]   # Simple round-robin slicing

    if is_main:
        pbar = tqdm(total=len(prompts), ncols=90,
                    desc="Generating", unit="sample")

    bs = args.batch_size
    n_batches = math.ceil(len(local_prompts) / bs)
    local_preds: List[str] = []

    for i in range(n_batches):
        chunk = local_prompts[i*bs:(i+1)*bs]
        if not chunk:
            break
        templ  = [wrap_prompt(tok, p) for p in chunk]
        batch  = tok(templ, return_tensors="pt",
                     padding=True, truncation=True,
                     max_length=args.prompt_max_len).to(model.device)

        with torch.inference_mode(), torch.cuda.amp.autocast():
            gen_ids = model.generate(
                **batch,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

        new_ids = gen_ids[:, batch.input_ids.shape[1]:]
        outs = tok.batch_decode(new_ids, skip_special_tokens=True)
        local_preds.extend([o.strip() for o in outs])

        if is_main:
            pbar.update(len(chunk))

    if is_main:
        pbar.close()

    # 4) Gather predictions from all processes
    gathered = accelerator.gather(local_preds)   # list[list[str]]
    preds = [o for sub in gathered for o in sub]

    # 5) Save results on the main process
    if is_main:
        for r, o in zip(recs, preds):
            r["predict_output"] = o
        Path(args.output).write_text(
            json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✓ {len(recs)} results saved to {args.output}")

# ----------------------- CLI ------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--input", required=True)
    pa.add_argument("--output", required=True)
    pa.add_argument("--model",  default="Qwen/Qwen2.5-7B-Instruct")
    pa.add_argument("--batch_size", type=int, default=4)
    pa.add_argument("--prompt_max_len", type=int, default=1024)
    pa.add_argument("--max_new_tokens", type=int, default=512)
    pa.add_argument("--bf16", action="store_true")
    pa.add_argument("--load_8bit", action="store_true")
    pa.add_argument("--greedy", action="store_true")
    pa.add_argument("--temperature", type=float, default=0.7)
    pa.add_argument("--top_p", type=float, default=0.9)
    pa.add_argument("--repetition_penalty", type=float, default=1.0)
    args = pa.parse_args()
    main(args)
