#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference_single.py
Run inference with Transformers + real-time progress bar
==========================================================================

Example usage:

CUDA_VISIBLE_DEVICES=1 python inference_single.py \
  --model /path/to/your_model \
  --input /path/to/input.json \
  --output /path/to/output.json \
  --prompt_max_len 4096 \
  --max_new_tokens 512 \
  --bf16 \
  --batch_size 1
"""
import argparse, json, math, sys, torch
from pathlib import Path
from typing import List
from tqdm.auto import tqdm                    # Real-time progress bar
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)


# ---------- Model Loading --------------------------------------------------
def load_model(model_name: str, bf16: bool, load_8bit: bool):
    dtype = torch.bfloat16 if bf16 else torch.float16
    q_cfg = BitsAndBytesConfig(load_in_8bit=True) if load_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        quantization_config=q_cfg,
        trust_remote_code=True,
    ).eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tok

def wrap_prompt(tok, text: str) -> str:
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs,
                                   tokenize=False,
                                   add_generation_prompt=True)


# ---------- Main Logic -----------------------------------------------------
def main(a):
    recs = json.loads(Path(a.input).read_text(encoding="utf-8"))
    prompts = [r["input"] for r in recs]

    model, tok = load_model(a.model, a.bf16, a.load_8bit)

    bs = a.batch_size
    n_batches = math.ceil(len(prompts) / bs)
    preds: List[str] = []

    pbar = tqdm(total=len(prompts), ncols=90,
                desc="Generating", unit="sample")

    for i in range(n_batches):
        chunk = prompts[i*bs: (i+1)*bs]
        templ = [wrap_prompt(tok, p) for p in chunk]

        batch = tok(templ,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=a.prompt_max_len).to(model.device)

        with torch.inference_mode(), torch.cuda.amp.autocast():
            gen = model.generate(
                **batch,
                max_new_tokens=a.max_new_tokens,
                do_sample=not a.greedy,
                temperature=a.temperature,
                top_p=a.top_p,
                repetition_penalty=a.repetition_penalty,
            )

        new_ids = gen[:, batch.input_ids.shape[1]:]
        outs = tok.batch_decode(new_ids, skip_special_tokens=True)
        outs = [o.strip() for o in outs]

        preds.extend(outs)
        pbar.update(len(chunk))               # Update progress bar in real time
        if a.verbose:
            for o in outs:
                print(o, flush=True)

    pbar.close()

    # Write results
    for r, o in zip(recs, preds):
        r["predict_output"] = o
    Path(a.output).write_text(
        json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ Finished. Results saved to {a.output}")


# ---------- CLI Entry ------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--input",  required=True)
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
    pa.add_argument("--verbose", action="store_true",
                    help="print each generated text")
    args = pa.parse_args()
    main(args)



