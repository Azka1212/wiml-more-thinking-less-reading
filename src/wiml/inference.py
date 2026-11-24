"""Qwen3 inference. Importing this module requires the GPU dependencies.

The package's offline analysis never imports this module.
"""

import random

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor,
)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_qwen_thinking(output_ids, tokenizer):
    ids = output_ids.tolist() if hasattr(output_ids, "tolist") else list(output_ids)
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    valid = (
        end_think_id is not None
        and end_think_id != tokenizer.unk_token_id
        and end_think_id in ids
    )
    if valid:
        idx = len(ids) - 1 - ids[::-1].index(end_think_id)
        thinking_ids, final_ids = ids[:idx + 1], ids[idx + 1:]
    else:
        thinking_ids, final_ids = [], ids
    thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
    final = tokenizer.decode(final_ids, skip_special_tokens=True).strip()
    return thinking, final


class FiniteLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Model produced non-finite generation logits.")
        return scores


class QwenGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        revision = {"revision": cfg.model_revision} if cfg.model_revision else {}
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, **revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        kwargs = {"dtype": dtype, "attn_implementation": "eager", **revision}
        if cfg.use_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit Qwen3 loading requires CUDA.")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        elif torch.cuda.is_available():
            kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
        self.model.eval()
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def __call__(self, prompt, enable_thinking, seed, sample=True):
        set_all_seeds(seed)
        cfg, model, tokenizer = self.cfg, self.model, self.tokenizer
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        batch = tokenizer(rendered, return_tensors="pt")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        kwargs = {
            "do_sample": sample,
            "max_new_tokens": cfg.max_new_tokens_think if enable_thinking else cfg.max_new_tokens_direct,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "logits_processor": [FiniteLogitsProcessor()],
            "renormalize_logits": True,
        }
        if sample:
            kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)
        generated = model.generate(**batch, **kwargs)
        new_ids = generated[0, batch["input_ids"].shape[1]:]
        thinking, final = split_qwen_thinking(new_ids, tokenizer)
        return {
            "thinking": thinking,
            "final": final,
            "generated_tokens": int(len(new_ids)),
            "thinking_chars": len(thinking),
        }
