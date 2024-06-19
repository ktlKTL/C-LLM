#!/usr/bin/env python
# encoding: utf-8
"""
File Description:
Author: rightyonghu
Created Time: 2023/10/25
"""
from typing import Optional, List
import torch
import numpy as np
import evaluate


def prepare_model_for_training(
        model,
        output_layer_name: Optional[str] = "lm_head",
        layer_norm_names: Optional[List[str]] = ["norm", "ln_f", "ln_attn", "ln_mlp"]
):
    """
    https://github.com/hiyouga/LLaMA-Efficient-Tuning/blob/main/src/llmtuner/extras/misc.py#L91
    Includes:
    (1) cast the layernorm in fp32
    (2) make output embedding layer require grads
    (3) upcast the lm_head to fp32
    Inspired by: https://github.com/huggingface/peft/blob/c0209c35abbf88c63aa267800d98a8e212ed0a42/src/peft/utils/other.py#L35
    :return:
    """
    for name, param in model.named_parameters():
        param.requires_grad = False
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_layer_name in name:
            param.data = param.data.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    return model


class SFTDataCollector:
    """
    SFT data collector
    1. mask prompt
    2. mask pad
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        prompt_ids_list = [self.tokenizer.encode(each['prompt'], add_special_tokens=False) for each in features]
        target_ids_list = [self.tokenizer.encode(each['response'], add_special_tokens=False) for each in features]
        input_ids_list = [p + t + [self.tokenizer.eos_token_id] for p, t in zip(prompt_ids_list, target_ids_list)]
        prompt_lens = [len(prompt_ids) for prompt_ids in prompt_ids_list]
        input_lens = [len(input_ids) for input_ids in input_ids_list]
        max_seq_len = max(input_lens)
        processed_input_ids_list = []
        processed_label_ids_list = []
        for input_ids, input_len, prompt_len in zip(input_ids_list, input_lens, prompt_lens):
            processed_input_ids = input_ids + [self.tokenizer.pad_token_id] * (max_seq_len - input_len)
            processed_input_ids_list.append(torch.LongTensor(processed_input_ids))
            processed_label_ids = [-100] * prompt_len + input_ids[prompt_len:] + [-100] * (max_seq_len - input_len)
            processed_label_ids_list.append(torch.LongTensor(processed_label_ids))
        input_ids = torch.stack(processed_input_ids_list)
        labels = torch.stack(processed_label_ids_list)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class SEQCLSDataCollector:
    """
    句子级二分类
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        texts = [f['query'] for f in features]
        labels = [f['label'] for f in features]
        tokenized = self.tokenizer(texts, padding=True, return_tensors='pt')
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        labels = torch.LongTensor(labels)
        return {
            "input_ids": input_ids,
            'attention_mask': attention_mask,
            "labels": labels,
        }


class BinaryClassificationMetric:
    """
    二分类评估
    """

    def __init__(self):
        self.metric = evaluate.load("accuracy")

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)
