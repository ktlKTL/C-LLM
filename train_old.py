from dataclasses import dataclass, field
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import HfArgumentParser, TrainingArguments, AutoModelForCausalLM, Trainer, AutoTokenizer
from trl.trainer.utils import PeftSavingCallback
from llm_train_util import prepare_model_for_training, SFTDataCollector
from transformers.trainer_utils import get_last_checkpoint
import os

@dataclass
class FinetuneArguments:
    """
    微调参数
    """
    pretrained_model_path: str = field()
    train_dataset_path: str = field()
    eval_dataset_path: str = field()
    pad_token_id: int = field(default=0)
    lora_rank: int = field(default=16)
    lora_alpha: float = field(default=32.0)
    lora_dropout: float = field(default=0.1)
    lora_target: str = field(default="W_pack")
    ft_type: str = field(default="lora")


def create_and_prepare_dataset(data_path):
    """
    创建数据
    :return:
    """
    train_dataset = load_dataset("json", data_files=data_path)

    def preprocess_function(example):
        """
        预处理
        :param example:
        :return:
        """
        prompt = f"任务: 纠错文本\n输入: {example['src']}\n输出: "
        response = example['tgt']
        return {
            'prompt': prompt,
            'response': response,
        }

    train_dataset = train_dataset.map(preprocess_function, batched=False)
    return train_dataset['train']


def train():
    """
    训练模型
    :return:
    """
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)).parse_args_into_dataclasses()
    # load model
    model = AutoModelForCausalLM.from_pretrained(finetune_args.pretrained_model_path,
                                                 trust_remote_code=True)
    if finetune_args.ft_type == 'lora':
        model = prepare_model_for_training(model)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=finetune_args.lora_alpha,
            lora_dropout=finetune_args.lora_dropout,
            target_modules=[target.strip() for target in finetune_args.lora_target.split(",")]
        )
        model = get_peft_model(model, lora_config)
    # load tokenizer
    if 'Lang16' in finetune_args.pretrained_model_path:
        import sys
        sys.path.append(finetune_args.pretrained_model_path)
        from tokenization_hackt5 import HackT5TokenizerFast
        tokenizer = HackT5TokenizerFast.from_pretrained(finetune_args.pretrained_model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(finetune_args.pretrained_model_path, trust_remote_code=True, use_fast=False)
    if 'Baichuan' in finetune_args.pretrained_model_path:
        tokenizer.pad_token_id = 0
    elif 'Qwen' in finetune_args.pretrained_model_path:
        tokenizer.pad_token_id = 151643
        tokenizer.eos_token_id = 151643
    elif 'internlm' in finetune_args.pretrained_model_path:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif 'Skywork' in finetune_args.pretrained_model_path:
        tokenizer.pad_token_id = 0
    # load dataset
    train_dataset = create_and_prepare_dataset(finetune_args.train_dataset_path)
    eval_dataset = create_and_prepare_dataset(finetune_args.eval_dataset_path)
    # start train
    training_args.ddp_find_unused_parameters = False
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=SFTDataCollector(tokenizer),
        callbacks=[PeftSavingCallback()] if finetune_args.ft_type == 'lora' else None,
    )
    trainer.train()


if __name__ == '__main__':
    train()
