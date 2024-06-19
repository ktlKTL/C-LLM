import os.path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
from torch import nn

class VocabularyPruner(object):

    def check(self, old_model_name_or_path, new_model_name_or_path, text):
        # 检查模型裁剪后，生成结果是否一致
        max_length = 20

        # 使用老模型对文本编码
        old_model = AutoModelForCausalLM.from_pretrained(old_model_name_or_path,trust_remote_code=True)
        old_tokenizer = AutoTokenizer.from_pretrained(old_model_name_or_path,trust_remote_code=True,use_fast=False)
        old_input_ids = old_tokenizer(text, return_tensors='pt').input_ids
        old_output = old_model.generate(old_input_ids, max_length=max_length, do_sample=False, num_beams=1)
        old_output_text = old_tokenizer.batch_decode(old_output)
        print('old_output:{}'.format(old_output_text))

        # 使用新模型对文本编码
        new_model = AutoModelForCausalLM.from_pretrained(new_model_name_or_path,trust_remote_code=True)
        new_tokenizer = AutoTokenizer.from_pretrained(new_model_name_or_path,trust_remote_code=True,use_fast=False)
        new_input_ids = new_tokenizer(text, return_tensors='pt').input_ids
        new_output = new_model.generate(new_input_ids, max_length=max_length, do_sample=False, num_beams=1)
        new_output_text = new_tokenizer.batch_decode(new_output)
        print('new_output:{}'.format(new_output_text))

        if old_output_text == new_output_text:
            print('output is same, succeed to prune.')
        else:
            print('output is not same, fail to prune.')

    def check_embedding(self, old_model_name_or_path, new_model_name_or_path, text):
        # 检查模型裁剪后，生成结果是否一致
        max_length = 20

        # 使用老模型对文本编码
        old_model = AutoModelForCausalLM.from_pretrained(old_model_name_or_path,trust_remote_code=True)
        old_tokenizer = AutoTokenizer.from_pretrained(old_model_name_or_path,trust_remote_code=True,use_fast=False)
        old_token = old_tokenizer.tokenize(text)
        print('old_token:{}'.format(old_token))
        
        old_input_ids = old_tokenizer(text, return_tensors='pt').input_ids

        print('old_input_ids:{}'.format(old_input_ids))

        old_model_input_embed = old_model.get_input_embeddings().weight.data[old_tokenizer(text, return_tensors='pt').input_ids]
        print("old_model_input_embed: ",old_model_input_embed)
        old_model_output_embed = old_model.get_output_embeddings().weight.data[old_tokenizer(text, return_tensors='pt').input_ids]
        print("old_model_output_embed: ", old_model_output_embed)

        # 使用新模型对文本编码
        new_model = AutoModelForCausalLM.from_pretrained(new_model_name_or_path,trust_remote_code=True)
        new_tokenizer = AutoTokenizer.from_pretrained(new_model_name_or_path,trust_remote_code=True,use_fast=False)
        new_token = new_tokenizer.tokenize(text)
        print('new_token:{}'.format(new_token))
        
        new_input_ids = new_tokenizer(text, return_tensors='pt').input_ids
        print('new_input_ids:{}'.format(new_input_ids))

        new_model_input_embed = new_model.get_input_embeddings().weight.data[new_tokenizer(text, return_tensors='pt').input_ids]
        print("new_model_input_embed: ",new_model_input_embed)
        new_model_output_embed = new_model.get_output_embeddings().weight.data[new_tokenizer(text, return_tensors='pt').input_ids]
        print("new_model_output_embed: ",new_model_output_embed)

        print("equal? ", all((old_model_input_embed == new_model_input_embed)[0][0]), all((old_model_output_embed == new_model_output_embed)[0][0]))
          
    
    def update_ebeddings(self, model, new2old_token_id, new_embeds, new_lm_head):
        raise NotImplemented

    def prune(self, model_name_or_path, new_tokenizer_name_or_path, save_path, new_name_or_path=None):
        # 创建输出目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 加载新词表
        new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_name_or_path, trust_remote_code=True,use_fast=False)
        # 加载原词表
        old_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True,use_fast=False)

        print("new_tokenizer:",len(new_tokenizer)," old_tokenizer: ",len(old_tokenizer))
        # 检查新词表是否为原词表的子集
        old_vocab = old_tokenizer.get_vocab()
        new_vocab = new_tokenizer.get_vocab()
        for token in tqdm(new_vocab.keys()):
            if token not in old_vocab:
                raise Exception('{} not exist'.format(token))
        print('new_tokenizer is subset of old_tokenizer')

        # 获得新词表中每个token_id到原词表的token_id的映射
        new2old_token_id = {}
        for token, token_id in tqdm(new_vocab.items()):
            old_token_id = old_vocab[token]
            new2old_token_id[token_id] = old_token_id

    
        for i in range(0,290): new2old_token_id[135272+i] = 151646 + i

        # 加载多语言模型
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype='auto', trust_remote_code=True)
        # 计算原模型的参数量
        old_params = sum(p.numel() for p in model.parameters())
        print("Total params of original model: %.2fM" % (old_params / 1e6))

        # 对于新词表中的每个token，取出其对应的权重，复制到新模型中
        vocab_size = len(new2old_token_id)
        hidden_size = model.config.hidden_size

        new_embeds = torch.nn.Embedding(vocab_size, hidden_size, dtype=model.dtype)
        new_lm_head = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=False, dtype=model.dtype)
        
        
        # 更新词表权重
        self.update_ebeddings(model, new2old_token_id, new_embeds, new_lm_head)

        model.config.__dict__['vocab_size'] = vocab_size
        if new_name_or_path is not None:
            model.config.__dict__['_name_or_path'] = new_name_or_path

        # 计算新模型的参数量
        new_params = sum(p.numel() for p in model.parameters())
        print("Total params of new model : %.2fM" % (new_params / 1e6))

        print('词表缩小为原来的:{}%'.format(round(len(new_tokenizer) / len(old_tokenizer), 4)*100))
        print('模型参数量缩小为原来的:{}%'.format(round(new_params / old_params, 4)*100))
        model.save_pretrained(save_path)
        new_tokenizer.save_pretrained(save_path)
        
        fin = open("new2old_token_id.json", "w", encoding="utf-8")
        fin.write(json.dumps(new2old_token_id, indent=4, ensure_ascii=False))
        fin.close()
        


class ModelVocabularyPruner(VocabularyPruner):

    def update_ebeddings(self, model, new2old_token_id, new_embeds, new_lm_head):
        for token_id, old_token_id in tqdm(new2old_token_id.items()):
            new_embeds.weight.data[token_id] = model.get_input_embeddings().weight.data[old_token_id]
            new_lm_head.weight.data[token_id] = model.get_output_embeddings().weight.data[old_token_id]
                
        model.set_input_embeddings(new_embeds)
        model.set_output_embeddings(new_lm_head)
        


if __name__ == "__main__":

    # 需要进行裁剪的模型路径
    model_name_or_path =  "./models/Qwen1.5-14B/"
    
    # 自己制作的词表的路
    new_tokenizer_name_or_path = './models/qwen1.5_new_tokenizer'
    save_path = './models/qwen1.5_14b_filter_model'
    pruner = ModelVocabularyPruner()
    
    # 裁剪
    pruner.prune(model_name_or_path, new_tokenizer_name_or_path, save_path)

    # 检查裁剪的模型与原模型是否一致
    pruner.check(model_name_or_path, save_path, text='长风破浪会有时')
    pruner.check_embedding(model_name_or_path, save_path, text='项伤是速')
