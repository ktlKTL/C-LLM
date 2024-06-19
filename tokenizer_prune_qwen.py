import json

from transformers import AutoTokenizer
from calcuate_metric import is_chinese_string


def prune():

    tokenizer = AutoTokenizer.from_pretrained('./models/Qwen1.5-0.5B', use_fast=False)
    
    need_to_delete_words = set()
    need_to_delete_word_ids = set()
    for word, index in tokenizer.encoder.items():
        word = bytearray([tokenizer.byte_decoder[c] for c in word]).decode("utf-8", errors=tokenizer.errors)
        if len(word) > 1 and is_chinese_string(word):
            need_to_delete_words.add(word)
            need_to_delete_word_ids.add(index)
            print(f'DELETE WORD {word}')
    print(len(need_to_delete_word_ids), 'chinese word need to be delete')
    need_to_delete_merge_ids = set()

    for merge, index in tokenizer.bpe_ranks.items():
        a, b = merge
        _a = bytearray([tokenizer.byte_decoder[c] for c in a]).decode("utf-8", errors=tokenizer.errors)
        _b = bytearray([tokenizer.byte_decoder[c] for c in b]).decode("utf-8", errors=tokenizer.errors)
        _a_b = bytearray([tokenizer.byte_decoder[c] for c in a + b]).decode("utf-8", errors=tokenizer.errors)
        if _a_b in need_to_delete_words or _a in need_to_delete_words or _b in need_to_delete_words:
            print(f'DELETE MERGE {_a_b}')
            need_to_delete_merge_ids.add(index)
    print(len(need_to_delete_merge_ids), 'merges need to be delete')
    print(len(need_to_delete_merge_ids), 'merges need to be delete')
    new_vocab = dict()
    for word, index in tokenizer.encoder.items():
        if index not in need_to_delete_word_ids:
            new_vocab[word] = len(new_vocab)
  
    with open('./models/qwen1.5_0.5b_new/vocab.json', 'wt', encoding='utf-8') as f:
        json.dump(new_vocab, f)
    new_merges = []
    for merge, index in tokenizer.bpe_ranks.items():
        if index not in need_to_delete_merge_ids:
            new_merges.append(" ".join(merge))

    with open('./models/qwen1.5_0.5b_new/merges.txt', 'wt', encoding='utf-8') as f:
        f.write("\n".join(new_merges))
        f.write("\n")


def main():
    text = '我和我的祖国'
    tokenizer = AutoTokenizer.from_pretrained('./models/Qwen1.5-0.5B', use_fast=False)
    print([tokenizer.decode(_id) for _id in tokenizer.encode(text)])
    print(len(tokenizer))    
    tokenizer = AutoTokenizer.from_pretrained('./models/qwen1.5_0.5b_new', use_fast=False)
    print([tokenizer.decode(_id) for _id in tokenizer.encode(text)])
    print(len(tokenizer))


if __name__ == '__main__':
    
    prune()
    # main()