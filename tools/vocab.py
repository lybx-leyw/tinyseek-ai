"""
本工具用于生成词表
"""

import torch
import random
import json
import numpy as np
from tools.fast_print import print_processing
from tools.tokenizer import Tokenizer
from collections import Counter
import torch.nn.functional as F

if __name__ == '__main__':
    # 做个尝试
    text = "只是一个小尝试罢了|end"
    tokens = Tokenizer.tokenize(text)   
    print(tokens)

class Vocab():
    def __init__(self, json_text_path, json_vocab_path, vocab_size, JIEBA=False):
        self.text_path = json_text_path
        self.vocab_path = json_vocab_path
        self.vocab_dict = {}
        self.vocab_size = vocab_size
        try:
            self.load_vocab(json_vocab_path)
            print_processing("词表加载成功！")
        except:
            print_processing("词表创建中...")
            self.create_vocab(json_text_path, json_vocab_path, vocab_size, JIEBA)
            print_processing("词表创建成功！")

    def create_vocab(self, json_data_path, json_vocab_path, max_size=5000, JIEBA=False):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        all_tokens = []
        freq_dict = Counter()
        with open(json_data_path, "r", encoding='utf-8') as data_file:
            for line_index, line in enumerate(data_file, 1):
                data = json.loads(line)
                conversations = data['text'].split('<|im_end|>')
                for item in conversations:
                    tokens = Tokenizer.tokenize(item, JIEBA=JIEBA)
                    all_tokens.extend(tokens)
                if line_index % 3000 == 0:
                    freq_dict.update(all_tokens)
                    all_tokens = []
                    print(f"第{line_index}行单词记录完毕") 

        sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:max_size]
        freq_dict = dict(sorted_items)

        # 先添加特殊符号
        prior_list = ['[PAD]','<|im_end|>','[MASK]','[UNK]','TinySeek', 
                      'tinyseek','TinySeek','tinySeek','Tinyseek','TINYSEEK','User',
                      'Assistant','[unused9]','[unused10]']
        vocab_dict = {}
        
        # 添加特殊符号到词表
        for index, item in enumerate(prior_list):
            vocab_dict[item] = index
        
        # 添加高频词
        add = len(prior_list)
        for index, key in enumerate(freq_dict.keys()):
            if index + add >= max_size:
                break
            vocab_dict[key] = index + add
        
        # 写入JSON文件
        with open(json_vocab_path, "w", encoding='utf-8') as tag_file:
            json.dump(vocab_dict, tag_file, ensure_ascii=False, indent=2)

        print(f"词表创建完毕。共有词{len(vocab_dict)}条")
        self.vocab_dict = vocab_dict
    
    def load_vocab(self, vocab_path):
        with open(vocab_path, "r", encoding='utf-8') as vocab_file:
            vocab_dict = json.load(vocab_file) 
            self.vocab_dict = vocab_dict
    
    def encode(self,text,max_len):
        tokens = Tokenizer.tokenize(text)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        ids = [self.vocab_dict.get(key,self.vocab_dict['[UNK]']) for key in tokens]
        while len(ids) < max_len:
            ids.append(self.vocab_dict['[PAD]'])
        return torch.tensor(ids).unsqueeze(0)
    
    def decode(self,ids):
        reverse_dict = {v: k for k, v in self.vocab_dict.items()}
        tokens = [reverse_dict.get(id,'[UNK]') for id in ids]
        return tokens

    def generate(self, model, input_tensor, config, seq_len, max_len, max_gen_len, repetition_penalty=2, FULL_MODEL=False,
                 gate_input = None, roll_idx = 0, PRINT = False):
        model = model.to(config['device'])
        if PRINT:
            print("\n\033[32mAssistant：\033[0m", end="", flush=True)
        if gate_input != None:
            current_len = seq_len-roll_idx
        generated_ids = input_tensor.clone().to(config['device'])
        keep_cnt = 0
        generate_index = 0
        
        is_first_token = True
        for _ in range(min(max_len - seq_len, max_gen_len)):
            generated_ids = generated_ids.to(config['device'])
            if FULL_MODEL == False:
                c_t, n_t, _ = model(generated_ids)
            else:
                c_t, n_t, _ = model(generated_ids,gate_input.to(config['device']),roll_idx)
            generated_so_far = generated_ids[0, seq_len:seq_len+generate_index-1]
            generated_so_far = generated_so_far[generated_so_far != config['pad_idx']]

            probs_1_vals = F.softmax(c_t[:, seq_len+generate_index-1, :], dim=-1)            
            probs_2_vals = F.softmax(n_t[:, seq_len+generate_index-1, :], dim=-1)

            generated_so_far = generated_ids[0, seq_len:seq_len+generate_index]
        
            for token_id in generated_so_far:
                token_id = token_id.item()
                if probs_1_vals[0,token_id] > 0:
                    probs_1_vals[0,token_id] /= repetition_penalty
                if probs_2_vals[0,token_id] > 0:
                    probs_2_vals[0,token_id] /= repetition_penalty

            if is_first_token and generate_index == 0:
                top_probs_1, top_indices_1 = torch.topk(probs_1_vals[0], 2)
                normalized_probs = top_probs_1 / top_probs_1.sum()
                chosen_index = torch.multinomial(normalized_probs, 1).item()
                next_token_1 = top_indices_1[chosen_index].unsqueeze(0).unsqueeze(0)
                
                is_first_token = False
            else:
                next_token_1 = torch.argmax(probs_1_vals[:, :], dim=-1).unsqueeze(0)
            
            next_token_2 = torch.argmax(probs_2_vals[:, :], dim=-1).unsqueeze(0)

            generated_ids[0, seq_len+generate_index] = next_token_1
            gate_input[0, current_len+generate_index] = next_token_1
            
            # 如果生成了特殊id就停止
            if next_token_1.item() == 0 or next_token_1.item() == 1 or next_token_1.item() == 2 or \
                next_token_1.item() == 9 or next_token_1.item() == 10:
                break
            if PRINT:
                token_text = self.decode(next_token_1.squeeze(0).tolist())
                print("\033[32m" + token_text[0] + "\033[0m", end="", flush=True)

            if probs_2_vals.max().item() > 0.8 and generate_index != 1:
                keep_cnt += 1
                generate_index += 1
                generated_ids[0, seq_len + generate_index] = next_token_2
                gate_input[0, current_len+generate_index] = next_token_2
                if next_token_2.item() == 0 or next_token_2.item() == 1 or next_token_2.item() == 2 or \
                    next_token_2.item() == 9 or next_token_2.item() == 10:
                    break
                if PRINT:
                    token_text = self.decode(next_token_2.squeeze(0).tolist())
                    print("\033[32m" + token_text[0] + "\033[0m", end="", flush=True)

            generate_index += 1
        
        generated_ids = generated_ids[0, :seq_len + generate_index]
        generated_text = self.decode(generated_ids.squeeze(0).tolist())
        if PRINT:
            print()
        return generated_text, keep_cnt