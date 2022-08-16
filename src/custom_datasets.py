from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from imblearn.over_sampling import RandomOverSampler

import os
import json
import pandas as pd
import numpy as np

from typing import Dict

tokenizer_dict = {
    'bert': AutoTokenizer.from_pretrained("bert-base-cased"),
    'roberta': AutoTokenizer.from_pretrained("roberta-base") 
}

class boolq_dataset_train(Dataset):
    def __init__(self, config=None, train=True, max_rows=None):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        with open(os.path.join(config['datapath'], config['data']['name'], config['data']['type'], config['data']['filename']), 'r') as handle:
            self.data = json.load(handle)
        
        print('total trianing data:',len(self.data))
        question, passage, answer = [], [], []
        for i in self.data:
            question.append(i['question'])
            passage.append(i['passage'])
            answer.append(i['answer'])
        self.df = pd.DataFrame({
            'question': question,
            'passage': passage,
            'answer': answer
        })
        print('total trianing data:',len(self.df))
        self.df = self.df.iloc[np.random.permutation(len(self.df))].reset_index()
        over_sampler = RandomOverSampler(random_state=42)
        X_res, y_res = over_sampler.fit_resample(self.df, self.df['answer'])
        
        
        question, passage, answer = [], [], []
        for i in range(len(X_res)):
            question.append(str(X_res['question'][i]))
            passage.append(str(X_res['passage'][i]))
            answer.append(bool(X_res['answer'][i]))
        self.df = {
            'question': question,
            'passage': passage,
            'answer': answer
        }
        print('total trianing data:',len(self.df))

        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
                   
        tmp = self.tokenizer(self.df['question'], self.df['passage'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.df_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.df['answer']
            }
        elif self.config['model']=='roberta':
            self.df_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.df['answer']
            }
        else:
            raise
        
    
    
    def __getitem__(self, index):
        
        answer = self.df_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.df_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.df_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.df_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.df_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.df_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return len(self.df['question'])

class boolq_dataset_val(Dataset):
    def __init__(self, config=None, train=True, max_rows=None):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        if train:
            self.boolq = load_dataset('boolq')['train'].shuffle(seed=42)
        else:
            self.boolq = load_dataset('boolq')['validation'].shuffle(seed=42)
            
        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
        
        tmp = self.tokenizer(self.boolq['question'], self.boolq['passage'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.boolq['answer']
            }
        elif self.config['model']=='roberta':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.boolq['answer']
            }
        else:
            raise

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        answer = self.boolq_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.boolq_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return self.boolq.num_rows

class boolq_dataset_train_org(Dataset):
    def __init__(self, config=None, train=True, max_rows=345):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        self.boolq = load_dataset('boolq')['train'].shuffle(seed=42)
            
        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
        
        tmp = self.tokenizer(self.boolq['question'], self.boolq['passage'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.boolq['answer']
            }
        elif self.config['model']=='roberta':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.boolq['answer']
            }
        else:
            raise

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        answer = self.boolq_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.boolq_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        if self.max_rows:
            return self.max_rows
        return self.boolq.num_rows

class cb_dataset_train(Dataset):
    def __init__(self, config=None, train=True, max_rows=None):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        with open(os.path.join(config['datapath'], config['data']['name'], config['data']['type'], config['data']['filename']), 'r') as handle:
            self.data = json.load(handle)['Data']
        
        print('total trianing data:',len(self.data))
        question, passage, answer = [], [], []
        for i in self.data:
            question.append(i['premise'])
            passage.append(i['hypothesis'])
            answer.append(i['label'])
        self.df = pd.DataFrame({
            'question': question,
            'passage': passage,
            'answer': answer
        })
        print('total trianing data:',len(self.df))
        self.df = self.df.iloc[np.random.permutation(len(self.df))].reset_index()
        over_sampler = RandomOverSampler(random_state=42)
        X_res, y_res = over_sampler.fit_resample(self.df, self.df['answer'])
        
        
        question, passage, answer = [], [], []
        for i in range(len(X_res)):
            question.append(str(X_res['question'][i]))
            passage.append(str(X_res['passage'][i]))
            answer.append(bool(X_res['answer'][i]))
        self.df = {
            'question': question,
            'passage': passage,
            'answer': answer
        }
        print('total trianing data:',len(self.df))

        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
                   
        tmp = self.tokenizer(self.df['question'], self.df['passage'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.df_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.df['answer']
            }
        elif self.config['model']=='roberta':
            self.df_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.df['answer']
            }
        else:
            raise
        
    
    
    def __getitem__(self, index):
        
        answer = self.df_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.df_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.df_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.df_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.df_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.df_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return len(self.df['question'])

class cb_dataset_val(Dataset):
    def __init__(self, config=None, train=True, max_rows=None):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        if train:
            self.boolq = load_dataset('super_glue', 'cb')['train'].shuffle(seed=42)
        else:
            self.boolq = load_dataset('super_glue', 'cb')['validation'].shuffle(seed=42)
            
        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
        
        tmp = self.tokenizer(self.boolq['premise'], self.boolq['hypothesis'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.boolq['label']
            }
        elif self.config['model']=='roberta':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.boolq['label']
            }
        else:
            raise

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        answer = self.boolq_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.boolq_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return self.boolq.num_rows

class cb_dataset_train_org(Dataset):
    def __init__(self, config=None, train=True, max_rows=94):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        self.boolq = load_dataset('super_glue', 'cb')['train'].shuffle(seed=42)
            
        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
        
        tmp = self.tokenizer(self.boolq['premise'], self.boolq['hypothesis'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.boolq['label']
            }
        elif self.config['model']=='roberta':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.boolq['answer']
            }
        else:
            raise

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        answer = self.boolq_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.boolq_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        if self.max_rows:
            return self.max_rows
        return self.boolq.num_rows

class copa_dataset_train(Dataset):
    def __init__(self, config=None, train=True, max_rows=None):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        with open(os.path.join(config['datapath'], config['data']['name'], config['data']['type'], config['data']['filename']), 'r') as handle:
            self.data = json.load(handle)
        
        print('total trianing data:',len(self.data))
        target = []
        sentence1 = []
        sentence2 = []
        for t in range(len(self.data['label'])):
            target.append(int(self.data['label'][t].replace('choice1', '0').replace('choice2', '1')))
            qtype = 'because' if self.data['question'][t] == 'cause' else 'so'
            sentence1.append(f"{self.data['premise'][t]} {qtype} {self.data['choice1'][t]}")
            sentence2.append(f"{self.data['premise'][t]} {qtype} {self.data['choice2'][t]}")


        self.df = pd.DataFrame({
            'sentence1': sentence1,
            'sentence2': sentence2,
            'answer': target
        })
        print('total trianing data:',len(self.df))
        self.df = self.df.iloc[np.random.permutation(len(self.df))].reset_index()
        over_sampler = RandomOverSampler(random_state=42)
        X_res, y_res = over_sampler.fit_resample(self.df, self.df['answer'])
        
        
        sentence1, sentence2, answer = [], [], []
        for i in range(len(X_res)):
            sentence1.append(str(X_res['sentence1'][i]))
            sentence2.append(str(X_res['sentence2'][i]))
            answer.append(bool(X_res['answer'][i]))
        self.df = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'answer': answer
        }
        print('total trianing data:',len(self.df))

        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
                   
        tmp = self.tokenizer(self.df['sentence1'], self.df['sentence2'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.df_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.df['answer']
            }
        elif self.config['model']=='roberta':
            self.df_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.df['answer']
            }
        else:
            raise
        
    
    
    def __getitem__(self, index):
        
        answer = self.df_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.df_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.df_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.df_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.df_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.df_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return len(self.df['sentence1'])

class copa_dataset_val(Dataset):
    def __init__(self, config=None, train=True, max_rows=None):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        if train:
            self.boolq = load_dataset('super_glue', 'copa')['train'].shuffle(seed=42)
        else:
            self.boolq = load_dataset('super_glue', 'copa')['validation'].shuffle(seed=42)


        print('total trianing data:',len(self.boolq))
        target = []
        sentence1 = []
        sentence2 = []
        for t in range(len(self.boolq['label'])):
            target.append(int(self.boolq['label'][t]))#.replace('choice1', '0').replace('choice2', '1')))
            qtype = 'because' if self.boolq['question'][t] == 'cause' else 'so'
            sentence1.append(f"{self.boolq['premise'][t]} {qtype} {self.boolq['choice1'][t]}")
            sentence2.append(f"{self.boolq['premise'][t]} {qtype} {self.boolq['choice2'][t]}")


        self.boolq = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'label': target
        }
            
        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
        
        tmp = self.tokenizer(self.boolq['sentence1'], self.boolq['sentence2'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.boolq['label']
            }
        elif self.config['model']=='roberta':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.boolq['label']
            }
        else:
            raise

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        answer = self.boolq_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.boolq_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return len(self.boolq['sentence1'])


class copa_dataset_train_org(Dataset):
    def __init__(self, config=None, train=True, max_rows=400):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        self.boolq = load_dataset('super_glue', 'copa')['train'].shuffle(seed=42)
            
        print('total trianing data:',len(self.boolq))
        target = []
        sentence1 = []
        sentence2 = []
        for t in range(len(self.boolq['label'])):
            target.append(int(self.boolq['label'][t]))#.replace('choice1', '0').replace('choice2', '1')))
            qtype = 'because' if self.boolq['question'][t] == 'cause' else 'so'
            sentence1.append(f"{self.boolq['premise'][t]} {qtype} {self.boolq['choice1'][t]}")
            sentence2.append(f"{self.boolq['premise'][t]} {qtype} {self.boolq['choice2'][t]}")


        self.boolq = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'label': target
        }
            
        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
        
        tmp = self.tokenizer(self.boolq['sentence1'], self.boolq['sentence2'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.boolq['label']
            }
        elif self.config['model']=='roberta':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.boolq['label']
            }
        else:
            raise

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        answer = self.boolq_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.boolq_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        if self.max_rows:
            return self.max_rows
        return self.boolq.num_rows


class multirc_dataset_train(Dataset):
    def __init__(self, config=None, train=True, max_rows=None):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        with open(os.path.join(config['datapath'], config['data']['name'], config['data']['type'], config['data']['filename']), 'r') as handle:
            self.data = json.load(handle)
        
        print('total trianing data:',len(self.data))
        target = []
        sentence1 = []
        sentence2 = []
        for t in range(len(self.data['label'])):
            target.append(int(self.data['label'][t]))
            sentence2.append(f"{self.data['passage'][t]}")
            sentence1.append(f"{self.data['question'][t]} [SEP] {self.data['answer'][t]}")


        self.df = pd.DataFrame({
            'sentence1': sentence1,
            'sentence2': sentence2,
            'answer': target
        })
        print('total trianing data:',len(self.df))
        self.df = self.df.iloc[np.random.permutation(len(self.df))].reset_index()
        over_sampler = RandomOverSampler(random_state=42)
        X_res, y_res = over_sampler.fit_resample(self.df, self.df['answer'])
        
        
        sentence1, sentence2, answer = [], [], []
        for i in range(len(X_res)):
            sentence1.append(str(X_res['sentence1'][i]))
            sentence2.append(str(X_res['sentence2'][i]))
            answer.append(bool(X_res['answer'][i]))
        self.df = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'answer': answer
        }
        print('total trianing data:',len(self.df))

        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
                   
        tmp = self.tokenizer(self.df['sentence1'], self.df['sentence2'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.df_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.df['answer']
            }
        elif self.config['model']=='roberta':
            self.df_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.df['answer']
            }
        else:
            raise
        
    
    
    def __getitem__(self, index):
        
        answer = self.df_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.df_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.df_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.df_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.df_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.df_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return len(self.df['sentence1'])

class multirc_dataset_val(Dataset):
    def __init__(self, config=None, train=True, max_rows=None):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        if train:
            self.boolq = load_dataset('super_glue', 'multirc')['train'].shuffle(seed=42)
        else:
            self.boolq = load_dataset('super_glue', 'multirc')['validation'].shuffle(seed=42)


        print('total trianing data:',len(self.boolq))
        target = []
        sentence1 = []
        sentence2 = []
        for t in range(len(self.boolq['label'])):
            target.append(int(self.boolq['label'][t]))
            sentence2.append(f"{self.boolq['paragraph'][t]}")
            sentence1.append(f"{self.boolq['question'][t]} [SEP] {self.boolq['answer'][t]}")


        self.boolq = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'label': target
        }
            
        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
        
        tmp = self.tokenizer(self.boolq['sentence1'], self.boolq['sentence2'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.boolq['label']
            }
        elif self.config['model']=='roberta':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.boolq['label']
            }
        else:
            raise

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        answer = self.boolq_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.boolq_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return len(self.boolq['sentence1'])

class multirc_dataset_train_org(Dataset):
    def __init__(self, config=None, train=True, max_rows=200):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        self.boolq = load_dataset('super_glue', 'multirc')['train'].shuffle(seed=42)


        print('total trianing data:',len(self.boolq))
        target = []
        sentence1 = []
        sentence2 = []
        for t in range(len(self.boolq['label'])):
            target.append(int(self.boolq['label'][t]))
            sentence2.append(f"{self.boolq['paragraph'][t]}")
            sentence1.append(f"{self.boolq['question'][t]} [SEP] {self.boolq['answer'][t]}")


        self.boolq = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'label': target
        }
            
        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
        
        tmp = self.tokenizer(self.boolq['sentence1'], self.boolq['sentence2'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.boolq['label']
            }
        elif self.config['model']=='roberta':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.boolq['label']
            }
        else:
            raise

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        answer = self.boolq_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.boolq_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        # This was broken
        # return self.max_len
        return self.max_rows

class rte_dataset_train(Dataset):
    def __init__(self, config=None, train=True, max_rows=None):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        with open(os.path.join(config['datapath'], config['data']['name'], config['data']['type'], config['data']['filename']), 'r') as handle:
            self.data = json.load(handle)["train"]
        
        print('total trianing data:',len(self.data))
        answer_map = {
            'entailment': 0,
            'not_entailment': 1
        }

        target = []
        sentence1 = []
        sentence2 = []
        for t in range(len(self.data)):
            target.append(answer_map[self.data[t]['label']])
            sentence2.append(f"hypothesis: {self.data[t]['hypothesis']}")
            sentence1.append(f"premise: {self.data[t]['premise']}")


        self.df = pd.DataFrame({
            'sentence1': sentence1,
            'sentence2': sentence2,
            'answer': target
        })
        print('total trianing data:',len(self.df))
        self.df = self.df.iloc[np.random.permutation(len(self.df))].reset_index()
        over_sampler = RandomOverSampler(random_state=42)
        X_res, y_res = over_sampler.fit_resample(self.df, self.df['answer'])
        
        
        sentence1, sentence2, answer = [], [], []
        for i in range(len(X_res)):
            sentence1.append(str(X_res['sentence1'][i]))
            sentence2.append(str(X_res['sentence2'][i]))
            answer.append(bool(X_res['answer'][i]))
        self.df = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'answer': answer
        }
        print('total trianing data:',len(self.df))

        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
                   
        tmp = self.tokenizer(self.df['sentence1'], self.df['sentence2'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.df_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.df['answer']
            }
        elif self.config['model']=='roberta':
            self.df_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.df['answer']
            }
        else:
            raise
        
    
    
    def __getitem__(self, index):
        
        answer = self.df_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.df_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.df_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.df_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.df_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.df_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return len(self.df['sentence1'])

class rte_dataset_val(Dataset):
    def __init__(self, config=None, train=True, max_rows=None):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        self.boolq = load_dataset('super_glue', 'rte')['validation'].shuffle(seed=42)


        print('total trianing data:',len(self.boolq))
        target = []
        sentence1 = []
        sentence2 = []
        for t in range(len(self.boolq['label'])):
            target.append(int(self.boolq['label'][t]))
            sentence2.append(f"hypothesis: {self.boolq['hypothesis'][t]}")
            sentence1.append(f"premise: {self.boolq['premise'][t]}")


        self.boolq = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'label': target
        }
            
        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
        
        tmp = self.tokenizer(self.boolq['sentence1'], self.boolq['sentence2'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.boolq['label']
            }
        elif self.config['model']=='roberta':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.boolq['label']
            }
        else:
            raise

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        answer = self.boolq_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.boolq_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return len(self.boolq['sentence1'])

class rte_dataset_train_org(Dataset):
    def __init__(self, config=None, train=True, max_rows=2250):
            
        assert config!=None
        
        self.config = config
        self.train = train
        self.max_rows = max_rows
        
        self.boolq = load_dataset('super_glue', 'rte')['train'].shuffle(seed=42)


        print('total trianing data:',len(self.boolq))
        target = []
        sentence1 = []
        sentence2 = []
        for t in range(len(self.boolq['label'])):
            target.append(int(self.boolq['label'][t]))
            sentence2.append(f"hypothesis: {self.boolq['hypothesis'][t]}")
            sentence1.append(f"premise: {self.boolq['premise'][t]}")


        self.boolq = {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'label': target
        }
            
        if config['model']=='bert':
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") 
        elif config['model']=='roberta':
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base") 
        else:
            raise
        
        tmp = self.tokenizer(self.boolq['sentence1'], self.boolq['sentence2'], truncation="only_second", padding=True)
        
        if self.config['model']=='bert':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'token_type_ids': tmp['token_type_ids'],
                'answer': self.boolq['label']
            }
        elif self.config['model']=='roberta':
            self.boolq_enc = {
                'input_ids': tmp['input_ids'],
                'attention_mask': tmp['attention_mask'],
                'answer': self.boolq['label']
            }
        else:
            raise

    def process_data(self, text, max_len):
        text = str(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        padding_length = max_len - len(ids)
        
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
        }
    
    
    def __getitem__(self, index):
        
        answer = self.boolq_enc['answer'][index]
        
        target = torch.from_numpy(np.array([int(answer)])).long()
        
        if self.config['model']=='bert':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
                'types': torch.tensor(self.boolq_enc['token_type_ids'][index], dtype=torch.long),
            }
        elif self.config['model']=='roberta':
            question_tokens = {
                'ids': torch.tensor(self.boolq_enc['input_ids'][index], dtype=torch.long),
                'mask': torch.tensor(self.boolq_enc['attention_mask'][index], dtype=torch.long),
            }
        else:
            raise
        
        return question_tokens, target

    def __len__(self):
        return self.max_rows

class wsc_dataset_base(Dataset):
    def __init__(self, model, split):
        self.model = model

        self.dataset = self.load_dataset(split)
        print(f'Total training data loaded: {len(self.dataset)}')

        self.dataset = self.dataset.map(self.format_dataset)
        print(f'Training data after formatting loaded: {len(self.dataset)}')

        self.tokenizer = tokenizer_dict[self.model]

        tokens = self.tokenizer(
            self.dataset['sentence1'],
            self.dataset['sentence2'],
            truncation="only_second",
            padding=True
        )
        
        self.dataset_enc = {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'answer': self.dataset['label']
        }

        if self.model == 'bert':
            self.dataset_enc['token_type_ids'] = tokens['token_type_ids']

    def load_dataset(self, split: str) -> Dataset:
        return load_dataset('super_glue', 'wsc', split=split).shuffle(seed=42)

    def format_dataset(self, example: Dict[str, str]) -> Dict[str, str]:
        example['sentence1']  = f"{example['text']}"
        example['sentence2']  = f"{example['span1_text']} [SEP] "
        example['sentence2'] += f"{example['span2_text']}"
        return example

    def __getitem__(self, index):
        target = torch.tensor(self.dataset_enc['answer'][index], dtype=torch.long)

        question_tokens = {
            'ids': torch.tensor(self.dataset_enc['input_ids'][index], dtype=torch.long),
            'mask': torch.tensor(self.dataset_enc['attention_mask'][index], dtype=torch.long),
        }
        if self.model == 'bert':
            question_tokens['types'] = torch.tensor(self.dataset_enc['token_type_ids'][index], dtype=torch.long),

        return question_tokens, target

    def __len__(self):
        return len(self.dataset)

class wsc_dataset_train(wsc_dataset_base):
    def __init__(self, config, **kwargs):
        path_to_data = os.path.join(
            config['datapath'],
            config['data']['name'],
            config['data']['type'],
            config['data']['filename']
        )
        super().__init__(model=config['model'], split=path_to_data)

    def load_dataset(self, split: str) -> Dataset:
        return load_dataset("json", data_files=split, field="data", split="train")

    def format_dataset(self, example: Dict[str, str]) -> Dict[str, str]:
        # TODO: Is this how we should be formatting WSC?
        passage = example['sentence1'] + ' ' + example['sentence2']
        example['sentence1']  = f"passage: {passage}"
        example['sentence2']  = f"entity 1: {example['answer1']}"
        example['sentence2'] += f"entity 2: {example['answer2']}"

        label = 1 if example['answer1'] == example['answer2'] else 0
        example['label'] = label

        return example

class wsc_dataset_train_org(wsc_dataset_base):
    def __init__(self, config, **kwargs):
        super().__init__(model=config['model'], split='train')

class wsc_dataset_val(wsc_dataset_base):
    def __init__(self, config, **kwargs):
        super().__init__(model=config['model'], split='validation')

class wic_dataset_base(Dataset):
    def __init__(self, model, split):
        self.model = model

        self.dataset = self.load_dataset(split)
        print(f'Total training data loaded: {len(self.dataset)}')

        self.dataset = self.dataset.map(self.format_dataset)
        print(f'Training data after formatting loaded: {len(self.dataset)}')

        self.tokenizer = tokenizer_dict[self.model]

        tokens = self.tokenizer(
            self.dataset['sentence1'],
            self.dataset['sentence2'],
            truncation="only_second",
            padding=True
        )
        
        self.dataset_enc = {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'answer': self.dataset['label']
        }

        if self.model == 'bert':
            self.dataset_enc['token_type_ids'] = tokens['token_type_ids']

    def load_dataset(self, split: str) -> Dataset:
        return load_dataset('super_glue', 'wic', split=split).shuffle(seed=42)

    def format_dataset(self, example: Dict[str, str]) -> Dict[str, str]:
        example['sentence1'] = f"{example['sentence1']}."
        example['sentence2'] = f"{example['sentence2']}. [SEP] {example['word']}"
        return example

    def __getitem__(self, index):
        target = torch.tensor(self.dataset_enc['answer'][index], dtype=torch.long)

        question_tokens = {
            'ids': torch.tensor(self.dataset_enc['input_ids'][index], dtype=torch.long),
            'mask': torch.tensor(self.dataset_enc['attention_mask'][index], dtype=torch.long),
        }
        if self.model == 'bert':
            question_tokens['types'] = torch.tensor(self.dataset_enc['token_type_ids'][index], dtype=torch.long),

        return question_tokens, target

    def __len__(self):
        return len(self.dataset)

class wic_dataset_train(wic_dataset_base):
    def __init__(self, config, **kwargs):
        path_to_data = os.path.join(
            config['datapath'],
            config['data']['name'],
            config['data']['type'],
            config['data']['filename']
        )
        super().__init__(model=config['model'], split=path_to_data)
    
    def load_dataset(self, split: str) -> Dataset:
        return load_dataset("json", data_files=split, field="data", split="train")

    def format_dataset(self, example: Dict[str, str]) -> Dict[str, str]:
        example = super().format_dataset(example)
        example['label'] = 1 if example['label'] == 'T' else 0
        return example

class wic_dataset_train_org(wic_dataset_base):
    # kwargs added to backwards compatability. the train and max_rows kwargs weren't used
    def __init__(self, config, **kwargs):
        super().__init__(model=config['model'], split='train')

# In the eariler classes we were using the validation split. Why?
# Why not use the test split. Is it because the test set does not have
# the labels in come cases?
class wic_dataset_val(wic_dataset_base):
    # kwargs added to backwards compatability. the train and max_rows kwargs weren't used
    def __init__(self, config, **kwargs):
        super().__init__(model=config['model'], split='validation')

class record_dataset_base(Dataset):
    def __init__(self, model, split):
        self.model = model

        self.dataset = self.load_dataset(split)
        print(f'Total training data loaded: {len(self.dataset)}')

        self.dataset = self.dataset.map(self.format_dataset)
        print(f'Training data after formatting loaded: {len(self.dataset)}')

        self.tokenizer = tokenizer_dict[self.model]

        tokens = self.tokenizer(
            self.dataset['passage'],
            self.dataset['query'],
            truncation="only_second",
            padding=True
        )
        
        self.dataset_enc = {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            # TODO: How do we add the answers it's often the case the answer
            # Is a list of entities. This will currently throw an error.
            'answer': self.dataset['label']
        }

        if self.model == 'bert':
            self.dataset_enc['token_type_ids'] = tokens['token_type_ids']

    def load_dataset(self, split: str) -> Dataset:
        return load_dataset('super_glue', 'record', split=split).shuffle(seed=42)

    def format_dataset(self, example: Dict[str, str]) -> Dict[str, str]:
        # TODO: How do we add the entities in the prompt? A list of entities
        # in the passage is provided for each question.
        example['query'] = f"query: {example['query']}."
        example['passage'] = f"passage: {example['passage']}."
        return example

    def __getitem__(self, index):
        target = torch.tensor(self.dataset_enc['answer'][index], dtype=torch.long)

        question_tokens = {
            'ids': torch.tensor(self.dataset_enc['input_ids'][index], dtype=torch.long),
            'mask': torch.tensor(self.dataset_enc['attention_mask'][index], dtype=torch.long),
        }
        if self.model == 'bert':
            question_tokens['types'] = torch.tensor(self.dataset_enc['token_type_ids'][index], dtype=torch.long),

        return question_tokens, target

    def __len__(self):
        return len(self.dataset)

class record_dataset_train(record_dataset_base):
    def __init__(self, config, **kwargs):
        path_to_data = os.path.join(
            config['datapath'],
            config['data']['name'],
            config['data']['type'],
            config['data']['filename']
        )
        super().__init__(model=config['model'], split=path_to_data)

    def load_dataset(self, split: str) -> Dataset:
        return load_dataset("json", data_files=split, field="data", split="train")

class record_dataset_train_org(record_dataset_base):
    def __init__(self, config, **kwargs):
        super().__init__(model=config['model'], split='train')

class record_dataset_val(record_dataset_base):
    def __init__(self, config, **kwargs):
        super().__init__(model=config['model'], split='validation')

dataloader_map = {
    'boolq': {
        'generated': {
            'train': boolq_dataset_train,
            'val': boolq_dataset_val 
        },
        'original': {
            'train': boolq_dataset_train_org,
            'val': boolq_dataset_val 
        }
    },
    'cb': {
        'generated': {
            'train': cb_dataset_train,
            'val': cb_dataset_val 
        },
        'original': {
            'train': cb_dataset_train_org,
            'val': cb_dataset_val 
        }
    },
    'copa': {
        'generated': {
            'train': copa_dataset_train,
            'val': copa_dataset_val,
            'test': copa_dataset_val 
        },
        'original': {
            'train': copa_dataset_train_org,
            'val': copa_dataset_val,
            'test': copa_dataset_val
        }
    },
    'multirc': {
        'generated': {
            'train': multirc_dataset_train,
            'val': multirc_dataset_val,
            'test': multirc_dataset_val 
        },
        'original': {
            'train': multirc_dataset_train_org,
            'val': multirc_dataset_val,
            'test': multirc_dataset_val
        }
    },
    'rte': {
        'generated': {
            'train': rte_dataset_train,
            'val': rte_dataset_val,
            'test': rte_dataset_val 
        },
        'original': {
            'train': rte_dataset_train_org,
            'val': rte_dataset_val,
            'test': rte_dataset_val
        }
    },
    'wic': {
        'generated': {
            'train': wic_dataset_train,
            'val': wic_dataset_val,
            'test': wic_dataset_val 
        },
        'original': {
            'train': wic_dataset_train_org,
            'val': wic_dataset_val,
            'test': wic_dataset_val
        }
    },
    'wsc': {
        'generated': {
            'train': wsc_dataset_train,
            'val': wsc_dataset_val,
            'test': wsc_dataset_val 
        },
        'original': {
            'train': wsc_dataset_train_org,
            'val': wsc_dataset_val,
            'test': wsc_dataset_val
        }
    },
    'record': {
        'generated': {
            'train': record_dataset_train,
            'val': record_dataset_val,
            'test': record_dataset_val 
        },
        'original': {
            'train': record_dataset_train_org,
            'val': record_dataset_val,
            'test': record_dataset_val
        }
    }
}
