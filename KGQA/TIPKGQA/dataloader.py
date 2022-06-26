import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from transformers import RobertaModel,RobertaTokenizer


class DatasetMetaQA(Dataset):
    def __init__(self, data, entities, entity2idx, que_rel, relations, rel2idx):
        self.data = data
        self.entities = entities
        self.entity2idx = entity2idx
        self.que_rel = que_rel
        self.relations = relations
        self.rel2idx = rel2idx
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.index_array = list(self.entities.keys())
        self.tokenizer_class = RobertaTokenizer
        self.pretrained_weights = './hfl/model/roberta/roberta-base'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)

    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # one_hot = -torch.ones(vec_len, dtype=torch.float32)
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        que = question_text.replace('NEF', '')
        relation = self.que_rel[que]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        #words_tokenized, attention_masks = self.tokenize_words(question_text)
        head_id = self.entity2idx[data_point[0].strip()]
        rel_id = []
        len_relation = len(relation)
        relation_mask = []
        for r in relation:
            rel = self.rel2idx[r]
            rel_id.append(rel)
            relation_mask.append(1)
        if len_relation < 5:
            for i in range(5-len_relation):
                rel_id.append(self.rel2idx['pad'])
                relation_mask.append(0)
        tail_ids = []
        words_ti_values = data_point[3]
        for a in range(64-len(words_ti_values)):
            words_ti_values.append(0)

        for tail_name in data_point[2]:
            tail_name = tail_name.strip()
            #TODO: dunno if this is right way of doing things
            if tail_name in self.entity2idx:
                tail_ids.append(self.entity2idx[tail_name])
        tail_onehot = self.toOneHot(tail_ids)
        return question_tokenized, attention_mask, head_id, tail_onehot, torch.tensor(words_ti_values, dtype=torch.float32), torch.tensor(rel_id, dtype=torch.long), torch.tensor(relation_mask, dtype=torch.long)

    def tokenize_question(self, question):
        word_mark = "<s> </s>"
        word_mark = self.tokenizer.tokenize(word_mark)[1]
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 64)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        a = self.tokenizer.convert_ids_to_tokens(question_tokenized)

        attention_mask = []
        for q in question_tokenized:
            # 1 means padding token
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

    def tokenize_words(self,question):
        words_list = []
        attention_masks = []
        for word in question.split(' '):
            word = "<s> " + word + " </s>"
            word_tokenized = self.tokenizer.tokenize(word)
            word_tokenized = self.pad_sequence(word_tokenized,8)
            word_tokenized = self.tokenizer.encode(word_tokenized,add_special_tokens=False)
            a = self.tokenizer.convert_ids_to_tokens(word_tokenized)
            attention_mask = []
            words_list.append(word_tokenized)
            for w in word_tokenized:
                if w == 1:
                    attention_mask.append(0)
                else:
                    attention_mask.append(1)

            attention_masks.append(attention_mask)
        words_tokenized = torch.tensor(words_list)

        return  words_tokenized, torch.tensor(attention_masks,dtype=torch.long)
'''
def _collate_fn(batch):
    #print(len(batch))
    #exit(0)
    sorted_seq = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)  # 将一个batch按照问题的长度降序
    sorted_seq_lengths = [len(i[0]) for i in sorted_seq]  # batch长度排序
    longest_sample = sorted_seq_lengths[0]
    minibatch_size = len(batch)
    words = torch.zeros((minibatch_size, longest_sample), dtype=torch.long, device=)
    attentions = torch.zeros((minibatch_size,longest_sample), dtype=torch.long)
    heads = []
    for x in range(minibatch_size):
        words_sorce = sorted_seq[x][0]
        words_len = len(words_sorce)
        attentions_sorce = sorted_seq[x][1]
        words[x].narrow(0,0,words_len).copy_(words_sorce)
        attentions[x].narrow(0,0,words_len).copy_(attentions_sorce)
        heads.append(sorted_seq[x][2])
        tail = sorted_seq[x][3].unsqueeze(0)

        #============================tf-idf处理============================
        value_tfidf = sorted_seq[x][4]
        ti_more = torch.full((1, 512), value_tfidf[0], dtype=torch.float32)
        for i in range(len(value_tfidf) - 1):
            ti_one = torch.full((1, 512), value_tfidf[i + 1], dtype=torch.float32)
            ti_more = torch.cat((ti_more, ti_one), dim=0)
        if len(value_tfidf) != longest_sample:
            for j in range(longest_sample - len(value_tfidf)):
                ti_one = torch.full((1, 512), 0, dtype=torch.float32)
                ti_more = torch.cat((ti_more, ti_one), dim=0)
        ti_more = ti_more.unsqueeze(0)
        if x == 0:
            q_value_tfidf = ti_more
        else:
            q_value_tfidf = torch.cat((q_value_tfidf, ti_more), dim=0)

        if x == 0:
            tails = tail
        else:
            tails = torch.cat((tails,tail),dim=0)

    return words, attentions, torch.tensor(heads, dtype=torch.long), tails, q_value_tfidf        #return {words_tokenized, attention_masks, head_id, tail_onehot}
'''

class DataLoaderMetaQA(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderMetaQA, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

