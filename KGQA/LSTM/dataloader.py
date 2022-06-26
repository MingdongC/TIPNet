import torch
import random
from torch.utils.data import Dataset, DataLoader
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np


class DatasetMetaQA(Dataset):
    '''
    :keyword
        data = {头实体，问题，答案}
        word2ix = {问题中不重复的词的编码}
        relations = {预训练KG中关系对应的向量表示}
        entities = {预训练KG中实体对应的向量表示}
        entity2idx = {实体对应的编码}
    '''
    def __init__(self, data, word2ix, relations, entities, entity2idx): #data = {头实体，问题，答案}
        self.data = data
        self.relations = relations
        self.entities = entities
        self.word_to_ix = {}
        self.entity2idx = entity2idx
        self.word_to_ix = word2ix
        self.pos_dict = defaultdict(list)   #defaultdict函数为字典中没出现的key设置默认值以免出现keyerror
        self.neg_dict = defaultdict(list)   #这个factory_function可以是list、set、str等等，作用是当key不存在时，返回的是工厂函数的默认值，比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应0
        self.index_array = list(self.entities.keys())


    def __len__(self):
        return len(self.data)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)     #编码onehot scatter {dim,indices,value填充值}
        return one_hot

    #对data{头实体，问题，答案}进行处理,返回{问题的index list，主题词的index，答案的onehot}
    def __getitem__(self, index):
        data_point = self.data[index]       #获取每一个data三元组{主题词，问题，答案}
        question_text = data_point[1]       #"1" = 问题
        question_ids = [self.word_to_ix[word] for word in question_text.split()]        #获得问题中的词的index
        head_id = self.entity2idx[data_point[0].strip()]        #获得主题词的index
        q_value_tfidf = data_point[3]
        tail_ids = []
        for tail_name in data_point[2]:     #获得答案的索引，并加入到tail_ids中 ids = {}
            tail_name = tail_name.strip()
            tail_ids.append(self.entity2idx[tail_name])
        tail_onehot = self.toOneHot(tail_ids)       #答案进行onehot编码，维度为KG中所有实体的数量
        return question_ids, head_id, tail_onehot, q_value_tfidf


#定义如何取样本的函数 batch = {问题index，主题词index，答案onehot}
def _collate_fn(batch):
    sorted_seq = sorted(batch, key=lambda sample: len(sample[0]), reverse=True)     #将一个batch按照问题的长度降序
    sorted_seq_lengths = [len(i[0]) for i in sorted_seq]        #batch长度排序
    longest_sample = sorted_seq_lengths[0]
    minibatch_size = len(batch)
    # print(minibatch_size)
    # aditay
    input_lengths = []
    p_head = []
    p_tail = []
    inputs = torch.zeros(minibatch_size, longest_sample, dtype=torch.long)      #保证所有问题的长度都一样，不足的用0填充
    for x in range(minibatch_size):
        # data_a = x[0]
        sample = sorted_seq[x][0]           #sample = {问题index}
        p_head.append(sorted_seq[x][1])     #主题词index
        tail_onehot = sorted_seq[x][2]
        value_tfidf = sorted_seq[x][3]
        p_tail.append(tail_onehot)
        seq_len = len(sample)
        input_lengths.append(seq_len)
        sample = torch.tensor(sample, dtype=torch.long)
        sample = sample.view(sample.shape[0])
        ti_more = torch.full((1,256), value_tfidf[0],dtype=torch.float32)
        for i in range(len(value_tfidf)-1):
            ti_one = torch.full((1,256),value_tfidf[i+1],dtype=torch.float32)
            ti_more = torch.cat((ti_more,ti_one),dim=0)
        if len(value_tfidf) != longest_sample:
            for j in range(longest_sample-len(value_tfidf)):
                ti_one = torch.full((1,256),0,dtype=torch.float32)
                ti_more = torch.cat((ti_more,ti_one),dim=0)
        ti_more = ti_more.unsqueeze(0)
        if x == 0:
            q_value_tfidf = ti_more
        else:
            q_value_tfidf = torch.cat((q_value_tfidf,ti_more),dim=0)
        inputs[x].narrow(0,0,seq_len).copy_(sample)     #narrow(dim,start,length) inputs = {问题index}

        #{问题index, 问题长度, 主题词index,答案onehot }
    return inputs, torch.tensor(input_lengths, dtype=torch.long), torch.tensor(p_head, dtype=torch.long), torch.stack(p_tail), q_value_tfidf


class DataLoaderMetaQA(DataLoader):
    '''
        批训练，把数据变成一小批一小批数据进行训练。
        DataLoader就是用来包装所使用的数据，每次抛出一批数据
        batch_size=1024, data=208970, sample=data/batch_size
    '''
    def __init__(self, *args, **kwargs):
        super(DataLoaderMetaQA, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

    

