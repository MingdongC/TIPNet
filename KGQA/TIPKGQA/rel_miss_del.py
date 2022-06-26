import numpy as np
import torch

def getRelationEmbeddings():
    r = {}
    rel_names = []
    relation_dict = '../../pretrained_models/embeddings/ComplEx_fbwq_half/relation_ids.del'
    # embedder = kge_model._relation_embedder
    f = open(relation_dict, 'r', encoding='UTF-8')
    for line in f:
        line = line[:-1].split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        rel_names.append(rel_name)
        # r[rel_name] = embedder._embeddings(torch.LongTensor([rel_id]))[0]       #初始化embedding
    f.close()
    return rel_names

def process_relQue_file(rQ_file):
    que_rel = {}
    rel_file = open(rQ_file, 'r')
    for line in rel_file.readlines():
        temp = line.strip().split('\t')
        question = temp[0]
        relations = temp[1].split('|')
        que_rel[question] = relations
    rel_file.close()
    return que_rel

def process_text_file(text_file):
    data_file = open(text_file, 'r')
    que = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        # if no answer
        if len(data_line) != 2:
            continue
        question = data_line[0].split('[')
        question = question[0]
        que.append(question)
    return que

arr = '../../data/QA_data/WebQuestionsSP/qa_train_webqsp.txt'
que_rel_path = '../../data/QA_data/WebQuestionsSP/que_rel_all.txt'
qa_train = process_text_file(arr)
qa_rel = process_relQue_file(que_rel_path)
rel_names = getRelationEmbeddings()

count = 0
rel_miss = []
qa_miss = []
for que in qa_train:
    rels = qa_rel[que]
    find_q = False
    for r in rels:
        find = False
        for i in range(len(rel_names)):
            if r == rel_names[i]:
                find = True
                break
        if not find:
            find_q = True
            rel_miss.append(r)
            count += 1
    if find_q:
        qa_miss.append(que)



print()
