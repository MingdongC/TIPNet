import pickle
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
import networkx as nx
from collections import defaultdict
from pruning_model import PruningModel
from pruning_dataloader import DatasetPruning, DataLoaderPruning

cws = pickle.load(open('webqsp_scores_full_kg.pkl', 'rb'))

f = open('../../data/fbwq_full/train-1111.txt', 'r')
triples = []
for line in f:
    line = line.strip().split('\t')
    triples.append(line)

# 构建KG
G = nx.Graph()
for t in tqdm(triples):
    e1 = t[0]
    e2 = t[2]
    G.add_node(e1)
    G.add_node(e2)
    G.add_edge(e1, e2)

triples_dict = defaultdict(set)     # 三元组 defaultdict(set) 初始化set
for t in tqdm(triples):
    pair = (t[0], t[2])
    triples_dict[pair].add(t[1])

def getRelationsFromKG(head, tail):
    return triples_dict[(head, tail)]

def getRelationsInPath(G, e1, e2):
    path = nx.shortest_path(G, e1, e2)      # 求最短路径
    relations = []
    if len(path) < 2:
        return []
    for i in range(len(path) - 1):
        head = path[i]
        tail = path[i+1]
        rels = list(getRelationsFromKG(head, tail))
        relations.extend(rels)
    return set(relations)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

f = open('../../data/fbwq_full/relations_all.dict', 'r')
rel2idx = {}
idx2rel = {}
for line in f:
    line = line.strip().split('\t')
    id = int(line[1])
    rel = line[0]
    rel2idx[rel] = id
    idx2rel[id] = rel
f.close()

def process_data_file(fname, rel2idx, idx2rel):
    f = open(fname, 'r')
    data = []
    for line in f:
        line = line.strip().split('\t')
        question = line[0].strip()
        #TODO only work for webqsp. to remove entity from metaqa, use something else
        #remove entity from question
        question = question.split('[')[0]
        rel_list = line[1].split('|')
        rel_id_list = []
        for rel in rel_list:
            rel_id_list.append(rel2idx[rel])
        data.append((question, rel_id_list, line[0].strip()))
    return data

model = PruningModel(rel2idx, idx2rel, 0.0)
checkpoint_file = "../../pretrained_models/relation_matching_models/best_mar12_3.pt"
model.load_state_dict(torch.load(checkpoint_file, map_location=lambda storage, loc: storage), strict=False)

data = process_data_file('../../data/fbwq_full/pruning_train.txt', rel2idx, idx2rel)
dataset = DatasetPruning(data=data, rel2idx = rel2idx, idx2rel = idx2rel)
print('Done')


def getHead(q):
    question = q.split('[')
    question_1 = question[0]
    question_2 = question[1].split(']')
    head = question_2[0].strip()
    return head

def get2hop(graph, entity):
    l1 = graph[entity]
    ans = []
    ans += l1
    for item in l1:
        ans += graph[item]
    ans = set(ans)
    if entity in ans:
        ans.remove(entity)
    return ans

def get3hop(graph, entity):
    l1 = graph[entity]
    ans = []
    ans += l1
    for item in l1:
        ans += graph[item]
    ans2 = []
    ans2 += ans
    for item in ans:
        ans2 += graph[item]
    ans2 = set(ans2)
    if entity in ans2:
        ans2.remove(entity)
    return ans2

def get1hop(graph, entity):
    l1 = graph[entity]
    ans = []
    ans += l1
    ans = set(ans)
    if entity in ans:
        ans.remove(entity)
    return ans

def getnhop(graph, entity, hops=1):
    if hops == 1:
        return get1hop(graph, entity)
    elif hops == 2:
        return get2hop(graph, entity)
    else:
        return get3hop(graph, entity)

def getAllRelations(head, tail):
    global G
    global triples_dict
    try:
        shortest_length = nx.shortest_path_length(G, head, tail)
    except:
        shortest_length = 0
    if shortest_length == 0:
        return set()
    if shortest_length == 1:
        return triples_dict[(head, tail)]
    elif shortest_length == 2:
        paths = [nx.shortest_path(G, head, tail)]
        relations = set()
        for p in paths:
            rels1 = triples_dict[(p[0], p[1])]
            rels2 = triples_dict[(p[1], p[2])]
            relations = relations.union(rels1)
            relations = relations.union(rels2)
        return relations
    else:
        return set()

def removeHead(question):
    question = question = question.split('[')[0]
    return question

num_for_testing = 100

# 两跳限制
def limit_2hop(cws):
    num_correct = 0
    for q in tqdm(cws[:num_for_testing]):
        question = q['question']
        question_nohead = question
        answers = q['answers']
        candidates = q['candidates']
        head = q['head']
        question_tokenized, attention_mask = dataset.tokenize_question(question)
        scores = model.get_score_ranked(question_tokenized=question_tokenized, attention_mask=attention_mask)
        pruning_rels_scores, pruning_rels_torch = torch.topk(scores, 5)
        pruning_rels = set()
        pruning_rels_threshold = 0.5
        for s, p in zip(pruning_rels_scores, pruning_rels_torch):
            if s > pruning_rels_threshold:
                pruning_rels.add(idx2rel[p.item()])

        my_answer = ""
        head_nbhood = get2hop(G, head)
        #     max_intersection = 0
        for c in candidates:
            #         candidate_rels = getAllRelations(head, c)
            if c in head_nbhood:
                candidate_rels = getAllRelations(head, c)
                intersection = pruning_rels.intersection(candidate_rels)  # 返回交集
                if len(intersection) > 0:
                    my_answer = c
                    break
        if my_answer == "":
            my_answer = candidates[0]
        if my_answer in answers:
            num_correct += 1
    print('Accuracy of 2hop-limit is:', num_correct / num_for_testing)

# Algorithm mentioned in paper, sec 4.4.1
# slower than the previous one but does not have neighbourhood restriction
def relFiltering(cws):
    num_correct = 0
    for q in tqdm(cws[:num_for_testing]):
        question = q['question']
        question_nohead = question
        answers = q['answers']
        candidates = q['candidates']
        candidates_scores = q['scores']
        head = q['head']
        question_tokenized, attention_mask = dataset.tokenize_question(question)
        scores = model.get_score_ranked(question_tokenized=question_tokenized, attention_mask=attention_mask)
        pruning_rels_scores, pruning_rels_torch = torch.topk(scores, 2)
        pruning_rels = set()
        pruning_rels_threshold = 0.5 # threshold to consider as written in sec 4.4.1
        for s, p in zip(pruning_rels_scores, pruning_rels_torch):
            if s > pruning_rels_threshold:
                pruning_rels.add(idx2rel[p.item()])
        gamma = 1.0
        max_score = 0.0
        max_score_relscore = 0.0
        max_score_answer = ""
        for score, c in zip(candidates_scores, candidates):
            actual_rels = getAllRelations(head, c)
            relscore = len(actual_rels.intersection(pruning_rels))
            totalscore = score + gamma*relscore
            if totalscore > max_score:
                max_score = totalscore
                max_score_relscore = relscore
                max_score_answer = c
        is_correct = False
        if max_score_answer in answers:
            num_correct += 1
            is_correct = True

    print('Accuracy of relation filtering is:', num_correct/num_for_testing)

limit_2hop(cws)
relFiltering(cws)

'''
# following code is just for investigating
# i haven't removed it since it might be useful if
# someone wants to explore
qid=2
model.eval()
qe = cws[qid]['qe']
p1, p2 = torch.topk(model.get_score_ranked(qe), 5)
question = cws[qid]['question']
print(question)
# print(idx2rel[pred])
for p in p2:
    print(idx2rel[p.item()])
candidates = cws[qid]['candidates']
print(candidates)
# print(cws[qid]['scores'])
head = getHead(question)
tail = candidates[1]
getAllRelations(head, tail)
'''