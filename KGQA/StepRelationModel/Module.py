import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformers import RobertaModel
import random
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000

class StepRelation(nn.Module):
    def __init__(self, device, relation_dim, linear_drop, max_hop_num, ques_dim):
        super(StepRelation, self).__init__()
        self.max_hop_num = max_hop_num
        self.device = device
        self.relation_dim = relation_dim
        self.linear_drop = nn.Dropout(p=linear_drop)
        self.linear_num1 = nn.Linear(in_features=2*self.relation_dim, out_features=relation_dim)
        self.linear_num2 = nn.Linear(in_features=relation_dim, out_features=1)
        self.linear_pred_hop = nn.Linear(in_features=ques_dim, out_features=self.max_hop_num)
        for i in range(self.max_hop_num):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=relation_dim, out_features=relation_dim))


    def init_step0_relation(self, batch_size_cur):
        self.pre_hop_relation = torch.zeros(batch_size_cur, self.relation_dim).to(self.device)
        self.relations_per_hop = []
        self.attentions_list = []


    def get_next_hop_relation(self, question_embedding, words_embedding, words_mask, hop_cur):
        relaiton_pre = self.pre_hop_relation.unsqueeze(1)
        question_embedding = question_embedding.unsqueeze(1)
        question_linear = getattr(self, 'question_linear' + str(hop_cur))
        question_relation_linear = question_linear(self.linear_drop(question_embedding))

        words_relation_cat = self.linear_num1(self.linear_drop(torch.cat((relaiton_pre, question_relation_linear), dim=-1)))
        words_question = self.linear_num2(self.linear_drop(words_relation_cat * words_embedding))

        atten_relation = F.softmax(words_question + (1 - words_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)
        relation_cur = torch.sum(atten_relation * words_embedding, dim=1)

        return relation_cur, atten_relation

    def predict_question_hops(self, question_embedding):
        #(batch_size, question_embedding)
        hop_pred_linear = F.softmax(self.linear_pred_hop(self.linear_drop(question_embedding)), dim=-1)

        return hop_pred_linear

    def forward(self, question_embedding, words_embedding, words_mask, batch_size_cur):
        self.init_step0_relation(batch_size_cur)
        for i in range(self.max_hop_num):
            relation_cur, attention_w = self.get_next_hop_relation(question_embedding, words_embedding, words_mask, i)
            self.relations_per_hop.append(relation_cur)
            self.attentions_list.append(attention_w)
            self.pre_hop_relation = relation_cur

        return self.relations_per_hop, self.attentions_list


class TransformerBasedFiltering(nn.Module):

    def __init__(self):
        super(TransformerBasedFiltering, self).__init__()
        pass

    def forward(self):
        pass