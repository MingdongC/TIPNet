import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformers import RobertaModel,RobertaTokenizer
import random
from transformer_base import Transformer

class RelationExtractor(nn.Module):

    def __init__(self, embedding_dim, relation_dim, num_entities, pretrained_embeddings, rel_embedding, device, entdrop, reldrop,
                 scoredrop, l3_reg, model, ls, do_batch_norm, freeze=True):
        super(RelationExtractor, self).__init__()
        self.device = device
        self.model = model
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        self.do_batch_norm = do_batch_norm
        if not self.do_batch_norm:
            print('Not doing batch norm')
        self.roberta_pretrained_weights = './hfl/model/roberta/roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)
        self.tokenizer_class = RobertaTokenizer
        self.tokenizer = self.tokenizer_class.from_pretrained(self.roberta_pretrained_weights)
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        if self.model == 'DistMult':
            multiplier = 1
            self.getScores = self.DistMult
        elif self.model == 'SimplE':
            multiplier = 2
            self.getScores = self.SimplE
        elif self.model == 'ComplEx':
            multiplier = 2
            self.getScores = self.ComplEx
        elif self.model == 'TuckER':
            # W_torch = torch.from_numpy(np.load(w_matrix))
            # self.W = nn.Parameter(
            #     torch.Tensor(W_torch), 
            #     requires_grad = not self.freeze
            # )
            self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (relation_dim, relation_dim, relation_dim)),
                                               dtype=torch.float, device="cuda", requires_grad=True))
            multiplier = 1
            self.getScores = self.TuckER
        elif self.model == 'RESCAL':
            self.getScores = self.RESCAL
            multiplier = 1
        else:
            print('Incorrect model specified:', self.model)
            exit(0)
        print('Model is', self.model)
        self.hidden_dim = 768
        self.relation_dim = relation_dim * multiplier
        if self.model == 'RESCAL':
            self.relation_dim = relation_dim * relation_dim

        self.num_entities = num_entities
        # self.loss = torch.nn.BCELoss(reduction='sum')
        self.loss = self.kge_loss

        # best: all dropout 0
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)
        self.fcnn_dropout = torch.nn.Dropout(0.1)

        # self.pretrained_embeddings = pretrained_embeddings
        # random.shuffle(pretrained_embeddings)
        # print(pretrained_embeddings[0])
        print('Frozen:', self.freeze)
        self.embedding = nn.Embedding.from_pretrained(torch.stack(pretrained_embeddings, dim=0), freeze=self.freeze)
        self.embedding_rel = nn.Embedding.from_pretrained(torch.stack(rel_embedding, dim=0), freeze=self.freeze)
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=self.freeze)
        print(self.embedding.weight.shape)
        # self.embedding = nn.Embedding(self.num_entities, self.relation_dim)
        # self.embedding.weight.requires_grad = False
        # xavier_normal_(self.embedding.weight.data)

        self.mid1 = 512
        self.mid2 = 512
        self.mid3 = 512
        self.mid4 = 512

        # self.lin1 = nn.Linear(self.hidden_dim, self.mid1)
        # self.lin2 = nn.Linear(self.mid1, self.mid2)
        # self.lin3 = nn.Linear(self.mid2, self.mid3)
        # self.lin4 = nn.Linear(self.mid3, self.mid4)
        # self.hidden2rel = nn.Linear(self.mid4, self.relation_dim)
        self.hidden2rel = nn.Linear(self.hidden_dim, self.relation_dim)
        self.hidden2rel_base = nn.Linear(self.mid2, self.relation_dim)
        self.halfRelDim2Tranf = nn.Linear(self.relation_dim, 256)

        if self.model in ['DistMult', 'TuckER', 'RESCAL', 'SimplE']:
            self.bn0 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
            self.bn2 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
        else:
            self.bn0 = torch.nn.BatchNorm1d(multiplier)
            self.bn2 = torch.nn.BatchNorm1d(multiplier)

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self._klloss = torch.nn.KLDivLoss(reduction='sum')
        self.Transformer = Transformer(src_pad_idx=0, trg_pad_idx=0, d_word_vec=512, d_inner=2048, n_layers=6, n_head=8, d_k=64,
                                       d_v=64, dropout=0.1, n_position=200, Ti_value=True)
        self.bert2tranf = nn.Linear(self.hidden_dim, 512)

    def set_bn_eval(self):
        self.bn0.eval()
        self.bn2.eval()

    def kge_loss(self, scores, targets):
        # loss = torch.mean(scores*targets)
        return self._klloss(
            F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1)
        )

    def applyNonLinear(self, outputs):
        # outputs = self.fcnn_dropout(self.lin1(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin2(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.lin3(outputs)
        # outputs = F.relu(outputs)
        # outputs = self.lin4(outputs)
        # outputs = F.relu(outputs)
        outputs = self.hidden2rel_base(outputs)
        # outputs = self.hidden2rel_base(outputs)
        return outputs

    def TuckER(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        x = head.view(-1, 1, head.size(1))

        W_mat = torch.mm(relation, self.W.view(relation.size(1), -1))
        W_mat = W_mat.view(-1, head.size(1), head.size(1))
        W_mat = self.rel_dropout(W_mat)
        x = torch.bmm(x, W_mat)
        x = x.view(-1, head.size(1))
        x = self.bn2(x)
        x = self.score_dropout(x)

        x = torch.mm(x, self.embedding.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def RESCAL(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        ent_dim = head.size(1)
        head = head.view(-1, 1, ent_dim)
        relation = relation.view(-1, ent_dim, ent_dim)
        relation = self.rel_dropout(relation)
        x = torch.bmm(head, relation)
        x = x.view(-1, ent_dim)
        x = self.bn2(x)
        x = self.score_dropout(x)
        x = torch.mm(x, self.embedding.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def DistMult(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s = self.bn2(s)
        s = self.score_dropout(s)
        ans = torch.mm(s, self.embedding.weight.transpose(1, 0))
        pred = torch.sigmoid(ans)
        return pred

    def SimplE(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s_head, s_tail = torch.chunk(s, 2, dim=1)
        s = torch.cat([s_tail, s_head], dim=1)
        s = self.bn2(s)
        s = self.score_dropout(s)
        s = torch.mm(s, self.embedding.weight.transpose(1, 0))
        s = 0.5 * s
        pred = torch.sigmoid(s)
        return pred

    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            head = self.bn0(head)

        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim=1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.bn2(score)

        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))
        # pred = torch.sigmoid(score)
        pred = score
        return pred

    def del_tensor_ele(self, arr, index, dim):
        arr1 = arr[1:index + 1]
        arr2 = arr[index + 3:]
        arr = torch.cat((arr1, arr2), dim=dim)
        return arr

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        questions_lenth = []
        for bat in range(question_tokenized.size(0)):
            idx_word = self.tokenizer.convert_ids_to_tokens(question_tokenized[bat])
            for id in range(len(idx_word)):
                if idx_word[id] == '</s>':
                    lenth_que = id - 2
                    questions_lenth.append(lenth_que)
                    break
        a = self.roberta_model(question_tokenized, attention_mask=attention_mask)
        question_embedding = a[1]
        words_embedding = a[0]
        for bat in range(words_embedding.size(0)):
            words_embedding1 = self.del_tensor_ele(words_embedding[bat], questions_lenth[bat], 0).unsqueeze(0)
            words_mask1 = self.del_tensor_ele(attention_mask[bat], questions_lenth[bat], 0).unsqueeze(0)
            if bat == 0:
                words_embedding_all = words_embedding1
                words_mask_all = words_mask1
            else:
                words_embedding_all = torch.cat((words_embedding_all, words_embedding1), dim=0)
                words_mask_all = torch.cat((words_mask_all, words_mask1), dim=0)
        words_mask = words_mask_all
        words_embedding = words_embedding_all
        return question_embedding, words_embedding, words_mask

    def getWordsEmbedding(self, words_tokenized, attention_masks):
        src_mask = words_tokenized[..., 1]
        words_tokenized = words_tokenized.transpose(1, 0)
        attention_masks = attention_masks.transpose(1, 0)
        for idx in range(words_tokenized.size()[0]):
            word_tokenized = words_tokenized[idx]
            attention_mask = attention_masks[idx]
            roberta_last_hidden_states = self.roberta_model(word_tokenized, attention_mask=attention_mask)[0]
            states = roberta_last_hidden_states.transpose(1, 0)
            cls_embedding = states[0]
            word_embedding = cls_embedding.unsqueeze(0)
            if idx == 0:
                words_embedding = word_embedding
            else:
                words_embedding = torch.cat((words_embedding, word_embedding), 0)
        words_embedding = words_embedding.transpose(1, 0)
        return words_embedding, src_mask

    def words_ti_value_haddle(self, words_ti_value, question_tokenized):
        word_mark = "<s> </s>"
        word_mark = self.tokenizer.tokenize(word_mark)[1]
        for bat in range(question_tokenized.size(0)):
            tokens = question_tokenized[bat]
            tokens = tokens[1:]
            tokens = self.tokenizer.convert_ids_to_tokens(tokens)
            ti_value = words_ti_value[bat]
            for a in range(len(ti_value)):
                temp = torch.full((1, 512), ti_value[a], dtype=torch.float32, device=self.device)
                if a == 0:
                    temp11 = temp
                else:
                    temp11 = torch.cat((temp11, temp), dim=0)
            ti_value = temp11
            result = pre_ti_value = ti_value[0].unsqueeze(0)
            cur_ti_value = ti_value[1].unsqueeze(0)
            idxForTi = 2
            ti_one = torch.full((1, 512), 0, dtype=torch.float32, device=self.device)
            for x in range(len(tokens) - 1):
                a = tokens[x + 1]
                if tokens[x + 1] != word_mark and tokens[x + 1] != '</s>' and tokens[x + 1] != '<pad>':
                    if word_mark in tokens[x + 1]:
                        result = torch.cat((result, cur_ti_value), dim=0)
                        pre_ti_value = cur_ti_value
                        cur_ti_value = ti_value[idxForTi].unsqueeze(0)
                        idxForTi += 1
                    else:
                        result = torch.cat((result, pre_ti_value), dim=0)
                elif tokens[x] == '</s>' or tokens[x] == word_mark:
                    pass
                else:
                    result = torch.cat((result, ti_one), dim=0)
            result = result.unsqueeze(0)
            if bat == 0:
                results = result
            else:
                results = torch.cat((results, result), dim=0)

        return results

    def forward(self, question_tokenized, attention_mask, p_head, p_tail, words_ti_value, relation):
        question_embedding, words_embedding, src_mask = self.getQuestionEmbedding(question_tokenized, attention_mask)
        rel = relation.transpose(0,1)
        rel1 = self.halfRelDim2Tranf(self.embedding_rel(rel[0]))
        rel2 = self.halfRelDim2Tranf(self.embedding_rel(rel[1]))
        trg_rel_emb = torch.cat((rel1,rel2), dim=1).unsqueeze(1)

        temp1 = torch.sum(words_embedding, dim=1)

        words_embedding = self.bert2tranf(words_embedding)
        words_ti_value = self.words_ti_value_haddle(words_ti_value, question_tokenized)
        words_TiBased_embedding = self.Transformer(words_embedding, src_mask, words_ti_value, trg_rel_emb, trg_mask=None)
        # words_TiBased_embedding *= words_ti_value
        # temp = torch.sum(words_TiBased_embedding, dim=1)
        temp = words_TiBased_embedding.transpose(0,1)[0]

        rel_embedding = self.applyNonLinear(temp)
        p_head = self.embedding(p_head)
        pred = self.getScores(p_head, rel_embedding)
        actual = p_tail
        if self.label_smoothing:
            actual = ((1.0 - self.label_smoothing) * actual) + (1.0 / actual.size(1))
        loss = self.loss(pred, actual)
        if not self.freeze:
            if self.l3_reg:
                norm = torch.norm(self.embedding.weight, p=3, dim=-1)
                loss = loss + self.l3_reg * torch.sum(norm)
        return loss

    def get_score_ranked(self, head, question_tokenized, attention_mask, words_ti_value, relation):
        question_embedding, words_embedding, src_mask = self.getQuestionEmbedding(question_tokenized.unsqueeze(0),
                                                                                  attention_mask.unsqueeze(0))
        rel = relation
        rel1 =self.halfRelDim2Tranf(self.embedding_rel(rel[0]))
        rel2 = self.halfRelDim2Tranf(self.embedding_rel(rel[1]))
        trg_rel_emb = torch.cat((rel1,rel2), dim=0).unsqueeze(0).unsqueeze(1)
        words_embedding = self.bert2tranf(words_embedding)
        words_ti_value = self.words_ti_value_haddle(words_ti_value.unsqueeze(0), question_tokenized.unsqueeze(0))
        words_TiBased_embedding = self.Transformer(words_embedding, src_mask, words_ti_value, trg_rel_emb, trg_mask=None)

        # words_TiBased_embedding *= words_ti_value
        # temp = torch.sum(words_TiBased_embedding, dim=1)
        temp = words_TiBased_embedding.transpose(0,1)[0]

        rel_embedding = self.applyNonLinear(temp)
        # words_TiBased_embedding = words_TiBased_embedding.transpose(1,0)[1]
        # rel_embedding = self.applyNonLinear(words_TiBased_embedding)
        head = self.embedding(head).unsqueeze(0)
        scores = self.getScores(head, rel_embedding)
        # top2 = torch.topk(scores, k=2, largest=True, sorted=True)
        # return top2
        return scores

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


