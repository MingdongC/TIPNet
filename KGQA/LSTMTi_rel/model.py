import torch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_normal_
from transformer_base import Transformer


class RelationExtractor(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, relation_dim, num_entities, pretrained_embeddings,
                 rel_embedding, device, entdrop, reldrop, scoredrop, l3_reg, model, ls, w_matrix, bn_list, freeze=True):
        super(RelationExtractor, self).__init__()       #调用nn.Module__init__方法
        self.device = device
        self.bn_list = bn_list      #预训练模型的文件
        self.model = model
        self.freeze = freeze
        self.label_smoothing = ls
        self.l3_reg = l3_reg
        if self.model == 'DistMult':            #选择嵌入模型的评分函数
            multiplier = 1
            self.getScores = self.DistMult
        elif self.model == 'SimplE':
            multiplier = 2
            self.getScores = self.SimplE
        elif self.model == 'ComplEx':
            multiplier = 2
            self.getScores = self.ComplEx
        elif self.model == 'Rotat3':
            multiplier = 3
            self.getScores = self.Rotat3
        elif self.model == 'TuckER':
            W_torch = torch.from_numpy(np.load(w_matrix))
            self.W = nn.Parameter(
                torch.Tensor(W_torch), 
                requires_grad = True
            )
            # self.W = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (relation_dim, relation_dim, relation_dim)), 
            #                         dtype=torch.float, device="cuda", requires_grad=True))
            multiplier = 1
            self.getScores = self.TuckER
        elif self.model == 'RESCAL':
            self.getScores = self.RESCAL
            multiplier = 1
        else:
            print('Incorrect model specified:', self.model)
            exit(0)
        print('Model is', self.model)
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim * multiplier
        if self.model == 'RESCAL':
            self.relation_dim = relation_dim * relation_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)      #将问题中的q的word  固定embedding维度
        self.n_layers = 1
        self.bidirectional = True
        
        self.num_entities = num_entities
        self.loss = torch.nn.BCELoss(reduction='sum')       # Loss = -w * [p * log(q) + (1-p) * log(1-q)]  p为预测值，q为目标值 torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction=‘mean’)

        # best: all dropout 0
        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.pretrained_embeddings = pretrained_embeddings
        print('Frozen:', self.freeze)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=self.freeze)     #转成Embedding结构
        self.embedding_rel = nn.Embedding.from_pretrained(torch.FloatTensor(rel_embedding), freeze=self.freeze)     #转成Embedding结构
        # self.embedding = nn.Embedding(self.num_entities, self.relation_dim)
        # xavier_normal_(self.embedding.weight.data)

        self.mid1 = 256         #中间层的维度
        self.mid2 = 256

        # 设置全连接层 2
        self.lin1 = nn.Linear(hidden_dim * 2, self.mid1, bias=False)        #in_features=400, out_features=256, bias=false
        self.lin2 = nn.Linear(self.mid1, self.mid2, bias=False)             #in_features=256, out_features=256, bias=false
        xavier_normal_(self.lin1.weight.data)   #线性层初始化
        xavier_normal_(self.lin2.weight.data)
        self.hidden2rel = nn.Linear(self.mid2, self.relation_dim)           #关系隐藏层  2
        self.hidden2rel_base = nn.Linear(hidden_dim * 2, self.relation_dim)
        self.hop3rel2Transf = nn.Linear(self.relation_dim*3, 512)
        self.hop12rel2Transf = nn.Linear(self.relation_dim, 256)
        self.transf2rel = nn.Linear(512, self.relation_dim)

        if self.model in ['DistMult', 'TuckER', 'RESCAL', 'SimplE']:
            self.bn0 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
            self.bn2 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
        else:
            self.bn0 = torch.nn.BatchNorm1d(multiplier)
            self.bn2 = torch.nn.BatchNorm1d(multiplier)

        for i in range(3):
            for key, value in self.bn_list[i].items():
                self.bn_list[i][key] = torch.Tensor(value).to(device)

        
        self.bn0.weight.data = self.bn_list[0]['weight']
        self.bn0.bias.data = self.bn_list[0]['bias']
        self.bn0.running_mean.data = self.bn_list[0]['running_mean']
        self.bn0.running_var.data = self.bn_list[0]['running_var']

        self.bn2.weight.data = self.bn_list[2]['weight']
        self.bn2.bias.data = self.bn_list[2]['bias']
        self.bn2.running_mean.data = self.bn_list[2]['running_mean']
        self.bn2.running_var.data = self.bn_list[2]['running_var']

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.GRU = nn.LSTM(embedding_dim, self.hidden_dim, self.n_layers, bidirectional=self.bidirectional, batch_first=True)
        self.Transformer = Transformer(src_pad_idx=0, trg_pad_idx=0, d_word_vec=512, d_inner=2048, n_layers=6, n_head=8, d_k=64,
                                       d_v=64, dropout=0.1, n_position=200, Ti_value=True)


    def get_src_mask(self, words_embedding, question_length):

        batch_size = words_embedding.size(0)
        token_length = words_embedding.size(1)
        src_mask = torch.zeros(batch_size, token_length, dtype=torch.long)
        for batch in range(batch_size):
            que_lenth = question_length[batch]
            a = src_mask[batch]
            actual_mask = torch.full((1, que_lenth.item()), 1, dtype=torch.long).squeeze(0)
            src_mask[batch].narrow(0, 0, que_lenth.item()).copy_(actual_mask)

        return src_mask

    def applyNonLinear(self, outputs):
        outputs = self.lin1(outputs)
        outputs = F.relu(outputs)
        outputs = self.lin2(outputs)
        outputs = F.relu(outputs)
        outputs = self.hidden2rel(outputs)
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

        x = torch.mm(x, self.embedding.weight.transpose(1,0))
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
        x = torch.mm(x, self.embedding.weight.transpose(1,0))
        pred = torch.sigmoid(x)
        return pred

    def DistMult(self, head, relation):
        head = self.bn0(head)
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        s = head * relation
        s = self.bn2(s)
        s = self.score_dropout(s)
        ans = torch.mm(s, self.embedding.weight.transpose(1,0))
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
        s = torch.mm(s, self.embedding.weight.transpose(1,0))
        s = 0.5 * s
        pred = torch.sigmoid(s)
        return pred

    def ComplEx(self, head, relation):
        '''
        :param head: (batch_size,400)
        :param relation: (batch_size,60)
        '''
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)        #分割，和cat相反
        head = self.bn0(head)           #batchnormalize
        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)       #tenser 维度调换
        re_head = head[0]       #实部
        im_head = head[1]       #虚部

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        score = self.bn2(score)
        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        pred = torch.sigmoid(score)
        return pred

    def Rotat3(self, head, relation):
        pi = 3.14159265358979323846
        relation = F.hardtanh(relation) * pi
        r = torch.stack(list(torch.chunk(relation, 3, dim=1)), dim=1)
        h = torch.stack(list(torch.chunk(head, 3, dim=1)), dim=1)
        h = self.bn0(h)
        h = self.ent_dropout(h)
        r = self.rel_dropout(r)
        
        r = r.permute(1, 0, 2)
        h = h.permute(1, 0, 2)

        x = h[0]
        y = h[1]
        z = h[2]

        # need to rotate h by r
        # r contains values in radians

        for i in range(len(r)):
            sin_r = torch.sin(r[i])
            cos_r = torch.cos(r[i])
            if i == 0:
                x_n = x.clone()
                y_n = y * cos_r - z * sin_r
                z_n = y * sin_r + z * cos_r
            elif i == 1:
                x_n = x * cos_r - y * sin_r
                y_n = x * sin_r + y * cos_r
                z_n = z.clone()
            elif i == 2:
                x_n = z * sin_r + x * cos_r
                y_n = y.clone()
                z_n = z * cos_r - x * sin_r

            x = x_n
            y = y_n
            z = z_n

        s = torch.stack([x, y, z], dim=1)        
        s = self.bn2(s)
        s = self.score_dropout(s)
        s = s.permute(1, 0, 2)
        s = torch.cat([s[0], s[1], s[2]], dim = 1)
        ans = torch.mm(s, self.embedding.weight.transpose(1,0))
        pred = torch.sigmoid(ans)
        return pred
    
    def forward(self, sentence, p_head, p_tail, question_len, q_value_tfidf, relation):
        embeds = self.word_embeddings(sentence)

        # ==================transformer based=================== #
        # rel = relation.transpose(0,1)
        # if len(rel) == 3:
        #     rel1 = self.embedding_rel(rel[0])
        #     rel2 = self.embedding_rel(rel[1])
        #     rel3 = self.embedding_rel(rel[2])
        #     trg_rel_emb1 = torch.cat((rel1, rel2), dim=1)
        #     trg_rel_emb2 = torch.cat((trg_rel_emb1, rel3), dim=1).unsqueeze(1)
        #     trg_rel_emb = self.hop3rel2Transf(trg_rel_emb2)
        #
        # else:
        #     rel1 = self.hop12rel2Transf(self.embedding_rel(rel[0]))
        #     rel2 = self.hop12rel2Transf(self.embedding_rel(rel[1]))
        #     trg_rel_emb = torch.cat((rel1, rel2), dim=1).unsqueeze(1)


        # embeds = (embeds*0) + (q_value_tfidf*1)
        packed_output = pack_padded_sequence(embeds, question_len, batch_first=True)        #question_len是问题的实际长度 ;[batch_size, seq_len, feature]
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)
        # outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        # words_embedding = self.applyNonLinear(outputs)
        # src_mask = self.get_src_mask(outputs, question_len)
        # words_TiBased_embedding = self.Transformer(words_embedding, src_mask, q_value_tfidf, trg_rel_emb, trg_mask=None).squeeze(1)
        # words_TiBased_embedding = self.transf2rel(words_TiBased_embedding)

        outputs_ti = outputs * q_value_tfidf
        outputs_ti = outputs_ti.sum(dim=1)
        rel_embedding = self.applyNonLinear(outputs_ti)                #(128,200)
        p_head = self.embedding(p_head)         #(128,400) 从预训练模型获取预训练向量  --主题词,输入index
        pred = self.getScores(p_head, rel_embedding)        #用ComplEx计算主题词和关系的相似度，返回一个预测实体
        actual = p_tail                 #p_tail 是43234个词的one-hot编码
        if self.label_smoothing:
            actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1))             #标签顺滑  是将对应词的维度的1，减小，使模型减少过拟合
        loss = self.loss(pred, actual)
        # reg = -0.001
        # best: reg is 1.0
        # self.l3_reg = 0.002
        # self.gamma1 = 1
        # self.gamma2 = 3
        if not self.freeze:
            if self.l3_reg:
                norm = torch.norm(self.embedding.weight, p=3, dim=-1)
                loss = loss + self.l3_reg * torch.sum(norm)
        return loss
        
    def get_relation_embedding(self, head, sentence, sent_len):
        embeds = self.word_embeddings(sentence.unsqueeze(0))
        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        # rel_embedding = self.hidden2rel(outputs)
        rel_embedding = self.applyNonLinear(outputs)
        return rel_embedding

    def get_score_ranked(self, head, sentence, sent_len, q_value_tfidf, relation):
        embeds = self.word_embeddings(sentence.unsqueeze(0))

        # ==================transformer based=================== #
        # rel = relation.unsqueeze(1)
        # if len(rel) == 3:
        #     rel1 = self.embedding_rel(rel[0])
        #     rel2 = self.embedding_rel(rel[1])
        #     rel3 = self.embedding_rel(rel[2])
        #     trg_rel_emb1 = torch.cat((rel1, rel2), dim=1)
        #     trg_rel_emb2 = torch.cat((trg_rel_emb1, rel3), dim=1).unsqueeze(1)
        #     trg_rel_emb = self.hop3rel2Transf(trg_rel_emb2)
        #
        # else:
        #     rel1 = self.hop12rel2Transf(self.embedding_rel(rel[0]))
        #     rel2 = self.hop12rel2Transf(self.embedding_rel(rel[1]))
        #     trg_rel_emb = torch.cat((rel1, rel2), dim=1).unsqueeze(1)

        packed_output = pack_padded_sequence(embeds, sent_len, batch_first=True)
        outputs, (hidden, cell_state) = self.GRU(packed_output)
        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)
        # outputs = torch.cat([hidden[0,:,:], hidden[1,:,:]], dim=-1)
        # rel_embedding = self.hidden2rel(outputs)
        # rel_embedding = self.applyNonLinear(outputs)

        # words_embedding = self.applyNonLinear(outputs)
        # src_mask = self.get_src_mask(outputs, sent_len)
        # words_TiBased_embedding = self.Transformer(words_embedding, src_mask, q_value_tfidf, trg_rel_emb, trg_mask=None).squeeze(1)
        # words_TiBased_embedding = self.transf2rel(words_TiBased_embedding)

        outputs_ti = outputs * q_value_tfidf
        outputs_ti = outputs_ti.sum(dim=1)
        rel_embedding = self.applyNonLinear(outputs_ti)                #(128,200)

        head = self.embedding(head).unsqueeze(0)
        score = self.getScores(head, rel_embedding)
        
        top2 = torch.topk(score, k=2, largest=True, sorted=True)
        return top2

    def q_embed_with_tfidf(self, qword_to_freq, qword_embedsings):

        return
        




