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

class RelationExtractor(nn.Module):

    def __init__(self, embedding_dim, relation_dim, num_entities, pretrained_embeddings, device, entdrop, reldrop, scoredrop, l3_reg, model, ls, do_batch_norm, freeze=True):
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

        if self.model in ['DistMult', 'TuckER', 'RESCAL', 'SimplE']:
            self.bn0 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
            self.bn2 = torch.nn.BatchNorm1d(self.embedding.weight.size(1))
        else:
            self.bn0 = torch.nn.BatchNorm1d(multiplier)
            self.bn2 = torch.nn.BatchNorm1d(multiplier)


        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)        
        self._klloss = torch.nn.KLDivLoss(reduction='sum')
        self.Transformer = Transformer(src_pad_idx=0, d_word_vec=512, d_inner=2048, n_layers=6, n_head=8, d_k=64,
                                       d_v=64, dropout=0.1, n_position=200)
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
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            head = self.bn0(head)

        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.embedding.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.bn2(score)

        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        # pred = torch.sigmoid(score)
        pred = score
        return pred

    def del_tensor_ele(self, arr, index, dim):
        arr1 = arr[1:index+1]
        arr2 = arr[index+3:]
        arr = torch.cat((arr1,arr2), dim=dim)
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
            if bat ==0 :
                words_embedding_all = words_embedding1
                words_mask_all = words_mask1
            else:
                words_embedding_all = torch.cat((words_embedding_all,words_embedding1), dim=0)
                words_mask_all = torch.cat((words_mask_all,words_mask1), dim=0)
        words_mask = words_mask_all
        words_embedding = words_embedding_all
        return question_embedding, words_embedding, words_mask

    def getWordsEmbedding(self,words_tokenized, attention_masks):
        src_mask = words_tokenized[..., 1]
        words_tokenized = words_tokenized.transpose(1,0)
        attention_masks = attention_masks.transpose(1,0)
        for idx in range(words_tokenized.size()[0]):
            word_tokenized = words_tokenized[idx]
            attention_mask = attention_masks[idx]
            roberta_last_hidden_states = self.roberta_model(word_tokenized,attention_mask=attention_mask)[0]
            states = roberta_last_hidden_states.transpose(1,0)
            cls_embedding = states[0]
            word_embedding = cls_embedding.unsqueeze(0)
            if idx == 0:
                words_embedding = word_embedding
            else:
                words_embedding = torch.cat((words_embedding,word_embedding),0)
        words_embedding = words_embedding.transpose(1,0)
        return  words_embedding, src_mask

    def words_ti_value_haddle(self, words_ti_value, question_tokenized):
        word_mark = "<s> </s>"
        word_mark = self.tokenizer.tokenize(word_mark)[1]
        for bat in range(question_tokenized.size(0)):
            tokens = question_tokenized[bat]
            tokens = tokens[1:]
            tokens = self.tokenizer.convert_ids_to_tokens(tokens)
            ti_value = words_ti_value[bat]
            for a in range(len(ti_value)):
                temp = torch.full((1,512), ti_value[a], dtype=torch.float32)
                if a == 0:
                    temp11 = temp
                else:
                    temp11 = torch.cat((temp11, temp), dim=0)
            ti_value = temp11
            result = pre_ti_value = ti_value[0].unsqueeze(0)
            cur_ti_value = ti_value[1].unsqueeze(0)
            idxForTi = 2
            ti_one = torch.full((1, 512), 0, dtype=torch.float32)
            for x in range(len(tokens)-1):
                a = tokens[x+1]
                if tokens[x+1]!=word_mark and tokens[x+1]!='</s>' and tokens[x+1]!='<pad>':
                    if word_mark in tokens[x+1]:
                        result = torch.cat((result, cur_ti_value), dim=0)
                        pre_ti_value = cur_ti_value
                        cur_ti_value = ti_value[idxForTi].unsqueeze(0)
                        idxForTi += 1
                    else:
                        result = torch.cat((result, pre_ti_value), dim=0)
                elif tokens[x]=='</s>' or tokens[x]==word_mark:
                    pass
                else:
                    result =torch.cat((result, ti_one), dim=0)
            result = result.unsqueeze(0)
            if bat == 0:
                results = result
            else:
                results = torch.cat((results, result),dim=0)

        return results

    def forward(self, question_tokenized, attention_mask, p_head, p_tail, words_ti_value):
        question_embedding, words_embedding, src_mask = self.getQuestionEmbedding(question_tokenized, attention_mask)
        words_embedding = self.bert2tranf(words_embedding)
        words_ti_value = self.words_ti_value_haddle(words_ti_value, question_tokenized)
        words_TiBased_embedding = self.Transformer(words_embedding, src_mask, words_ti_value)
        #a = words_TiBased_embedding.transpose(1,0)[1].detach().numpy()
        #b = torch.sum(words_TiBased_embedding, dim=1).detach().numpy()


        #words_TiBased_embedding *= words_ti_value
        mean = words_TiBased_embedding.size(1)
        temp = torch.sum(words_TiBased_embedding, dim=1, keepdim=True)/mean
        c = temp.detach().numpy()
        #rel_embedding = self.applyNonLinear(words_TiBased_embedding)
        rel_embedding = self.applyNonLinear(temp)
        p_head = self.embedding(p_head)
        pred = self.getScores(p_head, rel_embedding)
        actual = p_tail
        if self.label_smoothing:
            actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1)) 
        loss = self.loss(pred, actual)
        if not self.freeze:
            if self.l3_reg:
                norm = torch.norm(self.embedding.weight, p=3, dim=-1)
                loss = loss + self.l3_reg * torch.sum(norm)
        return loss
        

    def get_score_ranked(self, head, question_tokenized, attention_mask, words_ti_value):
        question_embedding, words_embedding, src_mask = self.getQuestionEmbedding(question_tokenized.unsqueeze(0), attention_mask.unsqueeze(0))
        words_embedding = self.bert2tranf(words_embedding)
        words_ti_value = self.words_ti_value_haddle(words_ti_value.unsqueeze(0), question_tokenized.unsqueeze(0))
        words_TiBased_embedding = self.Transformer(words_embedding, src_mask, words_ti_value)

        words_TiBased_embedding *= words_ti_value
        temp = torch.sum(words_TiBased_embedding, dim=1)
        rel_embedding = self.applyNonLinear(temp)
        #words_TiBased_embedding = words_TiBased_embedding.transpose(1,0)[1]
        #rel_embedding = self.applyNonLinear(words_TiBased_embedding)
        head = self.embedding(head).unsqueeze(0)
        scores = self.getScores(head, rel_embedding)
        # top2 = torch.topk(scores, k=2, largest=True, sorted=True)
        # return top2
        return scores


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6,
                 n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200, emb_src_trg_weight_sharing=True,
                 trg_emb_prj_weight_sharing=True, scale_emb_or_prj = 'prj', Ti_value=True):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model
        print("use ti value:", Ti_value)

        self.encoder = Encoder(d_word_vec=d_word_vec, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                               d_model=d_model, d_inner=d_inner, pad_idx=src_pad_idx, dropout=dropout,
                               n_position=n_position, scale_emb=scale_emb, Ti_value=True)

        assert d_model == d_word_vec


    def forward(self, src_emb, src_mask_data, words_ti_value):

        src_mask = get_pad_mask(src_mask_data,self.src_pad_idx)
        enc_output = self.encoder(src_emb, src_mask, words_ti_value)

        return enc_output

class Encoder(nn.Module):
    def __init__(self, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner
                 , pad_idx, dropout=0.1, n_position=200, scale_emb=False, Ti_value=True):
        super().__init__()
        self.position_enc = Positonal_Encoding(d_word_vec, n_position=n_position)
        self.words_ti_value = Tf_idf_Value()
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([Encoder_Layer(d_model, d_inner, n_head, d_k,
                                                        d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.Ti_value = Ti_value

    def forward(self, src_emb, src_mask, words_ti_value, return_attention=False):

        enc_self_attn_list = []
        enc_output = src_emb
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.position_enc(enc_output)
        if self.Ti_value:
            enc_output = self.dropout(self.words_ti_value(enc_output, words_ti_value))
        else:
            enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_self_attn = enc_layer(enc_output, self_attn_mask=src_mask)
            enc_self_attn_list += [enc_self_attn] if return_attention else []

        if return_attention:
            return enc_output, enc_self_attn_list
        return enc_output

class Positonal_Encoding(nn.Module):
    def __init__(self, d_hidden, n_position=200):
        super(Positonal_Encoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hidden))

    def _get_sinusoid_encoding_table(self, n_position, d_hidden):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hidden) for hid_j in range(d_hidden)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])       #n_position：一个序列中单词的最多位置数
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()       #x={batch_size, words, dim} ; pos_table(tensor)={1, n_position, dim}

class Tf_idf_Value(nn.Module):
    def __init__(self):
        super(Tf_idf_Value, self).__init__()

    def forward(self, x, words_ti_value):
        #a = x.detach().numpy()
        #b = words_ti_value.detach().numpy()
        return x * words_ti_value.clone().detach()

class Decoder(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class Encoder_Layer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(Encoder_Layer,self).__init__()
        self.attention_self = Multi_Head_Attention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.position_ff = Position_wise_Feed_Forward(d_model, d_inner, dropout=dropout)

    def forward(self,encoder_input, self_attn_mask=None):
        encoder_output, encoder_self_attention = self.attention_self(encoder_input, encoder_input,
                                                                     encoder_input, mask=self_attn_mask)
        encoder_output = self.position_ff(encoder_output)
        return encoder_output, encoder_self_attention

class Decoder_layer(nn.Module):
    def __init__(self):
        pass

class Multi_Head_Attention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.weight_q = nn.Linear(d_model, n_head*d_k, bias=False)
        self.weight_k = nn.Linear(d_model, n_head*d_k, bias=False)
        self.weight_v = nn.Linear(d_model, n_head*d_v, bias=False)
        self.fc = nn.Linear(n_head*d_v, d_model, bias=False)

        self.attenttion = ScaledDotProductAttention(temperature= d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        res_net = q

        q = self.weight_q(q).view(sz_b, len_q, n_head, d_k)
        k = self.weight_k(k).view(sz_b, len_k, n_head, d_k)
        v = self.weight_v(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask  = mask.unsqueeze(1)
        q, attention  = self.attenttion(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += res_net
        q = self.layer_norm(q)

        return q, attention

class Position_wise_Feed_Forward(nn.Module):

    def __init__(self, d_input, d_hidden, dropout=0.1):
        super().__init__()
        self.weight_1 = nn.Linear(d_input, d_hidden)
        self.weight_2 = nn.Linear(d_hidden, d_input)
        self.layer_norm = nn.LayerNorm(d_input, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res_net = x
        x = self.weight_2(F.relu(self.weight_1(x)))
        x = self.dropout(x)
        x += res_net
        x = self.layer_norm(x)

        return x

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
