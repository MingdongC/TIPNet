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
from transformers import RobertaModel

class PruningModel(nn.Module):

    def __init__(self, rel2idx, idx2rel, ls):
        super(PruningModel, self).__init__()
        self.label_smoothing = ls
        self.rel2idx = rel2idx
        self.idx2rel = idx2rel

        self.roberta_pretrained_weights = '../hfl/model/roberta/roberta-base'
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_pretrained_weights)

        self.roberta_dim = 768
        self.mid1 = 512
        self.mid2 = 512
        self.mid3 = 256
        self.mid4 = 256
        self.fcnn_dropout = torch.nn.Dropout(0.1)
        # self.lin1 = nn.Linear(self.roberta_dim, self.mid1)
        # self.lin2 = nn.Linear(self.mid1, self.mid2)
        # self.lin3 = nn.Linear(self.mid2, self.mid3)
        # self.lin4 = nn.Linear(self.mid3, self.mid4)
        # self.hidden2rel = nn.Linear(self.mid4, len(self.rel2idx))
        self.hidden2rel = nn.Linear(self.roberta_dim, len(self.rel2idx))

        self.loss = torch.nn.BCELoss(reduction='sum')       # BCELoss多标签多分类损失

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)        

    def applyNonLinear(self, outputs):
        # outputs = self.fcnn_dropout(self.lin1(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin2(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin3(outputs))
        # outputs = F.relu(outputs)
        # outputs = self.fcnn_dropout(self.lin4(outputs))
        # outputs = F.relu(outputs)
        outputs = self.hidden2rel(outputs)
        # outputs = self.hidden2rel_base(outputs)
        return outputs
    '''
    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_last_hidden_states = self.roberta_model(question_tokenized, attention_mask=attention_mask)[0]
        states = roberta_last_hidden_states.transpose(1,0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        # question_embedding = torch.mean(roberta_last_hidden_states, dim=1)
        return question_embedding
    '''

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

    def forward(self, question_tokenized, attention_mask, rel_one_hot):
        question_embedding, words_embedding, words_mask = self.getQuestionEmbedding(question_tokenized, attention_mask)
        prediction = self.applyNonLinear(question_embedding)
        prediction = torch.sigmoid(prediction)
        actual = rel_one_hot
        if self.label_smoothing:
            actual = ((1.0-self.label_smoothing)*actual) + (1.0/actual.size(1)) 
        loss = self.loss(prediction, actual)
        return loss

    def get_score_ranked(self, question_tokenized, attention_mask):
        question_embedding = self.getQuestionEmbedding(question_tokenized.unsqueeze(0), attention_mask.unsqueeze(0))
        prediction = self.applyNonLinear(question_embedding)
        prediction = torch.sigmoid(prediction).squeeze()
        # top2 = torch.topk(scores, k=2, largest=True, sorted=True)
        # return top2
        return prediction
        




