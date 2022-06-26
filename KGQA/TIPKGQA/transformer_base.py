import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
import numpy as np


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6,
                 n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200, emb_src_trg_weight_sharing=True,
                 trg_emb_prj_weight_sharing=True, scale_emb_or_prj = 'prj', Ti_value=True):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model
        print("use ti value:", Ti_value)

        self.encoder = Encoder(d_word_vec=d_word_vec, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                               d_model=d_model, d_inner=d_inner, pad_idx=src_pad_idx, dropout=dropout,
                               n_position=n_position, scale_emb=scale_emb, Ti_value=True)
        self.decoder = Decoder(d_word_vec=d_word_vec, n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                               d_model=d_model, d_inner=d_inner, pad_idx=src_pad_idx, dropout=dropout,
                               n_position=n_position, scale_emb=scale_emb)

        assert d_model == d_word_vec


    def forward(self, src_emb, src_mask_data, words_ti_value, trg_emb, trg_mask_data):

        src_mask = get_pad_mask(src_mask_data, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_mask_data, self.trg_pad_idx)
        enc_output = self.encoder(src_emb, src_mask, words_ti_value)
        dec_output = self.decoder(trg_emb, trg_mask, enc_output, src_mask)

        return dec_output

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

class Decoder(nn.Module):
    def __init__(self, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner
                 , pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.position_enc = Positonal_Encoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([Decoder_layer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                                          for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model


    def forward(self, trg_emb, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = trg_emb
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list = [dec_slf_attn] if return_attns else []
            dec_enc_attn_list = [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output

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
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):

        super(Decoder_layer, self).__init__()

        self.slf_attn = Multi_Head_Attention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = Multi_Head_Attention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = Position_wise_Feed_Forward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn

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

class Tf_idf_Value(nn.Module):
    def __init__(self):
        super(Tf_idf_Value, self).__init__()

    def forward(self, x, words_ti_value):
        #a = x.detach().numpy()
        #b = words_ti_value.detach().numpy()
        return x * words_ti_value.clone().detach()
