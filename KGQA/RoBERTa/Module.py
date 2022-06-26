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


class StepRelation(nn.Module):
    def __init__(self):
        pass

    def init_step(self):
        pass

    def get_cur_hop_relation(self):
        pass

    def predict_question_hops(self):
        pass

    def foward(self):
        pass


class TransformerBasedFiltering(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass