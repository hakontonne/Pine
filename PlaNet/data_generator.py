import random

import numpy as np
import torch
from torch import nn
import torch.functional as F

import env_data
from transformers import BertModel, BertTokenizer

class Bert(nn.Module):

    def __init__(self, device):
        super(Bert, self).__init__()
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.bert = BertModel.from_pretrained('bert-large-cased').eval().to(self.device)

    def forward(self, x):
        tokens = self.tokenizer(x, padding=True, return_tensors='pt').to(self.device)

        return self.bert(**tokens)[0][:, 0, :]


def generate_positive_sample(bert_model):
    x = []
    y = []
    for env in env_data.official_envs.keys():
        target = env_data.official_envs[env]
        target_embedding = bert_model(target)
        poitives = env_data.other_envs[env]
        for p in poitives:
            p_embedding = bert_model(p)
            x.append(torch.cat((target_embedding, p_embedding)))
            y.append(1)

    return x, y



def generate_negative_sample(bert_model, n_samples_per):
    x = []
    y = []


    for env in env_data.official_envs.keys():
        target = env_data.official_envs[env]
        target_embedding = bert_model(target)
        negatives = list(env_data.other_envs.keys())
        if target in negatives: negatives.remove(target)

        for _ in range(n_samples_per):
            l = random.choice(negatives)
            negative_embedding = bert_model(random.choice(env_data.other_envs[l]))

            x.append(torch.cat((target_embedding, negative_embedding)))
            y.append(0)

    return x, y







