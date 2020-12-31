import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import pdb
import math

class SimpleLanguageModel(nn.Module):

    def __init__(self, feature_dim, predicate_dim, object_dim):
        super().__init__()
        self.predicate_dim = predicate_dim
        self.object_dim = object_dim

        self.linear1 = nn.Linear(object_dim, feature_dim)
        self.normalize1 = nn.BatchNorm1d(feature_dim)
        self.linear2 = nn.Linear(object_dim, feature_dim)
        self.normalize2 = nn.BatchNorm1d(feature_dim)
        self.linear3 = nn.Linear(predicate_dim, feature_dim)
        self.normalize3 = nn.BatchNorm1d(feature_dim)
        self.linear4 = nn.Linear(feature_dim, feature_dim // 2)
        self.normalize4 =  nn.BatchNorm1d(feature_dim // 2)
        self.linear5 = nn.Linear(feature_dim // 2, 1)


    def forward(self, subj, obj, predi):
        subj_feature = self.linear1(F.one_hot(subj, num_classes=self.object_dim).to(torch.float32))
        subj_feature = self.normalize1(subj_feature)
        subj_feature = F.relu(subj_feature)

        obj_feature = self.linear2(F.one_hot(obj, num_classes=self.object_dim).to(torch.float32))
        obj_feature = self.normalize2(obj_feature)
        obj_feature = F.relu(obj_feature)

        predi_feature = self.linear3(F.one_hot(predi, num_classes=self.predicate_dim).to(torch.float32))
        predi_feature = self.normalize3(predi_feature)
        predi_feature = F.relu(predi_feature)

        fused_feature = subj_feature + predi_feature + obj_feature
        fused_feature = self.linear4(fused_feature)
        fused_feature = self.normalize4(fused_feature)
        fused_feature = F.relu(fused_feature)

        logit = self.linear5(fused_feature).squeeze()

        return logit