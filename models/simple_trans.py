import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import pdb
import math

# TODO: put in utils
TRANS_VEC_DIM = {
    "raw_absolute": 24,
    "aligned_absolute": 18,
    "aligned_relative": 12
}

class SimpleTransModel(nn.Module):

    def __init__(self, feat_dim, predicate_dim, trans_vec, with_class, object_dim):
        """
        :param feat_dim:
        :param predicate_dim: number of predicates
        :param trans_vec: list of transformation vectors used
        :param with_class: whether or not to fuse class information
        :param object_dim:
        """
        super().__init__()
        self.predicate_dim = predicate_dim
        self.trans_vec = trans_vec
        self.with_class = with_class
        self.object_dim = object_dim
        assert len(self.trans_vec) > 0
        assert len(set(self.trans_vec)) == len(self.trans_vec)

        inp_size = 0
        for x in trans_vec:
            inp_size +=  TRANS_VEC_DIM[x]

        self.model_tv = nn.Sequential(
            nn.Linear(inp_size, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
        )

        if with_class:
            self.model_subj = nn.Sequential(
                nn.Linear(self.object_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
            )

            self.model_obj = nn.Sequential(
                nn.Linear(self.object_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
            )

        self.model_end = self.obj = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, predicate_dim),
        )

    def forward(self, feat_dict, predi):
        feat_tv = torch.cat([feat_dict[x] for x in self.trans_vec], 1)
        feat_tv = self.model_tv(feat_tv)

        if self.with_class:
            feat_subj = F.one_hot(feat_dict['subj'], num_classes=self.object_dim).to(torch.float32)
            feat_subj = self.model_subj(feat_subj)

            feat_obj = F.one_hot(feat_dict['obj'], num_classes=self.object_dim).to(torch.float32)
            feat_obj = self.model_obj(feat_obj)

            feat_tv = feat_subj + feat_obj + feat_tv

        feat = self.model_end(feat_tv)

        predi_onehot = F.one_hot(predi, num_classes=self.predicate_dim).to(torch.float32)
        return torch.sum(feat * predi_onehot, 1)
