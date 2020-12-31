import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import pdb
import math


class VTransE(nn.Module):

    def __init__(self, feature_dim, visual_feature_size, predicate_embedding_dim, predicate_dim, object_dim, pretrained,
                 backbone):
        super().__init__()

        self.object_dim = object_dim
        self.predicate_dim = predicate_dim
        self.visual_feature_size = visual_feature_size

        self.backbone = torchvision.models.__dict__[backbone](pretrained=pretrained)
        self.backbone = nn.Sequential(self.backbone.conv1,
                                      self.backbone.bn1,
                                      self.backbone.relu,
                                      self.backbone.maxpool,
                                      self.backbone.layer1,
                                      self.backbone.layer2,
                                      self.backbone.layer3,
                                      self.backbone.layer4
                                     )


        self.scale_factor = nn.Parameter(torch.empty(3))
        nn.init.uniform_(self.scale_factor)

        self.linear1 = nn.Linear(visual_feature_size * visual_feature_size * feature_dim, visual_feature_size * visual_feature_size * 64)
        self.batchnorm1 = nn.BatchNorm1d(visual_feature_size * visual_feature_size * 64)
        self.linear2 = nn.Linear(visual_feature_size * visual_feature_size * feature_dim, visual_feature_size * visual_feature_size * 64)
        self.batchnorm2 = nn.BatchNorm1d(visual_feature_size * visual_feature_size * 64)

        # one hot feature + location features + visual features
        _feature_dim = object_dim + 4 + (visual_feature_size * visual_feature_size * 64)
        self.W_o = nn.Linear(_feature_dim, predicate_embedding_dim)
        self.W_s = nn.Linear(_feature_dim, predicate_embedding_dim)
        self.W_p = nn.Linear(predicate_embedding_dim, predicate_dim)


    def forward(self, subj, obj, full_im, t_s, t_o, bbox_s, bbox_o, predi):
        img_feature_map = self.backbone(full_im)
        subj_img_feature = []
        obj_img_feature = []
        for i in range(bbox_s.size(0)):
            bbox_subj = self.fix_bbox(7 * bbox_s[i], 7)
            bbox_obj = self.fix_bbox(7 * bbox_o[i], 7)
            subj_img_feature.append(
                F.upsample(
                    img_feature_map[
                        i : (i + 1), :,
                        bbox_subj[0] : bbox_subj[1], bbox_subj[2] : bbox_subj[3]
                    ],
                    self.visual_feature_size, mode='bilinear')
            )
            obj_img_feature.append(
                F.upsample(
                    img_feature_map[
                        i : (i + 1), :,
                        bbox_obj[0] : bbox_obj[1], bbox_obj[2] : bbox_obj[3]
                    ],
                    self.visual_feature_size, mode='bilinear')
            )
        subj_img_feature = torch.cat(subj_img_feature)
        obj_img_feature = torch.cat(obj_img_feature)
        subj_img_feature = subj_img_feature.view(subj_img_feature.size(0), -1)
        obj_img_feature = obj_img_feature.view(obj_img_feature.size(0), -1)
        subj_img_feature = F.relu(self.batchnorm1(self.linear1(subj_img_feature)))
        obj_img_feature = F.relu(self.batchnorm2(self.linear2(obj_img_feature)))

        subj = F.one_hot(subj, num_classes=self.object_dim).to(torch.float32)
        obj = F.one_hot(obj, num_classes=self.object_dim).to(torch.float32)

        x_s = torch.cat([
            subj * self.scale_factor[0],
            t_s * self.scale_factor[1],
            subj_img_feature * self.scale_factor[2]
        ], 1)
        x_o = torch.cat([
            obj * self.scale_factor[0],
            t_o * self.scale_factor[1],
            obj_img_feature *  self.scale_factor[2]
        ], 1)


        v_s = F.relu(self.W_s(x_s))
        v_o = F.relu(self.W_o(x_o))

        predi_onehot = F.one_hot(predi, num_classes=self.predicate_dim).to(torch.float32)
        return torch.sum(self.W_p(v_o - v_s) * predi_onehot, 1)


    def fix_bbox(self, bbox, size):
        assert size > 0
        new_bbox = [int(bbox[0]), min(size, int(math.ceil(bbox[1]))),
                    int(bbox[2]), min(size, int(math.ceil(bbox[3])))]
        assert 0 <= new_bbox[0] <= new_bbox[1] <= size and 0 <= new_bbox[2] <= new_bbox[3] <= size

        # taking care of degenerate case
        # arising because a few objects have height/width = 0
        if new_bbox[0] == new_bbox[1]:
            if new_bbox[1] < size:
                new_bbox[1] += 1
            else:
                new_bbox[0] -= 1

        if new_bbox[2] == new_bbox[3]:
            if new_bbox[3] < size:
                new_bbox[3] += 1
            else:
                new_bbox[2] -= 1

        return new_bbox
