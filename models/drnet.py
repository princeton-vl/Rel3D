import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import pdb
import math
from models.layers import HourglassKaiyu as Hourglass

class DRNet(nn.Module):

    def __init__(self, feature_dim, predicate_dim, object_dim, pretrained, dropout, num_layers, backbone,
                 two_stream, rgbd, only_2d, only_appr):
        super(DRNet, self).__init__()
        self.num_layers = num_layers
        self.predicate_dim = predicate_dim
        self.object_dim = object_dim
        self.dropout = dropout
        # if both two_stream and rgdb are false, we use only rgb input
        self.two_stream = two_stream
        self.rgbd = rgbd
        # if both only_2d and only_appr are false, then both streams are used
        self.only_2d = only_2d
        self.only_appr = only_appr

        if self.only_2d:
            assert not (self.rgbd or self.two_stream)
        assert not (self.rgbd and self.two_stream)

        output_size = 0
        if not self.only_appr:
            self.pos_module = nn.Sequential(OrderedDict([
                ('conv1_p', nn.Conv2d(2, 32, 5, 2, 2)),
                ('normalize1_p', nn.BatchNorm2d(32)),
                ('relu1_p', nn.ReLU()),
                ('conv2_p', nn.Conv2d(32, 64, 3, 1, 1)),
                ('normalize2_p', nn.BatchNorm2d(64)),
                ('relu2_p', nn.ReLU()),
                ('maxpool2_p', nn.MaxPool2d(2)),
                ('hg', Hourglass(8, 64)),
                ('normalize_p', nn.BatchNorm2d(64)),
                ('relu_p', nn.ReLU()),
                ('maxpool_p', nn.MaxPool2d(2)),
                ('conv3_p', nn.Conv2d(64, 256, 4)),
                ('normalize3_p', nn.BatchNorm2d(256)),
            ]))
            output_size += 256

        if not self.only_2d:
            if self.rgbd:
                if pretrained:
                    print("Using pretrained model with RGBD input, this will work but might not be the correct strategy")

                _module = torchvision.models.__dict__[backbone](pretrained=pretrained)
                _module.fc = nn.Linear(512, 256)
                self.appr_module = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(4, 4, 3, 1, 1)),
                    ('normalize1', nn.BatchNorm2d(4)),
                    ('relu1', nn.ReLU()),
                    ('conv2', nn.Conv2d(4, 4, 3, 1, 1)),
                    ('normalize2', nn.BatchNorm2d(4)),
                    ('relu2', nn.ReLU()),
                    ('conv3', nn.Conv2d(4, 3, 3, 1, 1)),
                    ('normalize3', nn.BatchNorm2d(3)),
                    ('relu3', nn.ReLU()),
                    ('model', _module)
                ]))
                output_size += 256

            elif self.two_stream:
                self.appr_module = torchvision.models.__dict__[backbone](pretrained=pretrained)
                self.appr_module.fc = nn.Linear(512, 256)
                self.depth_module = torchvision.models.__dict__[backbone](pretrained=False)
                self.depth_module.fc = nn.Linear(512, 256)
                output_size += 512

            else:
                self.appr_module = torchvision.models.__dict__[backbone](pretrained=pretrained)
                self.appr_module.fc = nn.Linear(512, 256)
                output_size += 256

        self.PhiR_0 = nn.Linear(output_size, feature_dim)
        self.normalize = nn.BatchNorm1d(feature_dim)

        self.PhiA = nn.Linear(object_dim, feature_dim)
        self.PhiB = nn.Linear(object_dim, feature_dim)
        self.PhiR = nn.Linear(feature_dim, feature_dim)

        self.fc = nn.Linear(feature_dim, predicate_dim)

    def forward(self, subj, obj, img_crop, bbox_mask, predi, depth_crop):
        if not self.only_appr:
            pos_feature = self.pos_module(bbox_mask)
            if pos_feature.size(0) == 1:
                pos_feature = torch.unsqueeze(torch.squeeze(pos_feature), 0)
            else:
                pos_feature = torch.squeeze(pos_feature)
            pos_feature = F.dropout(pos_feature, p=self.dropout, training=self.training)
        else:
            pos_feature = 0

        if not self.only_2d:
            if self.rgbd:
                appr_feature = self.appr_module(
                    torch.cat([img_crop, depth_crop], 1))
            elif self.two_stream:
                appr_feature = self.appr_module(img_crop)
                depth_feature = self.depth_module(depth_crop.expand(-1, 3, 224, 224))
                appr_feature = torch.cat([appr_feature, depth_feature], -1)
            else:
                appr_feature = self.appr_module(img_crop)

            appr_feature = F.dropout(appr_feature, p=self.dropout, training=self.training)
        else:
            appr_feature = 0

        if self.only_2d:
            final_features = pos_feature
        elif self.only_appr:
            final_features = appr_feature
        else:
            final_features = torch.cat([appr_feature, pos_feature], 1)

        qr = F.relu(self.normalize(self.PhiR_0(final_features)))

        qa = F.one_hot(subj, num_classes=self.object_dim).to(torch.float32)
        qb = F.one_hot(obj, num_classes=self.object_dim).to(torch.float32)
        for i in range(self.num_layers):
            qr = F.relu(self.PhiA(qa) + self.PhiB(qb) + self.PhiR(qr))

        qr = self.fc(qr)

        predi_onehot = F.one_hot(predi, num_classes=self.predicate_dim).to(torch.float32)
        return torch.sum(qr * predi_onehot, 1)
