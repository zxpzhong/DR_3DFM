# from efficientnet_pytorch import EfficientNet
from model.effnet import EfficientNet
import torch.nn.functional as F
import torch.nn as nn
import torch
from model.arcface import ArcMarginProduct,AddMarginProduct,SphereProduct

class efficientnet(nn.Module):
    def __init__(self, num_classes=1000,name = 'efficientnet-b1'):
        super(efficientnet, self).__init__()
        self.encoder = EfficientNet.from_pretrained(name)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        # self.classifier = ArcMarginProduct(1280, num_classes, s=30, m=0.1, easy_margin=True)
        self.classifier = nn.Linear(1280,num_classes)

    def forward(self, x,label=None,label_coarse2=None,f = False):
        # shallow_f: metric
        feature = self.encoder.extract_features(x)
        feature_last = self.GAP(feature)
        feature_last = feature_last.view(feature_last.size(0), -1)
        logit = self.classifier(feature_last)
        return logit


    def extract_feature(self,x):
        feature = self.encoder.extract_features(x)
        feature_last = self.GAP(feature)
        feature_last = feature_last.view(feature_last.size(0), -1)
        return feature_last