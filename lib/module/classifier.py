import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from lib.module.fc import FCNet

class SimpleClassifier(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, dropout, wn=True):
        super(SimpleClassifier, self).__init__()
        layers = [
            FCNet([in_dim, hid_dim], relu=True, wn=wn),
            nn.Dropout(dropout),
            FCNet([hid_dim, out_dim], relu=False, wn=wn)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
