# %%
import torch
import torch.nn.functional as F
import torch.nn as nn

from pretrainedmodels import xception
# Also try densenet, efficientnet, swin, etc.

# %%
class XceptionModel(nn.Module):
    def __init__(self, num_classes, pretrained, n_hidden_nodes, dropout_rate, freeze_backbone=False):
        super(XceptionModel, self).__init__()
        self.n_hidden_nodes = n_hidden_nodes
        self.dropout = nn.Dropout(p=dropout_rate)

        # INPUT_SIZE = 3 x 299 x 299
        self.backbone              = xception(num_classes=1000, pretrained='imagenet' if pretrained else False) 
        self.backbone.last_linear  = nn.Identity() # Outputs 2048

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        n_backbone_out          = 2048
        self.target_size        = 299

        if self.n_hidden_nodes is None:
            self.img_fc1 = nn.Linear(n_backbone_out, num_classes)
        else:
            self.img_fc1 = nn.Linear(n_backbone_out, n_hidden_nodes) # 2048 to 4096
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(n_hidden_nodes, n_hidden_nodes//2) # 4096 to 2048
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(n_hidden_nodes//2, n_hidden_nodes//4) # 2048 to 1024
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(n_hidden_nodes//4, num_classes) # 1024 to 100
            self.relu4 = nn.ReLU()

    def forward(self, x):
        if self.n_hidden_nodes is not None:
            out = self.backbone(x)
            out = self.dropout(out)
            out = self.img_fc1(out)
            out = self.relu1(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.dropout(out)
            out = self.fc3(out)
            out = self.relu4(out)
            out = self.fc4(out) # No activation and no softmax at the end (contained in F.cross_entropy())
        else:
            out = self.backbone(x)
            out = self.dropout(out)
            out = self.img_fc1(out)
        return out
# %%
class EfficientNetB0(nn.Module):
    ''' Building in progress... '''
    def __init__(self, pretrained, n_hidden_nodes, dropout_rate):
        super(EfficientNetB0, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        num_classes  = 100

        self.backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=pretrained)
        self.target_size = 299
        n_backbone_out = 1000

        self.img_fc1 = nn.Linear(n_backbone_out, n_hidden_nodes//2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden_nodes//2, n_hidden_nodes//4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(n_hidden_nodes//4, num_classes)
        self.relu3 = nn.ReLU()
    
    def forward(self, x):
        out = self.backbone(x)
        out = self.dropout(out)
        out = self.img_fc1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc3(out) # No activation and no softmax at the end (contained in F.cross_entropy())
        return out

# %%
if __name__=='__main__':
    model = EfficientNetB0(pretrained=True, n_hidden_nodes=1024, dropout_rate=0.5)

    print(model)

