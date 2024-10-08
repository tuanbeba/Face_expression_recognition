import timm
import torch
from torch import nn

class FaceExpModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.eff_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes = 7)

    def forward(self, images, labels=None):
        logits = self.eff_net(images)

        if labels !=None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return logits, loss

        return logits
    

if __name__=='__main__':
    model = FaceExpModel()
    x = torch.randn(1,3,48,48)
    print('Shape of output = ', model(x).shape)

