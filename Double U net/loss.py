import torch.nn as nn
import torch


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-15

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = torch.flatten(y_pred)
        y_true = torch.flatten(y_true)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
class DiceLoss_v2(nn.Module):
    def __init__(self):
        super(DiceLoss_v2, self).__init__()
        self.epsilon = 1e-5
    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        
        pre = predict.view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum()  #利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score
