import torch.nn.functional as F
import torch
import torch.nn as nn
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=1.8, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
def contrastive_loss(features, pos_pairs, neg_pairs, margin=2):
    # 正样本对损失
    if len(pos_pairs) > 0:
        pos_indices = torch.tensor(pos_pairs, dtype=torch.long, device=features.device)
        pos_distances = torch.norm(features[pos_indices[:, 0]] - features[pos_indices[:, 1]], dim=1, p=2)
        pos_loss = torch.sum(pos_distances ** 2)
    else:
        pos_loss = torch.tensor(0.0, device=features.device)
    # 负样本对损失
    if len(neg_pairs) > 0:
        neg_indices = torch.tensor(neg_pairs, dtype=torch.long, device=features.device)
        neg_distances = torch.norm(features[neg_indices[:, 0]] - features[neg_indices[:, 1]], dim=1, p=2)
        neg_loss = torch.sum(torch.clamp(margin - neg_distances, min=0.0) ** 2)
    else:
        neg_loss = torch.tensor(0.0, device=features.device)
    # 平均化损失
    total_pairs = len(pos_pairs) + len(neg_pairs)
    return (pos_loss + neg_loss) / total_pairs if total_pairs > 0 else torch.tensor(0.0, device=features.device)