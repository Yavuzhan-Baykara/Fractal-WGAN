import torch

class BCEWithLogitsLoss:
    def __init__(self):
        pass

    def __call__(self, predictions, targets):
        epsilon = 1e-12
        logits = 1 / (1 + torch.exp(-predictions + epsilon))
        loss = -(targets * torch.log(logits) + (1 - targets) * torch.log(1 - logits)).mean()
        return loss
