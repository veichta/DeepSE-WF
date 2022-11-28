import logging

import torch
from torch import nn


class TripletNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        anchor = self.embedding_net(anchor)
        positive = self.embedding_net(positive)
        negative = self.embedding_net(negative)
        return anchor, positive, negative


def triplet_cosine_loss(output):
    """Compute cosine triplet loss.

    Args:
        output: Output of the model
        target: Target labels

    Returns:
        loss: Cosine triplet loss
    """
    anchor, positive, negative = output

    positive_dist = 1 - nn.CosineSimilarity()(anchor, positive)
    negative_dist = 1 - nn.CosineSimilarity()(anchor, negative)

    return torch.mean(torch.clamp(0.1 + positive_dist - negative_dist, min=0.0))


def triplet_l2_loss(output):
    """Compute L2 distance loss.

    Args:
        output: Output of the model
        target: Target labels

    Returns:
        loss: L2 distance loss
    """
    anchor, positive, negative = output

    positive_dist = torch.sum((anchor - positive) ** 2, dim=1)
    negative_dist = torch.sum((anchor - negative) ** 2, dim=1)

    return torch.mean(torch.clamp(1 + positive_dist - negative_dist, min=0.0))


def triplet_cosine_acc(output):
    """Compute triplet accuracy.

    Args:
        output: Output of the model

    Returns:
        acc: Triplet accuracy
    """
    anchor, positive, negative = output

    positive_dist = 1 - nn.CosineSimilarity()(anchor, positive)
    negative_dist = 1 - nn.CosineSimilarity()(anchor, negative)

    return (torch.mean((positive_dist < negative_dist).float()).item())


def triplet_l2_acc(output):
    """Compute triplet accuracy.

    Args:
        output: Output of the model

    Returns:
        acc: Triplet accuracy
    """
    anchor, positive, negative = output

    positive_dist = torch.sum((anchor - positive) ** 2, dim=1)
    negative_dist = torch.sum((anchor - negative) ** 2, dim=1)

    return (torch.mean((positive_dist < negative_dist).float()).item())
