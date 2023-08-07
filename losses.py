""" all span_hidden should be already normalized. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import util


def get_contrastive_loss(span_hidden, span_clusters, temp=1, reduction=True):
    num_views, num_spans, hidden_size = span_hidden.size()[:3]
    device = span_hidden.device

    span_hidden = span_hidden.view(-1, hidden_size)
    sim = torch.matmul(span_hidden, span_hidden.t()) / temp
    sim_wo_self = sim + torch.log(1 - torch.eye(sim.size()[0], dtype=torch.float, device=device))
    denominator = torch.logsumexp(sim_wo_self, dim=-1, keepdim=True)
    nll = denominator - sim

    # Get labels
    span_clusters = span_clusters.unsqueeze(-1)
    labels = (span_clusters == span_clusters.t()).to(torch.float)
    labels = labels.repeat(num_views, num_views)
    labels.fill_diagonal_(0)

    positive_pair_exist = (labels.sum(dim=-1) > 0)
    nll = nll[positive_pair_exist]
    labels = labels[positive_pair_exist]

    # Get loss: nll for positive
    loss = (nll * labels).sum(dim=-1) / labels.sum(dim=-1)
    if reduction:
        loss = loss.mean()
    return loss
