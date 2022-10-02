"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
# from cortex_DIM.functions.gan_losses import get_positive_expectation, get_negative_expectation
import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [b++++sz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def local_global_loss_(l_enc, g_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    graph_node_num = l_enc.shape[1]
    batch_size = 16
    l_enc = l_enc.reshape(-1, 256)
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    for i in range(int(num_graphs/3)):
        pos_mask[i*3*graph_node_num:(i+1)*3*graph_node_num, [i, i+batch_size, i+2*batch_size]] = 1.
        neg_mask[i*3*graph_node_num:(i+1)*3*graph_node_num, [i, i+batch_size, i+2*batch_size]] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / (num_nodes * 3)
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 3))

    return E_neg - E_pos


class MultiLossLayer(nn.Module):
    def __init__(self):
        super(MultiLossLayer, self).__init__()
        def get_rand():
            return torch.tensor(0.8 * np.random.random() + 0.2)
        self.threshold = torch.tensor(0.05).cuda()
        self.loss = SupConLoss(temperature=0.1)
        self._sigmas_sq = nn.ParameterList([nn.Parameter(get_rand())] * 3)

    def forward(self, y1, y2):
        # intra loss
        ys = [y2]


        a = self.loss(y1)
        if self._sigmas_sq[0] < self.threshold:
            loss = a / (self._sigmas_sq[0] * 2) + torch.log(self._sigmas_sq[0])
        else:
            loss = a / (self.threshold * 2) 
        for i, y in enumerate(ys):
            b = self.loss(y)
            if self._sigmas_sq[i] < self.threshold:
                loss += b / (self._sigmas_sq[i] * 2) + torch.log(self._sigmas_sq[i])
            else:
                loss += b / (self.threshold * 2)
        return loss, a, b
