
# Compute contrastive loss given network output and match/non-match
# Author : Munch Quentin, 2020

import math
from random import randint
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5, non_match_loss_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.non_match_loss_weight = non_match_loss_weight

    def forward(self, out_A, out_B, match_A, match_B, non_match_A, non_match_B):
        # out_A, out_B : [1, D, H, W] => [1, H*W, D]
        # match_A, match_B, non_match_A, non_match_B : [nb_match,] with (u,v)
        # (u,v) -> image_width * v + u

        # get the number of match/non-match (tensor float)
        nb_match = match_A.size()[0]
        nb_non_match = non_match_A.size()[0]

        # create a tensor with the listed matched descriptors in the estimated descriptors map (net output)
        match_A_des = torch.index_select(out_A, 1, match_A).unsqueeze(0)
        match_B_des = torch.index_select(out_B, 1, match_B).unsqueeze(0)
        # create a tensor with the listed non-matched descriptors in the estimated descriptors map (net output)
        non_match_A_des = torch.index_select(out_A, 1, non_match_A).unsqueeze(0)
        non_match_B_des = torch.index_select(out_B, 1, non_match_B).unsqueeze(0)

        # calculate match loss (L2 distance)
        match_loss = 1.0/nb_match * (match_A_des - match_B_des).pow(2).sum()
        # calculate non-match loss (L2 distance with margin)
        zeros_vec = torch.zeros_like(non_match_A_des)
        pixelwise_non_match_loss = torch.max(zeros_vec, self.margin-((non_match_A_des - non_match_B_des).pow(2)))
        # Hard negative scaling (pixelwise)
        hard_negative_non_match = len(torch.nonzero(pixelwise_non_match_loss))
        # final non_match loss with hard negative scaling
        non_match_loss = self.non_match_loss_weight * 1.0/hard_negative_non_match * pixelwise_non_match_loss.sum()

        # final contrastive loss
        contrastive_loss = match_loss + non_match_loss

        return contrastive_loss, match_loss, non_match_loss
