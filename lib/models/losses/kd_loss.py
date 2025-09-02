import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from functools import partial

from .kl_div import KLDivergence
from .dist_kd import DIST
from .loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer, DeepWalkEmbeddingLoss


class KDLoss():
    '''
    kd loss wrapper.
    '''

    def __init__(self, student, teacher, ori_loss, kd_method='kdt4', student_module='', teacher_module='', ori_loss_weight=1.0, kd_loss_weight=1.0, args=None):
        self.student = student
        self.teacher = teacher
        self.ori_loss = ori_loss
        self.ori_loss_weight = ori_loss_weight
        self.kd_method = kd_method
        self.kd_loss_weight = kd_loss_weight
        

        self._teacher_out = None
        self._student_out = None
        self.args = args
        self.weight_or_logit = [] # if 1 weight, else logit, useful when self.kd_loss is a list

        # init kd loss
        if kd_method == 'kd':
            self.kd_loss = KLDivergence(tau=4)
        elif kd_method == 'dist':
            self.kd_loss = DIST(beta=1, gamma=1, tau=1)
        elif kd_method.startswith('dist_t'):
            tau = float(kd_method[6:])
            self.kd_loss = DIST(beta=1, gamma=1, tau=tau)
        elif kd_method.startswith('kdt'):
            tau = float(kd_method[3:])
            self.kd_loss = KLDivergence(tau)
        else:
            raise RuntimeError(f'KD method {kd_method} not found.')
        

        teacher.eval()

    def __call__(self, x, targets, return_residual=False):

        with torch.no_grad():
            feats_tec, logits_tec = self.teacher(x, is_feat=True)
            if self.args.after_pool:
                pre_fc_feat_tec = feats_tec[-1]
            else:
                pre_fc_feat_tec = feats_tec[-2] # the feature before pool

        # compute ori loss of student
        feats_stu, logits = self.student(x, is_feat=True)
        if self.args.after_pool:
            pre_fc_feat_stu = feats_stu[-1]
        else:
            pre_fc_feat_stu = feats_stu[-2]
        
        ori_loss = self.ori_loss(logits, targets) * self.ori_loss_weight
        
        feat_loss = torch.nn.Parameter(torch.tensor([0.0])).to(x)
        if self.args.feat_distill_weight > 0:
            delta_loss_fn = nn.MSELoss()
            feat_loss = delta_loss_fn(pre_fc_feat_stu, pre_fc_feat_tec.detach()) * self.args.feat_distill_weight
        
        kd_loss = torch.nn.Parameter(torch.tensor([0.0])).to(x)
        if self.kd_loss_weight > 0:
            kd_loss = self.kd_loss(logits, logits_tec.detach()) * self.kd_loss_weight

        final_loss = ori_loss + kd_loss + feat_loss

        if return_residual:
            return final_loss, pre_fc_feat_stu, pre_fc_feat_tec - pre_fc_feat_stu
        else:
            return final_loss
    
    
    def pattern_match(self, info):
        pattern = r'\d+\.\d+|\d+'
        matches = re.findall(pattern, info)
        matches = [float(match) for match in matches]
        return matches

    def linear_w(self, cur_ep, start_ep, max_ep, start_v=0, end_v=1):
        if cur_ep <= start_ep:
            return 0
        else:
            return start_v + (cur_ep-start_ep) *(end_v - start_v)/(max_ep - start_ep)


