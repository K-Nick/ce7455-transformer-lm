import sys
sys.path.insert(0, "../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, v_dim, k_dim, d_model, nhead) -> None:
        super().__init__()
        self.q_proj_mats = nn.Parameter(torch.Tensor(nhead, q_dim, d_model // nhead),
                                        requires_grad=True)  # [num_head, q_dim, d_model]
        self.v_proj_mats = nn.Parameter(torch.Tensor(nhead, v_dim, d_model // nhead),
                                        requires_grad=True)
        self.k_proj_mats = nn.Parameter(torch.Tensor(nhead, k_dim, d_model // nhead),
                                        requires_grad=True)
        self.dk = d_model // nhead
        self.d_model = d_model

        nn.init.xavier_uniform(self.q_proj_mats)
        nn.init.xavier_uniform(self.k_proj_mats)
        nn.init.xavier_uniform(self.v_proj_mats)

    def forward(self, q, v, k, attn_mask):
        bsz, seq_len, _ = q.shape
        q_proj = torch.einsum("ijk,hkd->ihjd", q, self.q_proj_mats)
        # [batch, nhead, seq_len, d_model // nhead]
        v_proj = torch.einsum("ijk,hkd->ihjd", v, self.v_proj_mats)
        k_proj = torch.einsum("ijk,hkd->ihjd", k, self.k_proj_mats)

        ##### self attention for each head #####
        attn_logit = torch.einsum(
            "ihqd,ihkd->ihqk", q_proj, k_proj) / torch.sqrt(self.d_model)  # [batch, nhead, seq_len, seq_len]

        ##### apply mask #####
        if attn_mask:
            neg_inf_mat = torch.zeros(bsz, seq_len, seq_len).masked_fill(
                mask=~attn_mask, value=-np.inf)
        # [batch, 1, seq_len, seq_len] for breadcasting
        neg_inf_mat = neg_inf_mat.unsqueeze(1)
        attn_logit += neg_inf_mat
        attn_score = F.softmax(attn_logit, dim=3)

        weighted_v = torch.einsum("ihkd,ihqk->iqhd", v_proj, attn_score)
        out_v = weighted_v.reshape((bsz, seq_len, -1))

        return out_v  # [batch, seq_len, d_model]


class FeedForward(nn.Module):
    def __init__(self, d_ff, d_model, dropout) -> None:
        super().__init__()
        self.ff_net = nn.Sequential(nn.Linear(d_model, d_ff),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.ff_net(x)


class AddNormWrapper(nn.Module):
    def __init__(self, layer, dim, dropout=0.1) -> None:
        super().__init__()
        self.layer = layer
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.dropout(self.layer(x)))
