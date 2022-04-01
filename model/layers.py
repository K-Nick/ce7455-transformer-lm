import sys
sys.path.insert(0, "..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.block import MultiHeadAttention
import numpy as np
from util.nn_utils import clones


# class MultiHeadAttention(nn.Module):
# """
# einsum version implementation
# """
#     def __init__(self, d_model, nhead) -> None:
#         super().__init__()
#         self.q_proj_mats = nn.Parameter(torch.Tensor(nhead, d_model, d_model // nhead),
#                                         requires_grad=True)  # [num_head, q_dim, d_model]
#         self.v_proj_mats = nn.Parameter(torch.Tensor(nhead, d_model, d_model // nhead),
#                                         requires_grad=True)
#         self.k_proj_mats = nn.Parameter(torch.Tensor(nhead, d_model, d_model // nhead),
#                                         requires_grad=True)
#         self.dk = d_model // nhead
#         self.d_model = d_model
#         self.attn_score = None

#         nn.init.xavier_uniform_(self.q_proj_mats)
#         nn.init.xavier_uniform_(self.k_proj_mats)
#         nn.init.xavier_uniform_(self.v_proj_mats)

#     def forward(self, q, v, k, attn_mask):
#         bsz, seq_len, _ = q.shape

#         # q_proj = torch.matmul(q.unsqueeze(1), self.q_proj_mats)
#         # v_proj = torch.matmul(v.unsqueeze(1), self.v_proj_mats)
#         # k_proj = torch.matmul(k.unsqueeze(1), self.k_proj_mats)

#         q_proj = torch.einsum("ijk,hkd->ihjd", q, self.q_proj_mats)
#         # [batch, nhead, seq_len, d_model // nhead]
#         v_proj = torch.einsum("ijk,hkd->ihjd", v, self.v_proj_mats)
#         k_proj = torch.einsum("ijk,hkd->ihjd", k, self.k_proj_mats)

#         ##### self attention for each head #####
#         attn_logit = torch.einsum(
#             "ihqd,ihkd->ihqk", q_proj, k_proj) / np.sqrt(self.dk)  # [batch, nhead, seq_len, seq_len]

#         ##### apply mask #####
#         neg_inf_mat = torch.zeros(bsz, seq_len, seq_len).type_as(q).masked_fill(
#             mask=attn_mask, value=-np.inf)
#         # [batch, 1, seq_len, seq_len] for breadcasting
#         neg_inf_mat = neg_inf_mat.unsqueeze(1)
#         attn_logit += neg_inf_mat
#         attn_score = F.softmax(attn_logit, dim=3)

#         self.attn_score = attn_score.detach()  # attention score forward hook

#         weighted_v = torch.einsum("ihkd,ihqk->iqhd", v_proj, attn_score)
#         out_v = weighted_v.reshape((bsz, seq_len, -1))

#         return out_v  # [batch, seq_len, d_model]

class MultiHeadAttention(nn.Module):
    """
    bmm version implementation
    """

    def __init__(self, d_model, nhead, self_attn=False, scaled=True) -> None:
        super().__init__()
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
        # self.q_proj_mats = nn.Parameter(torch.Tensor(nhead, d_model, d_model // nhead),
        #                                 requires_grad=True)  # [num_head, q_dim, d_model]
        # self.v_proj_mats = nn.Parameter(torch.Tensor(nhead, d_model, d_model // nhead),
        #                                 requires_grad=True)
        # self.k_proj_mats = nn.Parameter(torch.Tensor(nhead, d_model, d_model // nhead),
        #                                 requires_grad=True)

        # * packed projection trick*
        self.dk = d_model // nhead
        self.d_model = d_model
        self.nhead = nhead
        self.attn_score = None
        self.self_attn = self_attn
        self.scaled = scaled

        if not self_attn:
            self.q_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
        else:
            self.qkv_linear = nn.Linear(d_model, 3 * d_model)

        # nn.init.xavier_uniform_(self.q_proj_mats)
        # nn.init.xavier_uniform_(self.k_proj_mats)
        # nn.init.xavier_uniform_(self.v_proj_mats)

    def forward(self, q, v, k, attn_mask):
        bsz, seq_len, _ = q.shape
        d_model = self.d_model
        nhead = self.nhead
        dk = self.dk

        def _flat(x):
            # (B,N,D) -> (B*H,N,D/H)
            x_proj = x.reshape((bsz, seq_len, nhead, dk)
                               ).transpose(1, 2).reshape((bsz * nhead, seq_len, dk))
            return x_proj

        if not self.self_attn:
            q_proj = _flat(self.q_linear(q))
            v_proj = _flat(self.v_linear(v))
            k_proj = _flat(self.k_linear(k))  # (B*H, N, D/H)
        else:
            qkv_proj = self.qkv_linear(q)  # (B, N, 3 * D)
            q_proj, k_proj, v_proj = torch.split(qkv_proj, d_model, 2)
            q_proj = _flat(q_proj)
            k_proj = _flat(k_proj)
            v_proj = _flat(v_proj)

        ##### self attention for each head #####
        attn_logit = torch.bmm(q_proj, k_proj.transpose(
            1, 2)).reshape(bsz, nhead, seq_len, seq_len)  # (B, H, N, N)
        # attn_logit = torch.einsum(
        #     "bnk,bmk->bnm", q_proj, k_proj).reshape(bsz, nhead, seq_len, seq_len)
        if self.scaled:
            attn_logit /= np.sqrt(dk)

        ##### apply mask #####
        neg_inf_mat = torch.zeros(bsz, seq_len, seq_len).type_as(q).masked_fill(
            mask=attn_mask, value=-np.inf)
        # [batch, 1, seq_len, seq_len] for breadcasting
        neg_inf_mat = neg_inf_mat.unsqueeze(1)
        attn_logit += neg_inf_mat
        attn_score = F.softmax(attn_logit, dim=3)  # (B,H,N,N)

        self.attn_score = attn_score.detach()  # attention score forward hook

        # weighted_v = torch.bmm(attn_score.reshape(-1, seq_len, seq_len),
        #                        v_proj).reshape(bsz, nhead, seq_len, dk).transpose(1, 2).reshape(bsz, seq_len, -1)
        weighted_v = torch.einsum(
            "ihkd,ihqk->iqhd", v_proj.reshape(bsz, nhead, seq_len, dk), attn_score).reshape(bsz, seq_len, -1)

        return weighted_v  # [batch, seq_len, d_model]


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
    def __init__(self, dim, dropout=0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, layer):
        """
        use layer in forward because layer sometimes may have other input argument like mask, we can use lambda to wrap it outside
        """
        return self.norm(x + self.dropout(layer(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout) -> None:
        super().__init__()
        self.sublayers = clones(AddNormWrapper(d_model, dropout), 2)
        self.mhattn = MultiHeadAttention(d_model, nhead)
        self.ff = FeedForward(d_ff, d_model, dropout)

    def forward(self, x, attn_mask=None):
        x = self.sublayers[0](x, lambda x:
                              self.mhattn(x, x, x, attn_mask))
        # ! bug makes model converge???
        x = self.sublayers[1](x, self.ff)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, nlayer) -> None:
        super().__init__()
        self.layers = clones(encoder_layer, nlayer)

    def forward(self, src, mask=None):
        max_len = src.shape[1]  # [batch, seq_len, dim]

        x = src

        for layer in self.layers:
            x = layer(x, mask)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000) -> None:
        super().__init__()
        pe_i = torch.arange(d_model // 2).reshape((1, d_model // 2, 1))
        pe_pos = torch.arange(max_len).reshape((max_len, 1, 1))
        pe_sin = torch.sin(
            pe_pos / torch.exp(pe_i * 2 / d_model * -np.log(10000)))
        pe_cos = torch.cos(
            pe_pos / torch.exp((pe_i * 2 + 1) / d_model * -np.log(10000)))

        pe = torch.cat([pe_sin, pe_cos], dim=2).reshape((max_len, d_model))

        self.register_buffer("pe", pe)

    def forward(self, src):
        seq_len = src.shape[1]
        out = src + self.pe[:seq_len, :].unsqueeze(0)
        return out


# if __name__ == "__main__":
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     pe = PositionalEncoding(512)
#     x = torch.zeros(128, 256, 512)
#     # out = pe(x).cpu().numpy()
#     sns.heatmap(pe.pe.numpy())
#     plt.show()
#     plt.savefig("./tmp.png")
#     import ipdb
#     ipdb.set_trace()
