import sys
sys.path.insert(0, "..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import TransformerEncoder, TransformerEncoderLayer, PositionalEncoding
from data import WikiLMDatset


def build_transformer_lm(conf, pre_emb):
    class TransformerLM(nn.Module):
        def __init__(self, emb, pe, encoder, clf) -> None:
            super().__init__()
            self.emb = emb
            self.adapt_linear = nn.Linear(
                conf.MODEL.WORD_DIM, conf.MODEL.D_MODEL)
            self.pe = pe
            self.encoder = encoder
            self.clf = clf

        def forward(self, s):
            seq_len = s.shape[1]

            emb = self.emb(s)
            emb = self.adapt_linear(emb)
            emb = self.pe(emb)
            attn_mask = torch.triu(torch.ones(
                (seq_len, seq_len))).T.type_as(s).bool()
            emb = self.encoder(emb, mask=~attn_mask)
            # emb = emb[:, :]
            logit = self.clf(emb)

            return logit

    d_model = conf.MODEL.D_MODEL
    nhead = conf.MODEL.NHEAD
    d_ff = conf.MODEL.D_FF
    nlayer = conf.MODEL.NLAYER
    dropout = conf.MODEL.DROPOUT
    vocab_size = pre_emb.shape[0]

    emb = nn.Embedding.from_pretrained(pre_emb)
    pe = PositionalEncoding(d_model)
    if conf.MODEL.SOURCE != "torch.nn":
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout)
        encoder = TransformerEncoder(encoder_layer, nlayer)
    else:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_ff, dropout, batch_first=True, layer_norm_eps=1e-6)
        encoder = nn.TransformerEncoder(encoder_layer, nlayer)

    clf = nn.Sequential(nn.Linear(d_model, d_model // 2),
                        nn.ReLU(),
                        nn.Linear(d_model // 2, vocab_size))

    return TransformerLM(emb, pe, encoder, clf)


if __name__ == "__main__":
    from config import _C
    from util.io import load_pickle
    conf = _C.clone()
    cache_dict = load_pickle("../cache_bin/data_cache.w12.pkl")
    pre_emb = cache_dict["pre_emb"]

    transformer = build_transformer_lm(conf, pre_emb)
    import ipdb
    ipdb.set_trace()  # FIXME
