import sys
sys.path.insert(0, "..")

from torchtext.vocab import build_vocab_from_iterator, GloVe
import torch
from torch.utils.data import DataLoader, Dataset
from util.io import load_pickle, save_pickle
import numpy as np
import os


class WikiLMDatset(Dataset):
    def __init__(self, text_repo, win_size, vocab) -> None:
        super().__init__()
        self.tot_len = len(text_repo)
        text_repo = vocab.lookup_indices(text_repo)
        l = len(text_repo)
        trunc_l = (l - 1) // win_size * win_size + 1
        text_repo = np.array(text_repo[:trunc_l])
        self.text_repo = text_repo
        self.win_size = win_size
        self.num_idx = (trunc_l - 1) // win_size

    def __getitem__(self, idx):
        win_sz = self.win_size
        inp_text = self.text_repo[idx * win_sz: (idx + 1) * win_sz]
        target = self.text_repo[idx * win_sz + 1: (idx + 1) * win_sz + 1]
        return inp_text, target

    def __len__(self):
        return self.num_idx


def build_vocab(text_repo):
    vocab = build_vocab_from_iterator(
        [text_repo], special_first=True, specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def load_pretrained_emb(vocab, conf):
    glove_cache = os.path.join(conf.DATA.CACHE_DIR, ".glove_cache")
    word_dim = 300
    vec = GloVe(cache=glove_cache, unk_init=lambda x: torch.randn(word_dim))
    glove_vec = vec.get_vecs_by_tokens(
        vocab.get_itos(), lower_case_backup=True)

    return glove_vec


def prepare_data(data_dir):
    text_repo = {}
    for scope in ["train", "valid", "test"]:
        file_name = os.path.join(data_dir, f"{scope}.txt")
        text_repo[scope] = []
        with open(file_name, "r") as f:
            for line in f:
                line = line.strip()
                if line == '' or line[0] == '=':
                    continue
                tok_sent = line.strip().split(" ")
                text_repo[scope] += tok_sent

    print("=> data prepared")

    return text_repo["train"], text_repo["valid"], text_repo["test"]


def build_loader(conf):
    win_size = conf.DATA.WIN_SIZE

    expected_cache_file = os.path.join(
        conf.DATA.CACHE_DIR, f"data_cache.w{win_size}.pkl")

    if os.path.exists(expected_cache_file):
        cache_dict = load_pickle(expected_cache_file)
        train_set = cache_dict["train_set"]
        val_set = cache_dict["val_set"]
        test_set = cache_dict["test_set"]
        pre_emb = cache_dict["pre_emb"]
        vocab = cache_dict["vocab"]
    else:
        train_data, val_data, test_data = prepare_data("./data_bin")
        vocab = build_vocab(train_data)
        pre_emb = load_pretrained_emb(vocab, conf)
        train_set = WikiLMDatset(train_data, conf.DATA.WIN_SIZE, vocab)
        val_set = WikiLMDatset(val_data, conf.DATA.WIN_SIZE, vocab)
        test_set = WikiLMDatset(test_data, conf.DATA.WIN_SIZE, vocab)

        cache_dict = {"train_set": train_set,
                      "val_set": val_set,
                      "test_set": test_set,
                      "pre_emb": pre_emb,
                      "vocab": vocab}

        save_pickle(cache_dict, expected_cache_file, silent=False)

    print("=> dataset & vocab loaded")

    train_loader = DataLoader(train_set,
                              batch_size=conf.DATA.BATCH_SIZE,
                              shuffle=True,
                              pin_memory=True,
                              prefetch_factor=3,
                              num_workers=1
                              )
    val_loader = DataLoader(val_set,
                            batch_size=conf.DATA.BATCH_SIZE,
                            shuffle=False,
                            pin_memory=True,
                            prefetch_factor=3,
                            num_workers=1)
    test_loader = DataLoader(test_set,
                             batch_size=conf.DATA.BATCH_SIZE,
                             shuffle=False,
                             pin_memory=True,
                             prefetch_factor=3,
                             num_workers=1)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader, cache_dict
