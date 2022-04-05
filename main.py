import sys
sys.path.insert(0, "..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import argparse
from config import get_config
from data import build_loader, WikiLMDatset
from model import build_transformer_lm
from util.nn_utils import load_checkpoint, save_checkpoint, set_env_seed
from tqdm import tqdm
import wandb
import os
import time
import datetime


def read_args():
    args = argparse.ArgumentParser()
    args.add_argument("--cfg", type=str, required=True)
    args.add_argument("--batch-size", type=int)
    args.add_argument("--win_size", type=int)
    args.add_argument("--lr", type=float)
    args.add_argument("--data-dir", type=str)
    args.add_argument("--eval", action="store_true")
    args.add_argument("--use-ckpt", action="store_true")
    args.add_argument("--resume", type=str)
    args.add_argument("--dryrun", action="store_true")
    args.add_argument("--output", type=str)
    args.add_argument("--seed", type=int)
    args.add_argument("--source", type=str,
                      help="source of transformer encoder")

    return args.parse_args()


def evaluate(model, data_loader):
    model.eval()
    with torch.no_grad():
        num_batch = 0
        sum_ce_loss = 0.0

        for inp_text, target in data_loader:
            inp_text = inp_text.cuda()
            target = target.cuda()
            num_batch += 1
            logit = model(inp_text)
            logit = logit.transpose(2, 1)
            ce_loss = F.cross_entropy(logit, target)
            sum_ce_loss += ce_loss.item()
        mean_ce_loss = sum_ce_loss / num_batch
        pplx = np.exp(mean_ce_loss)
    return pplx


def main(conf):
    set_env_seed(conf.SEED)
    wandb_mode = "disabled" if conf.DEBUG_FLAG else None

    train_set, val_set, test_set, train_loader, val_loader, test_loader, cache_dict = build_loader(
        conf)
    pre_emb = cache_dict["pre_emb"]
    vocab = cache_dict["vocab"]
    model = build_transformer_lm(conf, pre_emb)
    model = model.cuda()
    optimizer = AdamW(model.parameters(), conf.TRAIN.LR)
    # scheduler = CosineAnnealingLR(optimizer,
    #                               T_max=1,
    #                               eta_min=1e-9)

    if not conf.OUTPUT:
        conf.defrost()
        conf.OUTPUT = f"./exps/{conf.MODEL.NAME}"
        conf.freeze()

    if conf.EVAL_MODE:
        val_pplx = evaluate(model, val_loader)
        test_pplx = evaluate(model, test_loader)
        print(f"val/pplx: {val_pplx:.4f} | test/pplx: {test_pplx:.4f}")
        return

    wandb.init(project="ce7455-transformer-lm",
               config=conf, name=conf.MODEL.NAME,
               mode=wandb_mode)
    wandb.define_metric("train/ce_loss", summary="min")
    wandb.define_metric("val/pplx", summary="min")
    wandb.define_metric("test/pplx", summary="min")
    wandb.run.log_code(".")

    os.makedirs(conf.OUTPUT, exist_ok=True)
    # sys.stdout = open(os.path.join(conf.OUTPUT, "stdout.log"), "w")
    start_epoch = conf.TRAIN.START_EPOCH
    num_epoch = conf.TRAIN.NUM_EPOCH
    best_val_pplx = np.inf
    best_test_pplx = np.inf
    bsz = conf.DATA.BATCH_SIZE
    num_train_batch = int(np.ceil(len(train_set) / bsz))

    for epoch in range(start_epoch, num_epoch):
        model.train()
        train_loss = 0.0
        st = time.time()

        for inp_text, target in tqdm(train_loader, desc=f"train epoch#{epoch:4d}"):
            inp_text = inp_text.cuda()
            target = target.cuda()

            logit = model(inp_text)
            logit = logit.transpose(2, 1)
            loss = F.cross_entropy(logit, target, ignore_index=vocab["<unk>"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            train_loss += loss.item()

        train_loss /= num_train_batch
        time_elapsed = time.time() - st
        print(
            f"epoch #{epoch:3d} | train/ce_loss: {train_loss:.4f} | time elapsed: {time_elapsed:.3f}")
        wandb.log({"train/ce_loss": train_loss,
                  "train/runtime": time_elapsed}, step=epoch)

        val_interval = conf.TRAIN.VAL_INTERVAL
        if epoch % val_interval == 0:
            val_pplx = evaluate(model, val_loader)
            if val_pplx < best_val_pplx:
                save_checkpoint(conf, model, optimizer,
                                None, epoch, is_best=True)
                best_val_pplx = val_pplx
            print(
                f"epoch #{epoch:3d} | val/pplx: {val_pplx:.4f} | val/best_pplx: {best_val_pplx:.4f}")
            wandb.log({"val/pplx": val_pplx}, step=epoch)

            test_pplx = evaluate(model, test_loader)
            if test_pplx < best_test_pplx:
                best_test_pplx = test_pplx
            wandb.log({"test/pplx": test_pplx}, step=epoch)
            print(
                f"epoch #{epoch:3d} | test/pplx: {test_pplx:.4f} | test/best_pplx: {best_test_pplx:.4f}")


if __name__ == "__main__":
    args = read_args()
    conf = get_config(args)
    main(conf)
