#!/usr/bin/env python3
"""
使用 SeqSetVAEPretrain + PoE（精度域加权）对 test 全量样本做可视化：
- 以每位病人的所有 set 的高斯后验 N(μ_i, σ_i^2) 做 PoE 得到唯一分布 N(μ_poe, σ_poe^2)
- 输出两类图：
  1) 基础二维散点：x=mean(|μ_poe|)，y=mean(σ_poe^2)
  2) 特征降维：把 [μ_poe, log σ_poe^2] 拼接做 PCA/UMAP 到 2D

加速：一次性编码一位病人的所有 set（padding 到同长），torch.inference_mode + AMP。
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

from model import SeqSetVAEPretrain
from dataset import SeqSetVAEDataModule
import config


def load_pretrain_model(ckpt_path: str, device: torch.device) -> SeqSetVAEPretrain:
    model = SeqSetVAEPretrain(
        input_dim=config.input_dim,
        reduced_dim=config.reduced_dim,
        latent_dim=config.latent_dim,
        levels=config.levels,
        heads=config.heads,
        m=config.m,
        beta=config.beta,
        lr=config.lr,
        ff_dim=config.ff_dim,
        transformer_heads=config.transformer_heads,
        transformer_layers=config.transformer_layers,
        warmup_beta=config.warmup_beta,
        max_beta=config.max_beta,
        beta_warmup_steps=config.beta_warmup_steps,
        free_bits=config.free_bits,
        transformer_dropout=config.transformer_dropout,
    ).to(device)
    model.eval()

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def encode_sets_batch(set_encoder: torch.nn.Module, sets: list, device: torch.device):
    """把一位病人的所有 set padding 到同长，一次性编码得到 [B,D] 的 mu/logvar"""
    if len(sets) == 0:
        return None, None
    max_len = max(s["var"].shape[1] for s in sets)
    B = len(sets)
    D = sets[0]["var"].shape[-1]
    var_pad = torch.zeros(B, max_len, D, device=device)
    val_pad = torch.zeros(B, max_len, 1, device=device)
    for i, s in enumerate(sets):
        n = s["var"].shape[1]
        var_pad[i, :n] = s["var"]
        val_pad[i, :n] = s["val"]

    if hasattr(set_encoder, "encode_from_var_val"):
        z_list, _ = set_encoder.encode_from_var_val(var_pad, val_pad)
    else:
        _, z_list, _ = set_encoder(var_pad, val_pad)
    _, mu, logvar = z_list[-1]  # [B,1,D]
    return mu.squeeze(1), logvar.squeeze(1)  # [B,D]


def poe_precision_weighted(mu_b: torch.Tensor,
                           logvar_b: torch.Tensor,
                           temp: float = 1.0,
                           clip_min: float = 1e-6,
                           clip_max: float = 1e6,
                           weight_mode: str = "precision_softmax",
                           prior_var: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """精度域加权 PoE：
    输入：mu_b/logvar_b 形状 [S,D]（该病人有 S 个 set）
    返回：mu_poe/var_poe 形状 [D]
    """
    eps = 1e-8
    var = torch.exp(logvar_b).clamp_min(eps)     # [S,D]
    prec = (1.0 / var).clamp(min=clip_min, max=clip_max)  # [S,D]

    if weight_mode == "precision_softmax":
        conf = prec.mean(dim=-1)  # [S]
        T = max(1e-3, float(temp))
        w = torch.softmax(conf / T, dim=0)  # [S]
    else:
        w = torch.full((mu_b.size(0),), 1.0 / max(1, mu_b.size(0)), device=mu_b.device)

    # 精度加权求和 + 先验
    sum_prec = (w[:, None] * prec).sum(dim=0) + 1.0 / max(prior_var, eps)  # [D]
    var_poe = 1.0 / sum_prec
    mu_poe = var_poe * (w[:, None] * prec * mu_b).sum(dim=0)
    return mu_poe, var_poe


def main():
    parser = argparse.ArgumentParser(description="SeqSetVAEPretrain + PoE 可视化（test 全量）")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--params_map", type=str, required=True)
    parser.add_argument("--label_file", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./poe_vis")
    parser.add_argument("--weight_mode", type=str, default="precision_softmax", choices=["precision_softmax", "uniform"])
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--clip_min", type=float, default=1e-6)
    parser.add_argument("--clip_max", type=float, default=1e6)
    parser.add_argument("--use_umap", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_pretrain_model(args.checkpoint, device)

    dm = SeqSetVAEDataModule(
        saved_dir=args.data_dir,
        params_map_path=args.params_map,
        label_path=args.label_file,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()
    dl = dm.test_dataloader()

    xs_basic, ys_basic, labels = [], [], []
    feat_list = []  # 保存 [μ_poe, log σ_poe^2]

    set_encoder = model.set_encoder
    amp = torch.cuda.is_available()

    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=amp, dtype=(torch.float16 if amp else torch.float32)):
        for bidx, batch in enumerate(dl):
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            pats = model._split_sets(batch["var"], batch["val"], batch["minute"], batch["set_id"], batch.get("padding_mask"))
            for i, sets in enumerate(pats):
                mu_b, logvar_b = encode_sets_batch(set_encoder, sets)
                if mu_b is None:
                    continue
                mu_poe, var_poe = poe_precision_weighted(
                    mu_b, logvar_b,
                    temp=args.temp,
                    clip_min=args.clip_min,
                    clip_max=args.clip_max,
                    weight_mode=args.weight_mode,
                )
                # 基础 2D：mean(|μ|) vs mean(var)
                xs_basic.append(mu_poe.abs().mean().item())
                ys_basic.append(var_poe.mean().item())
                labels.append(int(batch["label"][i].item()))
                # 用于 PCA/UMAP 的高维特征： [μ_poe, log σ_poe^2]
                feat = torch.cat([mu_poe, torch.log(var_poe + 1e-8)], dim=0).float().cpu().numpy()
                feat_list.append(feat)

            if (bidx + 1) % 200 == 0:
                print(f"processed {bidx+1} patients...")

    xs_basic = np.array(xs_basic)
    ys_basic = np.array(ys_basic)
    labels = np.array(labels)
    feats = np.stack(feat_list, axis=0) if feat_list else np.zeros((0, config.latent_dim * 2), dtype=np.float32)

    # 基础散点
    plt.figure(figsize=(7, 6))
    plt.scatter(xs_basic[labels == 0], ys_basic[labels == 0], s=8, alpha=0.5, color="tab:blue", label="neg")
    plt.scatter(xs_basic[labels == 1], ys_basic[labels == 1], s=10, alpha=0.8, color="tab:orange", label="pos")
    plt.xlabel("mean(|mu_poe|)")
    plt.ylabel("mean(var_poe)")
    plt.legend()
    plt.tight_layout()
    out_basic = os.path.join(args.save_dir, "poe_scatter_basic.png")
    plt.savefig(out_basic, dpi=220)
    print(f"Saved {out_basic}, points={len(xs_basic)}")

    # PCA 投影
    if feats.shape[0] > 1:
        pca = PCA(n_components=2, random_state=0)
        xy = pca.fit_transform(feats)
        plt.figure(figsize=(7, 6))
        plt.scatter(xy[labels == 0, 0], xy[labels == 0, 1], s=8, alpha=0.5, color="tab:blue", label="neg")
        plt.scatter(xy[labels == 1, 0], xy[labels == 1, 1], s=10, alpha=0.8, color="tab:orange", label="pos")
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")
        plt.legend()
        plt.tight_layout()
        out_pca = os.path.join(args.save_dir, "poe_pca.png")
        plt.savefig(out_pca, dpi=220)
        print(f"Saved {out_pca}")

    # 可选 UMAP
    if args.use_umap and feats.shape[0] > 10:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0)
            xy = reducer.fit_transform(feats)
            plt.figure(figsize=(7, 6))
            plt.scatter(xy[labels == 0, 0], xy[labels == 0, 1], s=8, alpha=0.5, color="tab:blue", label="neg")
            plt.scatter(xy[labels == 1, 0], xy[labels == 1, 1], s=10, alpha=0.8, color="tab:orange", label="pos")
            plt.xlabel("UMAP-1")
            plt.ylabel("UMAP-2")
            plt.legend()
            plt.tight_layout()
            out_umap = os.path.join(args.save_dir, "poe_umap.png")
            plt.savefig(out_umap, dpi=220)
            print(f"Saved {out_umap}")
        except Exception as e:
            print(f"UMAP 不可用或安装失败，已跳过：{e}")


if __name__ == "__main__":
    main()

