import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # noqa: F401  (kept if you later want PCA)

from model import SeqSetVAEPretrain
from dataset import SeqSetVAEDataModule
import config


# ---------------
# 路径（根据你的环境修改）
# ---------------
CKPT_PATH = \
    "/home/sunx/data/aiiih/projects/sunx/projects/SeqSetVAE/outputs/SeqSetVAE-PT/checkpoints/SeqSetVAE_pretrain_batch4.ckpt"
DATA_DIR = \
    "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
PARAMS_MAP = \
    "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"
LABELS_PATH = \
    "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"

# 输出图像（正/负样本 × mean/var）
OUT_POS_MEAN = "heatmap_pos_mean.png"
OUT_POS_VAR = "heatmap_pos_var.png"
OUT_NEG_MEAN = "heatmap_neg_mean.png"
OUT_NEG_VAR = "heatmap_neg_var.png"


def build_model(device: torch.device) -> SeqSetVAEPretrain:
    model = (
        SeqSetVAEPretrain(
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
        )
        .to(device)
        .eval()
    )

    state = torch.load(CKPT_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded weights: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


def build_dataloader():
    dm = SeqSetVAEDataModule(
        saved_dir=DATA_DIR,
        params_map_path=PARAMS_MAP,
        label_path=LABELS_PATH,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()
    return dm.test_dataloader()


def encode_sets_batch(set_encoder, sets, device: torch.device):
    """将一个病人的所有 set padding 到同长，一次性编码，得到 [S,D] 的 mu/logvar。

    返回：mu_b, logvar_b 形状均为 [S, D]；若该病人无有效 set，返回 (None, None)。
    """
    if len(sets) == 0:
        return None, None

    max_len = max(s["var"].shape[1] for s in sets)
    S = len(sets)
    D = sets[0]["var"].shape[-1]
    var_pad = torch.zeros(S, max_len, D, device=device)
    val_pad = torch.zeros(S, max_len, 1, device=device)
    for i, s in enumerate(sets):
        n = s["var"].shape[1]
        var_pad[i, :n] = s["var"]
        val_pad[i, :n] = s["val"]

    if hasattr(set_encoder, "encode_from_var_val"):
        z_list, _ = set_encoder.encode_from_var_val(var_pad, val_pad)
    else:
        _, z_list, _ = set_encoder(var_pad, val_pad)
    _, mu, logvar = z_list[-1]  # [S,1,D]
    return mu.squeeze(1), logvar.squeeze(1)  # [S,D]


def collect_one_pos_one_neg_matrices(model: SeqSetVAEPretrain, dl, device: torch.device):
    """
    选择一个正样本和一个负样本；对每个样本提取每个 set 的后验（最后一层）mu 与 var，
    返回形如：
        pos = {"label":1, "mu": np.ndarray[D,S], "var": np.ndarray[D,S]}
        neg = {"label":0, "mu": np.ndarray[D,S], "var": np.ndarray[D,S]}
    其中 D=config.latent_dim，S=该病人的 set 数；矩阵按 [维度×set] 排列，方便画热图。
    """
    pos_rec, neg_rec = None, None
    amp = torch.cuda.is_available()

    with torch.inference_mode(), torch.cuda.amp.autocast(
        enabled=amp, dtype=(torch.float16 if amp else torch.float32)
    ):
        for batch in dl:
            # 移动到设备
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # 拆分为该 batch（通常为 1 个病人）的 set 列表
            pats = model._split_sets(
                batch["var"], batch["val"], batch["minute"], batch["set_id"], batch.get("padding_mask")
            )

            for i, sets in enumerate(pats):
                label = int(batch["label"][i].item())
                mu_b, logvar_b = encode_sets_batch(model.set_encoder, sets, device)
                if mu_b is None:
                    continue
                var_b = torch.exp(logvar_b)  # [S, D]

                # 转置为 [D, S] 便于以 y=维度, x=set 作图
                mu_mat = mu_b.float().T.cpu().numpy()
                var_mat = var_b.float().T.cpu().numpy()

                rec = {"label": label, "mu": mu_mat, "var": var_mat}
                if label == 1 and pos_rec is None:
                    pos_rec = rec
                if label == 0 and neg_rec is None:
                    neg_rec = rec
                if pos_rec is not None and neg_rec is not None:
                    return pos_rec, neg_rec

    return pos_rec, neg_rec


def plot_heatmap(matrix: np.ndarray, title: str, out_file: str, value_type: str = "mean"):
    """
    画单个样本的热图：matrix 形状为 [D, S]，y=维度(0..D-1)，x=set index(0..S-1)。
    value_type: "mean" 使用对称色轴；"var" 使用非负色轴并做分位裁剪。
    """
    plt.figure(figsize=(10, 6))

    # 色轴范围
    if value_type == "mean":
        vmax = float(np.nanmax(np.abs(matrix))) if matrix.size > 0 else 1.0
        vmax = vmax if vmax > 1e-6 else 1.0
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        # 方差：0..p95，避免极端值拉伸
        if matrix.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = 0.0
            vmax = float(np.percentile(matrix, 95))
            vmax = vmax if vmax > 1e-6 else float(np.nanmax(matrix) + 1e-6)
        cmap = "viridis"

    im = plt.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xlabel("set index")
    plt.ylabel("latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=240)
    print(f"Saved {out_file}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    dl = build_dataloader()
    pos_rec, neg_rec = collect_one_pos_one_neg_matrices(model, dl, device)

    if pos_rec is None and neg_rec is None:
        print("No positive or negative sample found; nothing to plot.")
        return

    if pos_rec is not None:
        plot_heatmap(pos_rec["mu"], title="Positive sample - mean(μ)", out_file=OUT_POS_MEAN, value_type="mean")
        plot_heatmap(pos_rec["var"], title="Positive sample - var(σ²)", out_file=OUT_POS_VAR, value_type="var")
    else:
        print("No positive sample found.")

    if neg_rec is not None:
        plot_heatmap(neg_rec["mu"], title="Negative sample - mean(μ)", out_file=OUT_NEG_MEAN, value_type="mean")
        plot_heatmap(neg_rec["var"], title="Negative sample - var(σ²)", out_file=OUT_NEG_VAR, value_type="var")
    else:
        print("No negative sample found.")


if __name__ == "__main__":
    main()

