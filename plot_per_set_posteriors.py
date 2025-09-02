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

# 输出图像
OUT_FIG = "per_set_posteriors_5pos5neg.png"


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


def collect_patients_points(model: SeqSetVAEPretrain, dl, device: torch.device,
                            num_pos: int = 5, num_neg: int = 5):
    """从测试集中收集 5 个正样本和 5 个负样本。
    对每个被选中的病人，提取其每个 set 的后验分布均值（mu）的“逐维平均值”。

    返回一个列表，每个元素为 dict：
        {"label": 0/1, "x": np.ndarray[S], "y": np.ndarray[S]}
    其中 x 为 set 的索引（0..S-1），y 为 mean(mu)（按特征维求平均）。
    """
    pos_cnt, neg_cnt = 0, 0
    selected = []
    amp = torch.cuda.is_available()

    with torch.inference_mode(), torch.cuda.amp.autocast(
        enabled=amp, dtype=(torch.float16 if amp else torch.float32)
    ):
        for bidx, batch in enumerate(dl):
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            pats = model._split_sets(
                batch["var"],
                batch["val"],
                batch["minute"],
                batch["set_id"],
                batch.get("padding_mask"),
            )

            for i, sets in enumerate(pats):
                label = int(batch["label"][i].item())
                need_pos = (label == 1 and pos_cnt < num_pos)
                need_neg = (label == 0 and neg_cnt < num_neg)
                if not (need_pos or need_neg):
                    continue

                mu_b, logvar_b = encode_sets_batch(model.set_encoder, sets, device)
                if mu_b is None:
                    continue

                # 逐 set 的均值（对特征维做平均）并以 set 索引为横坐标
                S = mu_b.shape[0]
                x_vals = np.arange(S)
                y_vals = mu_b.mean(dim=-1).float().cpu().numpy()  # [S]

                selected.append({
                    "label": label,
                    "x": x_vals,
                    "y": y_vals,
                })

                if label == 1:
                    pos_cnt += 1
                else:
                    neg_cnt += 1

                if pos_cnt >= num_pos and neg_cnt >= num_neg:
                    break

            if pos_cnt >= num_pos and neg_cnt >= num_neg:
                break

    print(f"Selected patients: pos={pos_cnt}, neg={neg_cnt}")
    if pos_cnt < num_pos or neg_cnt < num_neg:
        print("Warning: dataset did not contain enough positives/negatives to meet the request.")
    return selected


def plot_patients_sets(selected, out_file: str = OUT_FIG):
    plt.figure(figsize=(8, 7))
    legend_pos_added = False
    legend_neg_added = False

    for rec in selected:
        color = "tab:orange" if rec["label"] == 1 else "tab:blue"
        label_name = "pos" if rec["label"] == 1 else "neg"
        show_label = (rec["label"] == 1 and not legend_pos_added) or \
                     (rec["label"] == 0 and not legend_neg_added)

        plt.plot(
            rec["x"], rec["y"],
            "-o",
            color=color,
            linewidth=1.2,
            markersize=4,
            alpha=0.9,
            label=(label_name if show_label else None),
        )

        if rec["label"] == 1:
            legend_pos_added = True
        else:
            legend_neg_added = True

    plt.xlabel("set idx")
    plt.ylabel("mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=240)
    print(f"Saved {out_file}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    dl = build_dataloader()
    selected = collect_patients_points(model, dl, device, num_pos=5, num_neg=5)
    if len(selected) == 0:
        print("No patients selected; nothing to plot.")
        return
    plot_patients_sets(selected, OUT_FIG)


if __name__ == "__main__":
    main()

