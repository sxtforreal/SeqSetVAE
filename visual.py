import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
import config

# Initialize model
model = SeqSetVAE(
    input_dim=config.input_dim,
    reduced_dim=config.reduced_dim,
    latent_dim=config.latent_dim,
    levels=config.levels,
    heads=config.heads,
    m=config.m,
    beta=config.beta,
    lr=config.lr,
    num_classes=config.num_classes,
    ff_dim=config.ff_dim,
    transformer_heads=config.transformer_heads,
    transformer_layers=config.transformer_layers,
    pretrained_ckpt=config.pretrained_ckpt,
    w=config.w,
    free_bits=config.free_bits,
)
pretrained_ckpt = "/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints/best_SeqSetVAE-v2.ckpt"
ckpt = torch.load(pretrained_ckpt, map_location=torch.device("cpu"))
state_dict = ckpt.get("state_dict", ckpt)
setvae_state = {
    k.replace("setvae.", ""): v
    for k, v in state_dict.items()
    if k.startswith("setvae.")
}
model.load_state_dict(setvae_state, strict=False)
model.eval()

# Data
saved_dir = "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
params_map_path = "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"
label_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
data_module = SeqSetVAEDataModule(saved_dir, params_map_path, label_path)
data_module.setup()
test_loader = data_module.test_dataloader()


sets_list = []

for batch in test_loader:
    var = batch["var"]
    val = batch["val"]
    time = batch["minute"]
    set_ids = batch["set_id"]
    sets = model._split_sets(var, val, time, set_ids)
    sets_list.append(sets)
    if len(sets_list) >= 1:
        break

# Compute components for visualization across all collected sets (sequences)
all_z_prims = []
all_h_seq = []
all_input_xs = []
all_recon_with = []

for sets in sets_list:
    S = len(sets)
    z_prims = []
    pos_list = []
    input_xs = []

    for s_dict in sets:
        var, val, time = s_dict["var"], s_dict["val"], s_dict["minute"]
        minute_val = time.unique()
        pos_list.append(minute_val)

        # Compute z_prim using SetVAE (without time/history)
        recon_without, z_list, _ = model.setvae(
            var, val
        )  # Optional: recon_without if needed
        z_sample, mu, logvar = z_list[-1]
        z_prims.append(z_sample.squeeze(1))

        # Compute input embedding (reduced scaled by value)
        if model.setvae.setvae.dim_reducer is not None:
            reduced = model.setvae.setvae.dim_reducer(var)
        else:
            reduced = var
        norms = torch.norm(reduced, p=2, dim=-1, keepdim=True)
        reduced_normalized = reduced / (norms + 1e-8)
        target_x = reduced_normalized * val
        input_xs.append(target_x)

    # Stack to get sequences
    z_seq = torch.stack(z_prims, dim=1)  # [B, S, latent]
    pos_tensor = torch.stack(pos_list, dim=1)  # [B, S]

    # Inject time information via RoPE
    z_seq_rope = model._apply_rope(z_seq, pos_tensor)

    # Apply transformer to get h_seq (with history and time)
    h_seq = model.transformer(z_seq_rope, mask=model._causal_mask(S, z_seq.device))

    # Compute reconstructed embeddings with time/history injection
    recon_with = []
    for idx, s_dict in enumerate(sets):
        N_t = s_dict["var"].size(1)
        recon = model.decoder(h_seq[:, idx], N_t)
        recon_with.append(recon)

    # Collect for all sequences
    all_z_prims.append(z_seq)
    all_h_seq.append(h_seq)
    all_input_xs.extend(input_xs)
    all_recon_with.extend(recon_with)

# Concatenate all points/sequences (assuming B=1 per sets)
z_all = (
    torch.cat(all_z_prims, dim=1).squeeze(0).detach().cpu().numpy()
)  # [total_S, latent]
h_all = (
    torch.cat(all_h_seq, dim=1).squeeze(0).detach().cpu().numpy()
)  # [total_S, latent]
input_all = (
    torch.cat(all_input_xs, dim=1).squeeze(0).detach().cpu().numpy()
)  # [total_points, dim]
recon_all = (
    torch.cat(all_recon_with, dim=1).squeeze(0).detach().cpu().numpy()
)  # [total_points, dim]

print(f"input shape: {input_all.shape}")
print(f"recon shape: {recon_all.shape}")
print(f"z shape: {z_all.shape}")
print(f"h shape: {h_all.shape}")

# Flatten for UMAP (already flat)
input_flat = input_all
recon_flat = recon_all
z_flat = z_all
h_flat = h_all

# UMAP for input and recon (same dim, fit together)
all_token_points = np.concatenate([input_flat, recon_flat], axis=0)
umap_token = umap.UMAP(n_components=2, random_state=1)
umap_token_emb = umap_token.fit_transform(all_token_points)

# Split UMAP results
n_points = input_flat.shape[0]
umap_input = umap_token_emb[:n_points]
umap_recon = umap_token_emb[n_points:]

# UMAP for z and h (same dim, fit together)
all_latent_points = np.concatenate([z_flat, h_flat], axis=0)
umap_latent = umap.UMAP(n_components=2, random_state=1)
umap_latent_emb = umap_latent.fit_transform(all_latent_points)

# Split UMAP results
n_latent = z_flat.shape[0]
umap_z = umap_latent_emb[:n_latent]
umap_h = umap_latent_emb[n_latent:]

# Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Left: Input embeddings vs Reconstructed (with time/history)
axs[0].scatter(
    umap_input[:, 0],
    umap_input[:, 1],
    color="black",
    label="Input Embeddings (reduced * val)",
    alpha=0.7,
)
axs[0].scatter(
    umap_recon[:, 0],
    umap_recon[:, 1],
    color="blue",
    label="Recon Embeddings (with time/history)",
    alpha=0.7,
)
axs[0].set_title("UMAP of Input and Reconstructed Embeddings")
axs[0].legend()

# Right: Z (before injection) vs H (after injection)
axs[1].scatter(
    umap_z[:, 0],
    umap_z[:, 1],
    color="red",
    label="Z (before time/history)",
    alpha=0.7,
)
axs[1].scatter(
    umap_h[:, 0],
    umap_h[:, 1],
    color="green",
    label="H (after time/history)",
    alpha=0.7,
)
axs[1].set_title("UMAP of Z and H (before/after time info)")
axs[1].legend()

# Optional: Arrows to show transition from Z to H for each point
for i in range(len(umap_z)):
    axs[1].arrow(
        umap_z[i, 0],
        umap_z[i, 1],
        umap_h[i, 0] - umap_z[i, 0],
        umap_h[i, 1] - umap_z[i, 1],
        head_width=0.05,
        color="gray",
    )

plt.tight_layout()
plt.show()
