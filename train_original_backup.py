import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from model import SeqSetVAE
from dataset import SeqSetVAEDataModule
import config

if __name__ == "__main__":
    seed_everything(0, workers=True)

    saved_dir = "/home/sunx/data/aiiih/data/mimic/processed/patient_ehr"
    params_map_path = "/home/sunx/data/aiiih/data/mimic/processed/stats.csv"
    label_path = "/home/sunx/data/aiiih/data/mimic/processed/oc.csv"
    data_module = SeqSetVAEDataModule(saved_dir, params_map_path, label_path)
    data_module.setup()
    print("Number of training data:", len(data_module.train_dataset))
    print("Number of validation data:", len(data_module.val_dataset))
    print("Number of test data:", len(data_module.test_dataset))

    logger = TensorBoardLogger(
        save_dir="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/logs",
        name=f"{config.name}",
    )

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
        freeze_ratio=0.0,  # Don't freeze pretrained parameters, let model adapt
        pretrained_ckpt=config.pretrained_ckpt,
        w=config.w,
        free_bits=config.free_bits,
        warmup_beta=config.warmup_beta,
        max_beta=config.max_beta,
        beta_warmup_steps=config.beta_warmup_steps,
        kl_annealing=config.kl_annealing,
    )

    checkpoint = ModelCheckpoint(
        dirpath="/home/sunx/data/aiiih/projects/sunx/projects/TEEMR/PT/outputs/checkpoints",
        filename=f"best_{config.name}",
        save_weights_only=True,
        save_last=False,
        every_n_train_steps=config.ckpt_every_n_steps,
        monitor="val_auc",
        mode="max",
        save_top_k=1,
        enable_version_counter=False,
    )

    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=3,
        verbose=True,
        strict=True,
    )

    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=logger,
        max_epochs=config.max_epochs,
        min_epochs=1,
        precision=config.precision,
        callbacks=[
            checkpoint,
            early_stopping,
        ],
        profiler="advanced",
        log_every_n_steps=config.log_every_n_steps,
        gradient_clip_val=config.gradient_clip_val,  # Use gradient clipping value from config
        gradient_clip_algorithm="norm",  # Use norm clipping instead of value clipping
        val_check_interval=0.04,
        limit_val_batches=0.1,
        accumulate_grad_batches=1,  # Add gradient accumulation
        detect_anomaly=True,  # Detect anomalies for debugging
    )

    trainer.fit(model, data_module)
