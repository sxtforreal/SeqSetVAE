# SeqSetVAE

Unified training interface with two modes:

- Pretrain (reconstruction+KL only, strict causal):
  ```bash
  python3 train.py --mode pretrain \
    --data_dir /path/to/patient_ehr \
    --params_map_path /path/to/stats.csv \
    --label_path /path/to/oc.csv
  ```
  - Batch: single patient per batch
  - Monitor: val_loss
  - Checkpoints: best + last, validated every 10% epoch

- Finetune (classification only):
  ```bash
  python3 train.py --mode finetune \
    --pretrained_ckpt /path/to/pretrain.ckpt \
    --batch_size 8 \
    --data_dir /path/to/patient_ehr \
    --params_map_path /path/to/stats.csv \
    --label_path /path/to/oc.csv
  ```
  - Batch: multi-patient
  - Backbone frozen and set to eval; only `cls_head` is trained with higher LR
  - Monitor: val_auc
  - Checkpoints: best + last, validated every 10% epoch

Notes:
- Time/position encoding is unified across modes with robust design: sinusoidal index + relative time buckets + ALiBi time bias, strict causal mask.
- For dataset structure and collate details, see `dataset.py`.