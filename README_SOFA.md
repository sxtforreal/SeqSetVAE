## 实时 SOFA 评分脚本

使用此脚本为每位病人的每个 set（时间点/测量组）计算实时 SOFA 分数。脚本按患者时间排序并在患者内部进行前向填充，以模拟“实时可得”的最新指标，然后计算各器官子分及总分。

### 安装依赖

```bash
pip install -r requirements.txt
```

### 输入数据格式（宽表）

- 至少包含：
  - `patient_id`: 病人唯一标识
  - `event_time`: 时间戳（ISO 或常见日期格式）
- 推荐包含 `set_id`（如果你的数据天然有 set 概念）。
- 其他列名可通过映射文件自定义，对应字段详见下文。

### 运行示例

```bash
python sofa_realtime.py \
  --input your_data.csv \
  --output sofa_scores.csv \
  --mapping sofa_mapping.example.yaml
```

不提供 `--mapping` 时，脚本默认使用与示例相同的列名。

### 字段映射说明（YAML）

参考 `sofa_mapping.example.yaml`，可按你的数据列名调整：

- **通用**
  - `patient_id`: 病人ID
  - `set_id`: 组/批次编号（可选）
  - `time`: 时间戳列名
- **呼吸**
  - `pao2_mmHg`: PaO2（mmHg）
  - `fio2_fraction`: FiO2（小数，如 0.4）
  - `spo2_percent`: SpO2（百分比，如 95）
  - `mech_vent`: 是否机械通气（0/1 或 True/False）
- **凝血**
  - `platelets_10e9_per_L`: 血小板（10^9/L）
- **肝功能**（两者其一）
  - `bilirubin_mg_dl` 或 `bilirubin_umol_L`
- **心血管**（剂量单位均为 mcg/kg/min）
  - `map_mmHg`, `norepinephrine_mcg_kg_min`, `epinephrine_mcg_kg_min`, `dopamine_mcg_kg_min`, `dobutamine_mcg_kg_min`
- **中枢神经**
  - `gcs_total`
- **肾功能**
  - `creatinine_mg_dl` 或 `creatinine_umol_L`
  - `urine_output_ml_24h`（24h 尿量，ml/24h）

### 评分规则（简述）

- **呼吸**：优先按 P/F 比值（PaO2/FiO2）。若缺失 PaO2，则按 S/F（SpO2/FiO2）近似阈值；当 P/F<200 或 P/F<100 时需机械通气才可分别计 3/4 分。
- **凝血**：血小板（10^9/L）阈值：<150/100/50/20 对应 1/2/3/4 分。
- **肝**：胆红素（mg/dL）阈值：≥1.2/2.0/6.0/12.0 对应 1/2/3/4 分（自动支持 umol/L 换算）。
- **心血管**：优先按升压药剂量（mcg/kg/min）分层；否则若 MAP<70 计 1 分。
- **中枢神经**：GCS 15/13-14/10-12/6-9/<6 对应 0/1/2/3/4 分。
- **肾**：按肌酐（mg/dL）和/或 24h 尿量计算，取两者分值较高者；肌酐 ≥1.2/2.0/3.5/5.0 对应 1/2/3/4 分；尿量 <500/<200 ml/24h 对应 3/4 分。

输出文件包含：`patient_id`、`set_id`（若存在）、`event_time`、六个子分以及 `sofa_total`。

### 常见问题

- **实时性如何体现？** 按患者时间排序后对缺失项前向填充，代表在某个 set 时间点使用“当时已知的最新值”进行评分。
- **单位不一致怎么办？** 脚本已内置胆红素（umol/L→mg/dL）与肌酐（umol/L→mg/dL）换算；其他单位请在导入前统一。
- **没有 PaO2 只有 SpO2？** 将使用 S/F 近似阈值进行呼吸子分评估。

