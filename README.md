# 苗情分析：小麦 RGB 氮素诊断模型（深度学习版）

本项目提供一个基于**小麦 RGB 图像**的氮素诊断深度学习模板，包含：
- 数据规范与组织方式
- 最新主流视觉模型（MobileNetV4）微调
- 训练与评估（MAE / R2），并对不同角度与品种进行对比
- 训练后模型的预测接口

> 适用于小麦田间或温室采集的 RGB 图像，用于氮素含量/氮素等级诊断。

## 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 数据组织

准备一个 CSV 文件，至少包含以下列：

- `image_path`：图像路径（相对于 CSV 文件所在目录或绝对路径）
- `target`：氮素标签（可以是连续值或等级标签）
- `angle`：拍摄角度（例如 `tilted` / `top`）
- `variety`：品种名称（例如 `Jing411` / `Xiaoyan22`）

示例：`data/sample.csv`

```csv
image_path,target,angle,variety
images/plot1.jpg,2.3,tilted,Jing411
images/plot2.jpg,1.7,top,Xiaoyan22
```

> 建议统一拍摄光照与视角，避免强光/阴影干扰。

## 3. 训练模型（深度学习）

```bash
python train.py \
  --csv data/sample.csv \
  --output models/nitrogen_model.pt \
  --model mobilenetv4_conv_small \
  --pretrained \
  --epochs 20 \
  --batch-size 16 \
  --device cuda
```

可选模型：
- `mobilenetv4_conv_small`（默认，推荐）
- 其他 MobileNetV4 变体（可通过 `timm.list_models("mobilenetv4*")` 查看）

训练日志会输出每个 epoch 的 MAE / R2 指标，并分别输出不同拍摄角度与品种的对比结果。

## 4. 预测

```bash
python predict.py \
  --model models/nitrogen_model.pt \
  --images data/images/plot1.jpg data/images/plot2.jpg
```

## 5. 如何进一步改进

- 替换为更强的 backbone（如 ConvNeXt、Swin Transformer）
- 加入数据增强策略（RandAugment、MixUp）
- 结合多时相、多光谱数据融合建模

## 6. 目录结构

```
.
├── data/
│   └── sample.csv
├── models/
├── train.py
├── predict.py
└── requirements.txt
```
