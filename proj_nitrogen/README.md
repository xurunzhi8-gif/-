# AMDR-Net：手机端小麦氮素多指标回归（PyTorch）

本项目用于 COMPAG 论文实验，覆盖：within-device / cross-device 评估、消融、backbone 对比、回归曲线、Grad-CAM、相关性分析。

## 1. 环境准备（Windows）
```bat
cd /d D:\2025实验数据\原阳
python -m venv .venv
.venv\Scripts\activate
pip install -r proj_nitrogen\requirements.txt
```

## 2. 数据划分（按 device + image_id 分层，含测试集）
```bat
cd /d D:\2025实验数据\原阳\proj_nitrogen
python scripts\split_from_metadata.py --config configs\config.yaml
```

## 3. within-device 训练与验证（iPhone / Nova）
```bat
python scripts\train_one_device.py --config configs\config.yaml --device iphone
python scripts\train_one_device.py --config configs\config.yaml --device nova
```

## 4. cross-device 测试
```bat
python scripts\test_cross_device.py --config configs\config.yaml --source iphone --target nova --run-tag base
python scripts\test_cross_device.py --config configs\config.yaml --source nova --target iphone --run-tag base
```

## 5. 消融实验（5 个开关）
```bat
python scripts\train_one_device.py --config configs\config.yaml --device iphone --ablation baseline
python scripts\train_one_device.py --config configs\config.yaml --device nova --ablation baseline
python scripts\train_one_device.py --config configs\config.yaml --device iphone --ablation no_fpn
python scripts\train_one_device.py --config configs\config.yaml --device iphone --ablation no_film
python scripts\train_one_device.py --config configs\config.yaml --device iphone --ablation no_task_decouple
python scripts\train_one_device.py --config configs\config.yaml --device iphone --ablation no_uncertainty
python scripts\train_one_device.py --config configs\config.yaml --device iphone --ablation no_consistency
```

## 6. Backbone 对比
```bat
python scripts\train_one_device.py --config configs\config.yaml --device iphone --backbone mobilenetv3_small
python scripts\train_one_device.py --config configs\config.yaml --device iphone --backbone mobilenetv4_small
python scripts\train_one_device.py --config configs\config.yaml --device iphone --backbone efficientnet_b0
python scripts\train_one_device.py --config configs\config.yaml --device iphone --backbone mobilevit_xs
```

## 7. 对应 run-tag 的跨设备测试
```bat
python scripts\test_cross_device.py --config configs\config.yaml --source iphone --target nova --run-tag baseline
python scripts\test_cross_device.py --config configs\config.yaml --source nova --target iphone --run-tag baseline
python scripts\test_cross_device.py --config configs\config.yaml --source iphone --target nova --run-tag no_fpn
python scripts\test_cross_device.py --config configs\config.yaml --source nova --target iphone --run-tag no_fpn
python scripts\test_cross_device.py --config configs\config.yaml --source iphone --target nova --run-tag backbone_mobilenetv3_small
python scripts\test_cross_device.py --config configs\config.yaml --source nova --target iphone --run-tag backbone_mobilenetv3_small
```

## 8. 论文表格与统计
```bat
python scripts\make_tables_device_angle.py --config configs\config.yaml
python scripts\make_paper_tables.py --config configs\config.yaml
```

## 9. 回归曲线（PNG + PDF）
```bat
python scripts\make_regression_plots.py --config configs\config.yaml
```

## 10. Grad-CAM 可解释性
```bat
python scripts\run_gradcam.py --config configs\config.yaml --device iphone --split val --task SNC_pct
python scripts\run_gradcam.py --config configs\config.yaml --device nova --split val --task shootN_gm2
python scripts\run_gradcam.py --config configs\config.yaml --device iphone --split val --task leafN_gm2
```

## 11. 相关性分析
```bat
python scripts\analyze_correlations.py --config configs\config.yaml
```

## 输出目录
- outputs/preds: 预测明细 CSV
- outputs/models: best 模型权重
- outputs/logs: 训练日志
- outputs/tables: 论文表格
- outputs/figs: 回归图与 Grad-CAM
