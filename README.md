# 基于价值迭代的倒立摆全局最优控制

[![Syntax Check](https://github.com/KirkG529/vi-pendulum/actions/workflows/ci.yml/badge.svg)](https://github.com/KirkG529/vi-pendulum/actions/workflows/ci.yml)

这是一个基于 Drake 的动态规划示例项目，围绕简单倒立摆（Pendulum）实现状态/输入离散化、价值迭代求解、最优策略提取、动画仿真与结果可视化。项目同时保留了双积分器示例，便于和单摆问题做对照。

[倒立摆动画预览](https://raw.githack.com/KirkG529/vi-pendulum/main/assets/animations/pendulum.html)

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── on_a_mesh.py
├── run_experiments.py
├── combine_images.py
├── assets/
│   ├── animations/
│   │   └── pendulum.html
│   └── figures/
│       └── combined_2x2.png
├── results/
│   ├── experiment_results.json
│   ├── experiments/
│   │   ├── mild/
│   │   └── rough/
│   └── legacy/
└── 课程设计报告模版.md
```

## 环境与依赖

建议使用 Python 虚拟环境后安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

运行单摆默认实验：

```bash
python on_a_mesh.py
```

运行单摆最短时间代价版本：

```bash
python on_a_mesh.py --min-time
```

导出单摆动画到 HTML：

```bash
python on_a_mesh.py --save-animation assets/animations/pendulum.html
```

对比最短时间与二次型代价，并保存策略/代价图：

```bash
python on_a_mesh.py --compare-costs --output-dir results/experiments/mild
```

生成实验汇总数据：

```bash
source .venv/bin/activate
python3 run_experiments.py
```

合并四张图为一张 2×2 对比图：

```bash
source .venv/bin/activate
python3 combine_images.py
```

## 结果文件

- `assets/figures/combined_2x2.png`：报告中使用的 2×2 对比图
- `assets/animations/pendulum.html`：摆起动画
- `results/experiment_results.json`：实验指标汇总
- `results/experiments/mild/`：基线实验图片
- `results/experiments/rough/`：其他实验图片

## 说明

- 脚本默认会在完成后保持运行，便于 Meshcat 可视化持续打开。
- 若只想运行完就退出，可加 `--no-keepalive`。
