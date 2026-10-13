# -*- coding: utf-8 -*-
"""
第 3 章 3.12.3 精通级 · 第 2 题
------------------------------------------------------------
题目：对比标准化(StandardScaler)与归一化(MinMaxScaler)处理后的数据分布差异。

结论要点（脚本会自动打印）：
  - 标准化后：均值≈0，标准差≈1，范围不固定（通常会超出 [0,1]，有负数）；
  - 归一化后：严格落在 [0,1]，但均值/方差不定；
  - 两者的分布“形状”（偏态方向、峰形）完全一致 —— 缩放不改变分布形状；
  - 关键观察：当列里存在极端值时，归一化会把其余绝大多数值挤到靠近 0 处。

运行： python q2_standard_vs_minmax.py
       图会保存到同目录 figures/ 下。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

matplotlib.use("Agg")  # 无界面环境也能出图；如需交互可改为默认后端
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体，避免图上中文变方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
RNG = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


def make_column(n: int = 1000, with_outlier: bool = True) -> np.ndarray:
    """构造一列右偏数据，可选地注入一个极端值。"""
    col = RNG.lognormal(mean=3.0, sigma=0.6, size=n)
    if with_outlier:
        col[0] = col.max() * 8.0  # 注入一个极端尖峰
    return col


def describe(name: str, arr: np.ndarray) -> dict:
    return {
        "处理方式": name,
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "偏度": round(float(pd.Series(arr).skew()), 4),
    }


def compare(original: np.ndarray, tag: str, slug: str) -> None:
    std = StandardScaler().fit_transform(original.reshape(-1, 1)).ravel()
    mmx = MinMaxScaler().fit_transform(original.reshape(-1, 1)).ravel()

    print("=" * 72)
    print(f"场景：{tag}")
    print("=" * 72)
    table = pd.DataFrame(
        [describe("原始数据", original), describe("标准化 StandardScaler", std), describe("归一化 MinMaxScaler", mmx)]
    )
    print(table.to_string(index=False))
    print(
        "\n观察：\n"
        f"  - 标准化后范围 [{std.min():.2f}, {std.max():.2f}]，出现负值，均值≈0、标准差≈1；\n"
        f"  - 归一化后范围 [{mmx.min():.2f}, {mmx.max():.2f}]，严格落在 [0,1]；\n"
        f"  - 三者的偏度方向一致（均为右偏），说明缩放不改变分布形状。\n"
        f"  - 归一化后中位数被压到 {np.median(mmx):.4f}（接近 0），"
        f"说明极端值把绝大多数正常样本挤在了低端。"
    )

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, data, title, color in zip(
        axes,
        [original, std, mmx],
        ["原始数据", "标准化 StandardScaler", "归一化 MinMaxScaler"],
        ["#7f8c8d", "#2980b9", "#c0392b"],
    ):
        ax.hist(data, bins=50, color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("取值")
        ax.set_ylabel("样本数")
        ax.grid(alpha=0.3)
    fig.suptitle(f"标准化 vs 归一化分布对比（{tag}）", fontsize=13)
    fig.tight_layout()
    out = OUT_DIR / f"q2_scaling_{slug}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  已保存图像 -> {out}\n")


def main() -> None:
    print("对比目标：同一列数据分别做标准化与归一化，比较统计量与分布形状。\n")

    print(">>> 场景 A：含极端值（更能暴露归一化的弱点）")
    compare(make_column(with_outlier=True), "含极端值", "with_outlier")

    print(">>> 场景 B：不含极端值（两者差异更温和）")
    compare(make_column(with_outlier=False), "不含极端值", "no_outlier")

    print("=" * 72)
    print("最终结论")
    print("=" * 72)
    print(
        "1) 数学上两者都只是平移 + 缩放（仿射变换），是保序的，不改变分布形状；\n"
        "2) 标准化以 μ 为中心、以 σ 为尺度，对离群值相对稳健，适用于线性/逻辑回归、\n"
        "   SVM、PCA、KNN、K-Means、神经网络等；\n"
        "3) 归一化把数据强行映射到 [0,1]，对极值极度敏感：一个 999 就能把其它样本\n"
        "   压到接近 0，导致有效信息被压缩，适用于图像像素、要求输入区间固定的网络；\n"
        "4) 特征分布严重右偏且存在离群值时，优先选标准化（或先做对数变换再缩放）。"
    )


if __name__ == "__main__":
    main()