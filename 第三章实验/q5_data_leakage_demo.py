# -*- coding: utf-8 -*-
"""
第 3 章 3.12.3 精通级 · 第 5 题
------------------------------------------------------------
题目：构造“数据泄露”演示实验：
  ① 先对全量数据做标准化再划分（错误做法）；
  ② 先划分、只在训练集标准化（正确做法）；
  ③ 两种做法分别训练同一模型并报告测试集准确率。
  并回答：为什么低维数据上两者差别很小？什么情况下泄露会造成巨大差异？

三种实验：
  A. 低维（10 维）缩放泄露      -> 差距通常只有千分之几；
  B. 高维（500 维）缩放泄露     -> 差距明显放大；
  C. 标签泄露（用全量标签构造的特征）-> 错误做法准确率虚高到接近满分。

运行： python q5_data_leakage_demo.py

============================================================
【阅读指南】本文件按“需掌握程度”做了三级标注：
    ★★★ 必掌握   —— 本题核心：泄露的成因、后果与防止
    ★   脚手架   —— 造数据、打印排版；读代码时可跳过

本题的 ★★★ 必掌握清单：
    1. “先划分、再一切含统计量的处理”是铁律；测试集只能 transform
    2. 为什么低维时差距很小，而高维 / 测试集占比大时会被放大
    3. 最危险的是“用全量标签构造特征”（标签泄露），准确率虚高到接近满分
    4. 一句话总结：任何“学参数”的步骤，都不许看到测试集
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


# ★ 脚手架（可跳过）：训练模型、返回测试集准确率，只是个工具函数
def fit_lr(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """训练逻辑回归并返回测试集准确率。"""
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


# ==========================================================================
# ★★★ 必掌握：缩放泄露对照实验（本题主角）
#   唯一的变量是“标准化用了谁的统计量”：
#     ① 错误：在全量 X 上 fit_transform —— 测试集的 μ/σ 提前流进了训练特征；
#     ② 正确：先划分，只在训练集 fit，测试集只 transform。
#   结论要点：
#     - 低维时差距只有千分之几（μ/σ 扰动小，且逻辑回归对平移缩放不敏感）；
#     - 维度越高、测试集占比越大，泄露被累积放大（可达 2~5 个百分点）。
# ==========================================================================
def scale_leak_experiment(n_features: int, n_informative: int, tag: str, n_samples: int = 1000) -> None:
    n_redundant = min(50, max(0, n_features - n_informative))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        flip_y=0.02,
        random_state=SEED,
    )

    # ① 错误做法（要警惕）：先在全量数据上 fit 标准化，测试集的 μ/σ 已流入训练特征
    X_scaled_all = StandardScaler().fit_transform(X)
    Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
        X_scaled_all, y, test_size=0.3, random_state=SEED, stratify=y
    )
    acc_wrong = fit_lr(Xw_tr, yw_tr, Xw_te, yw_te)

    # ② 正确做法（★必掌握）：先划分，再只用训练集 fit 标准化，测试集只 transform
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    acc_correct = fit_lr(scaler.transform(X_tr), y_tr, scaler.transform(X_te), y_te)

    print("-" * 72)
    print(f"{tag}（维度={n_features}）")
    print("-" * 72)
    print(f"① 先全量标准化再划分（错误） : {acc_wrong:.4f}")
    print(f"② 先划分后仅训练集标准化（正确）: {acc_correct:.4f}")
    print(f"→ 错误做法虚高：{(acc_wrong - acc_correct) * 100:+.3f} 个百分点\n")


# ==========================================================================
# ★★★ 必掌握：标签泄露（最危险的一类泄露）
#   用“全量标签”算目标编码，再划分训练/测试：
#   等于把每一行自己的标签，通过编码值写回了它自己的特征里，模型直接看到答案。
#   典型表现：验证/测试准确率虚高到接近满分，但线上真实效果只有随机水平。
#   正确做法：先划分，映射只用训练集标签去算；测试集查表，未见类别回退全局均值。
# ==========================================================================
def label_leak_experiment(n: int = 600, n_categories: int = 300) -> None:
    rng = np.random.default_rng(SEED)
    cat = rng.choice([f"ID{i:03d}" for i in range(n_categories)], size=n)  # 每类约 2 个样本，泄露最强
    y = rng.integers(0, 2, size=n)
    df = pd.DataFrame({"cat": cat, "y": y})

    # ① 错误做法：用“全量标签”算目标编码，再划分
    df["te_leak"] = df.groupby("cat")["y"].transform("mean")
    Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
        df[["te_leak"]].to_numpy(), df["y"].to_numpy(), test_size=0.3, random_state=SEED, stratify=df["y"]
    )
    acc_wrong = fit_lr(Xw_tr, yw_tr, Xw_te, yw_te)

    # ② 正确做法：先划分；只在训练集上算映射，测试集查表（未见类别回退全局均值）
    X_tr, X_te, y_tr, y_te = train_test_split(df[["cat"]], df["y"], test_size=0.3, random_state=SEED, stratify=df["y"])
    global_mean = float(y_tr.mean())
    mapping = pd.DataFrame({"cat": X_tr["cat"].values, "y": y_tr.values}).groupby("cat")["y"].mean()
    te_tr = X_tr["cat"].map(mapping).fillna(global_mean).to_numpy().reshape(-1, 1)
    te_te = X_te["cat"].map(mapping).fillna(global_mean).to_numpy().reshape(-1, 1)
    acc_correct = fit_lr(te_tr, y_tr.to_numpy(), te_te, y_te.to_numpy())

    print("-" * 72)
    print(f"C. 标签泄露（高基数类别，{n_categories} 类 / {n} 样本，每类约 2 条）")
    print("-" * 72)
    print(f"① 全量标签算目标编码后划分（错误）: {acc_wrong:.4f}")
    print(f"② 划分后仅训练集算目标编码（正确）: {acc_correct:.4f}")
    print(f"→ 错误做法虚高：{(acc_wrong - acc_correct) * 100:+.2f} 个百分点")
    print("  （错误做法把每一行自己的标签写进了它自己的特征里，等于把答案喂给模型）\n")


# ★ 脚手架：main 只是把 A/B/C 三个实验依次跑一遍，流程可跳过；
#   结尾的“结论”部分是本题要记的核心知识，建议精读
def main() -> None:
    print("=" * 72)
    print("数据泄露演示实验")
    print("=" * 72)
    print("实验设置：同一份数据、同一个模型（LogisticRegression），")
    print("唯一差异是“预处理时机”——全量 fit（错误） vs 训练集 fit（正确）。\n")

    scale_leak_experiment(n_features=10, n_informative=8, tag="A. 低维缩放泄露")
    scale_leak_experiment(n_features=500, n_informative=50, tag="B. 高维缩放泄露")
    label_leak_experiment()

    print("=" * 72)
    print("结论")
    print("=" * 72)
    print(
        "1) 为什么低维（10 维）差别很小？\n"
        "   标准化只用到两个统计量 μ、σ。维度低时，测试集对 μ/σ 的扰动极小，\n"
        "   而且逻辑回归对特征平移缩放不敏感，所以两种做法几乎同分（千分之几）。\n\n"
        "2) 什么情况下泄露会造成巨大差异？\n"
        "   (a) 维度极高：几百上千个特征，每个特征都“偷偷”带一点测试集信息（μ/σ），\n"
        "       累积起来能把错误做法抬升 2~5 个百分点；\n"
        "   (b) 预处理本身用到了标签：全量做特征选择 / PCA / 目标编码 / 过采样，\n"
        "       会把标签信息直接写进特征。如实验 C，错误做法准确率可虚高到接近满分，\n"
        "       而真实泛化能力只有随机水平——这是最危险、也最常见的一类泄露；\n"
        "   (c) 数据量小、测试集占比大：测试集统计量占比高，泄露效应被放大。\n\n"
        "3) 根本原则：先划分，再一切含统计量的处理；\n"
        "   所有“学参数”的步骤只在训练集 fit；用 Pipeline 把预处理包起来，\n"
        "   交给 cross_val_score 自动隔离。"
    )


if __name__ == "__main__":
    main()