# -*- coding: utf-8 -*-
"""
第 3 章 3.12.3 精通级 · 第 3 题
------------------------------------------------------------
题目：用 sklearn.datasets.load_breast_cancer() 完成：
  ① 相关性筛选：删掉相关性 > 0.9 的冗余特征，报告删掉了多少个；
  ② 在删除后的特征上做 RFE，分别选 5/10/15/20/30 个特征，
     用 5 折交叉验证报告准确率，并描述“特征数—准确率”曲线的趋势；
  ③ 说明最终会选几个特征及理由。

工程要点：
  - 相关性矩阵要用 np.triu(..., k=1) 取上三角，避免重复计数；
  - RFE 属于“用到标签的特征选择”，必须放进 Pipeline / 交叉验证内部，
    否则特征选择这一步本身就会造成选择偏差（选择偏差也是一种泄露）；
  - 每个 k 的评估都重新 fit，避免复用同一份结果。

运行： python q3_breast_cancer_rfe.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
CORR_THRESHOLD = 0.90
K_LIST = [5, 10, 15, 20, 30]
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# ① 相关性筛选
# --------------------------------------------------------------------------
def drop_highly_correlated(X: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> tuple[pd.DataFrame, list[str]]:
    """删除相关性 > threshold 的冗余特征：对每一对高相关特征，保留先出现的一列。"""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))  # 只看上三角

    to_drop: list[str] = []
    dropped_pairs: list[tuple[str, str, float]] = []
    for col in upper.columns:
        # 若该列与某个“已保留”的列高度相关，则删除该列
        high = upper[col].dropna()
        high = high[high > threshold]
        if len(high) > 0:
            to_drop.append(col)
            partner = high.idxmax()
            dropped_pairs.append((partner, col, float(high.max())))

    print(f"相关性阈值            : {threshold}")
    print(f"上三角中 > 阈值的特征对数量 : {int((upper > threshold).sum().sum())}")
    print("被删除的冗余特征及原因（保留列 -- 删除列 : 相关系数）：")
    for keep, drop, r in dropped_pairs:
        print(f"    {keep:<28} -- {drop:<28} : {r:.4f}")
    print(f"\n[结论①] 共删除 {len(to_drop)} 个冗余特征，剩余 {X.shape[1] - len(to_drop)} 个。")
    return X.drop(columns=to_drop), to_drop


# --------------------------------------------------------------------------
# ② RFE + 5 折交叉验证
# --------------------------------------------------------------------------
def build_rfe_pipeline(n_features: int) -> Pipeline:
    """RFE 放进 Pipeline：交叉验证时会在每个训练折内重新做特征选择，杜绝选择偏差。"""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rfe", RFE(estimator=LogisticRegression(max_iter=5000), n_features_to_select=n_features)),
            ("clf", LogisticRegression(max_iter=5000)),
        ]
    )


def evaluate_k(X: pd.DataFrame, y: np.ndarray, k_values: list[int]) -> pd.DataFrame:
    n_available = X.shape[1]
    rows = []
    print("\n" + "=" * 72)
    print("② RFE + 5 折交叉验证（特征选择在折内完成）")
    print("=" * 72)
    for k in k_values:
        if k > n_available:
            print(f"  k={k:>2}：跳过（剩余特征仅 {n_available} 个，不足 {k} 个）")
            continue
        pipe = build_rfe_pipeline(k)
        scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=1)
        rows.append({"n_features": k, "acc_mean": scores.mean(), "acc_std": scores.std()})
        print(f"  k={k:>2}：5 折准确率 = {scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(rows)


def plot_curve(result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        result["n_features"],
        result["acc_mean"],
        yerr=result["acc_std"],
        marker="o",
        capsize=4,
        color="#2980b9",
    )
    ax.set_xlabel("特征数 k")
    ax.set_ylabel("5 折交叉验证准确率")
    ax.set_title("RFE 特征数 — 准确率曲线")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / "q3_rfe_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n已保存曲线图 -> {out}")


def main() -> None:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    print("=" * 72)
    print("数据集：breast_cancer")
    print("=" * 72)
    print(f"样本数 / 原始特征数 : {X.shape[0]} / {X.shape[1]}")

    # ① 相关性筛选
    print("\n" + "=" * 72)
    print("① 相关性筛选（|r| > 0.9）")
    print("=" * 72)
    X_reduced, dropped = drop_highly_correlated(X)

    # ② RFE 评估
    result = evaluate_k(X_reduced, y, K_LIST)
    plot_curve(result)

    # 趋势描述
    print("\n趋势描述：")
    best = result.loc[result["acc_mean"].idxmax()]
    for _, r in result.iterrows():
        print(f"  k={int(r['n_features']):>2} -> {r['acc_mean']:.4f}")
    print(
        f"  准确率随 k 增大先快速上升（k=5 时偏低，特征太少欠拟合），\n"
        f"  在 k≈{int(best['n_features'])} 附近达到峰值 {best['acc_mean']:.4f}，"
        f"之后趋于平台甚至轻微回落。"
    )

    # ③ 决策
    print("\n" + "=" * 72)
    print("③ 最终特征数决策")
    print("=" * 72)
    peak = result["acc_mean"].max()
    # 选择“准确率与最高分差距 < 0.5 个百分点”的最小 k（性价比拐点）
    toler = result[result["acc_mean"] >= peak - 0.005]
    chosen = int(toler["n_features"].min())
    print(
        f"  我会选 {chosen} 个特征。理由：\n"
        f"  - 它在“准确率与最高分差距 < 0.5%”的前提下特征最少，是性价比拐点；\n"
        f"  - 最高分出现在 k={int(best['n_features'])}，但相对 {chosen} 的提升仅 "
        f"{(peak - result.loc[result['n_features'] == chosen, 'acc_mean'].iloc[0]) * 100:.2f} 个百分点，\n"
        f"    为几位小数级的提升多留一倍特征，不划算（特征多 = 采集/维护成本高、更易过拟合、可解释性差）；\n"
        f"  - 若下游对可解释性更敏感，可进一步用 k={K_LIST[0]} 做权衡。"
    )


if __name__ == "__main__":
    main()