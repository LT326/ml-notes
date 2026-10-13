# -*- coding: utf-8 -*-
"""
第 3 章 3.12.3 精通级 · 第 1 题
------------------------------------------------------------
题目：自行构造一份包含重复、缺失、异常的模拟表格，
      完整实现清洗、分层划分、标准化全套流水线。

工程要点（评分关注点）：
  1) 处理顺序：drop_duplicates -> 异常值判定与处理(IQR) -> 分层划分 -> 标准化；
  2) 防泄露铁律：所有“学参数”的步骤（中位数、均值、方差、最频繁类别）
     只允许在训练集上 fit，测试集只允许 transform，绝不 fit_transform；
  3) 全程固定随机种子，结果可复现；
  4) 每一步都打印处理前后的形状 / 统计量，自证过程。

运行： python q1_clean_split_scale.py

============================================================
【阅读指南】本文件按“需掌握程度”做了三级标注，读代码时对号入座：
    ★★★ 必掌握   —— 数据处理核心考点，最好能自己独立默写出来
    ★★  工程规范 —— 工程化的正确写法（防泄露 / 流水线），重点是理解“为什么”
    ★   脚手架   —— 造数据、画图、打印排版；只服务演示，读代码时可跳过

本题的 ★★★ 必掌握清单：
    1. 正确的处理顺序：去重 → 异常值处理 → 划分 → 标准化（顺序错 = 泄露）
    2. IQR 判定离群值 + clip 截断（Winsorize）
    3. stratify=y 分层划分：让训练/测试的标签比例保持一致
    4. “学参数”的步骤只能在训练集 fit，测试集只 transform
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
RNG = np.random.default_rng(SEED)

NUMERIC_COLS = ["age", "income", "height_cm", "score_a", "score_b", "score_c"]
CATEGORICAL_COLS = ["city"]
TARGET_COL = "churn"


# ==========================================================================
# ★ 脚手架（可跳过）
#   下面两个函数只负责“造出一份脏数据”：重复行 / 缺失值 / 离群值 / 一点标签信号。
#   里面的随机数公式（lognormal、logit、sigmoid…）不是数据预处理知识点，
#   读代码时扫一眼知道“这列是长什么样、怎么变脏的”即可，不必细究。
# ==========================================================================
def build_raw_table(n_rows: int = 600) -> pd.DataFrame:
    """构造含重复行、缺失值、离群值的模拟表格。"""
    age = RNG.normal(35, 8, size=n_rows).round(0)
    income = RNG.lognormal(mean=2.3, sigma=0.35, size=n_rows)  # 月收入，量级约 10
    height_cm = RNG.normal(170, 8, size=n_rows)
    score_a = RNG.normal(60, 12, size=n_rows)
    score_b = RNG.normal(75, 10, size=n_rows)
    score_c = RNG.normal(50, 15, size=n_rows)
    city = RNG.choice(
        ["北京", "上海", "广州", "深圳", "杭州"],
        size=n_rows,
        p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )

    # 标签：让“收入低 / 分数低”的人更倾向流失，制造一点可学习的信号
    logit = -0.6 * (score_a - 60) / 12.0 - 0.15 * (income - income.mean()) + 0.4
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (RNG.random(n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "height_cm": height_cm,
            "score_a": score_a,
            "score_b": score_b,
            "score_c": score_c,
            "city": city,
            TARGET_COL: churn,
        }
    )

    # 人为注入缺失（数值列 + 类别列）
    for col, frac in {"age": 0.05, "income": 0.08, "height_cm": 0.03, "city": 0.04}.items():
        idx = RNG.choice(df.index, size=int(len(df) * frac), replace=False)
        df.loc[idx, col] = np.nan

    # 人为注入离群值：把部分收入的数值写成 999（明显是错误数据）
    out_idx = RNG.choice(df.index, size=12, replace=False)
    df.loc[out_idx, "income"] = 999.0

    return df


def inject_duplicates(df: pd.DataFrame, n_dup: int = 40) -> pd.DataFrame:
    """复制若干行制造完全重复行（模拟多源合并 / 采集重复）。"""
    dup = df.sample(n=n_dup, random_state=SEED)
    return pd.concat([df, dup], ignore_index=True)


# ==========================================================================
# ★★★ 必掌握：异常值判定与处理（IQR 法）
#   - Q1 / Q3 / IQR 的定义与计算
#   - 异常边界：下界 = Q1 - 1.5*IQR 、上界 = Q3 + 1.5*IQR（k 就是 1.5 这个系数）
#   - 两种处理方式：
#       ① 截断 clip（Winsorize）：保留样本，只把极值压回边界
#       ② 删除：仅当确认是录入错误、且样本量充足时才用
#   - 本题选“截断”的理由：样本有限，999 是明显错误值，删行会白白丢样本
# ==========================================================================
def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """返回 IQR 法的下界、上界。"""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def handle_outliers_iqr(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> tuple[pd.DataFrame, dict]:
    """用 IQR 判定离群值，并采用“截断(Winsorize)”而非删除：保留样本、抑制极值。

    说明：本数据集收入列存在被污染的 999，且样本量有限，删除会损失样本，
    因此按业务规则选择“截断到边界”。若确认是录入错误，也可改为删除。
    """
    df = df.copy()
    report: dict[str, dict] = {}
    for col in cols:
        low, high = iqr_bounds(df[col].dropna(), k=k)
        mask = (df[col] < low) | (df[col] > high)
        report[col] = {
            "lower": round(low, 2),
            "upper": round(high, 2),
            "n_outlier": int(mask.sum()),
        }
        df[col] = df[col].clip(lower=low, upper=high)
    return df, report


# ==========================================================================
# ★★ 工程规范：把“填充 / 缩放 / 编码”组装成流水线
#   ColumnTransformer = 按“列类型”把不同列分发给各自的子 Pipeline：
#       数值列 -> 中位数填充 + 标准化
#       类别列 -> 众数填充 + 独热编码
#   把它们封装成“一个对象”之后，才能塞进 cross_val_score，
#   由交叉验证在每个训练折内自动 fit —— 这是工程上防泄露的关键手法。
# ==========================================================================
def build_preprocessor() -> ColumnTransformer:
    """数值列：中位数填充 + 标准化；类别列：众数填充 + 独热编码。"""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
        ]
    )


def main() -> None:
    print("=" * 70)
    print("【第 1 步】构造原始脏数据")
    print("=" * 70)
    df = inject_duplicates(build_raw_table(600), n_dup=40)
    print(f"原始形状（含重复）        : {df.shape}")
    print(f"缺失值合计                : {int(df.isna().sum().sum())}")
    print(f"income 中被污染的 999 数量 : {int((df['income'] == 999).sum())}")

    print("\n" + "=" * 70)
    print("【第 2 步】清洗：去重")
    print("=" * 70)
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"去重前 / 后行数           : {before} -> {df.shape[0]}（删除 {before - df.shape[0]} 行）")

    print("\n" + "=" * 70)
    print("【第 3 步】清洗：异常值判定与处理（IQR 截断）")
    print("=" * 70)
    df, out_report = handle_outliers_iqr(df, ["income", "height_cm", "score_c"], k=1.5)
    for col, info in out_report.items():
        print(f"{col:<10} 下界={info['lower']:>8} 上界={info['upper']:>8} 判定异常 {info['n_outlier']:>3} 个")
    print(f"处理 999 后 income 最大值  : {df['income'].max():.2f}")

    print("\n" + "=" * 70)
    print("【第 4 步】分层划分（stratify=y）")
    print("=" * 70)
    # ★★★ 必掌握：stratify=y —— 让训练集/测试集的标签比例与原始数据保持一致。
    #   分类任务（尤其类别不均衡时）默认都要加，否则测试集可能几乎全是某一类。
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    print(f"训练集 / 测试集形状       : {X_train.shape} / {X_test.shape}")
    print(f"训练集标签比例            : {y_train.value_counts(normalize=True).round(3).to_dict()}")
    print(f"测试集标签比例            : {y_test.value_counts(normalize=True).round(3).to_dict()}")

    print("\n" + "=" * 70)
    print("【第 5 步】标准化流水线：只在训练集 fit，测试集只 transform")
    print("=" * 70)
    # ★★★ 必掌握：全篇最核心的两行 —— fit 与 transform 的分工
    #   fit_transform 只在【训练集】用：中位数 / 均值 / 标准差都从训练集算出来；
    #   测试集只能 transform：直接套用训练集学到的统计量，绝不自己 fit。
    #   一句话记忆：测试集的任何“统计量”都不许回流到训练过程。
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)   # 关键：fit 只发生在训练集
    X_test_t = preprocessor.transform(X_test)         # 测试集绝不 fit_transform

    feat_names = list(preprocessor.get_feature_names_out())
    print(f"处理后训练集形状          : {X_train_t.shape}")
    print(f"处理后测试集形状          : {X_test_t.shape}")
    print(f"产出特征名（前 8 个）     : {feat_names[:8]} ...")

    # ★ 脚手架：下面几行只是“取回数值列、打印统计量”来自证结果，可跳过
    num_idx = [feat_names.index(f"num__{c}") for c in NUMERIC_COLS]
    num_block = X_train_t[:, num_idx]
    print("\n训练集数值块标准化后统计（均值≈0、标准差≈1 才说明 fit 正确）：")
    print(pd.DataFrame(num_block, columns=NUMERIC_COLS).agg(["mean", "std"]).round(4))

    print("\n[OK] 流水线跑通：去重 -> IQR 截断 -> 分层划分 -> 仅在训练集 fit 标准化。")
    print("[提示] 测试集均值/标准差不必等于 0/1，因为它复用的是训练集的统计量。")


if __name__ == "__main__":
    main()