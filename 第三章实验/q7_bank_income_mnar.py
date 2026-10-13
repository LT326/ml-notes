# -*- coding: utf-8 -*-
"""
第 3 章 3.12.3 精通级 · 第 7 题（讨论题，附代码演示）
------------------------------------------------------------
题目：银行信贷违约预测。“月收入”列有 30% 缺失，且怀疑“收入越高越不愿填”。
      请设计完整的处理方案：缺失机制判断、填充方式、缺失指示特征、文档记录假设，
      并说明方案可能引入的偏差。

脚本用一份模拟数据把上述设计“跑一遍”，并打印：
  ① 缺失机制诊断（证明是 MNAR，而非 MCAR/MAR）
  ② 不同填充策略相对“真值”的偏差
  ③ 缺失指示特征的价值
  ④ 建模对比：删行 vs 中位数填充 vs 分组填充+指示特征（ROC-AUC）
  ⑤ 数据字典 / 假设记录（可直接抄进项目文档）

运行： python q7_bank_income_mnar.py

============================================================
【阅读指南】本文件按“需掌握程度”做了三级标注：
    ★★★ 必掌握   —— 缺失值处理核心考点，要能自己讲清原理
    ★★  工程规范 —— 工程化的正确写法，重点理解“为什么”
    ★   脚手架   —— 造数据、打印排版；读代码时可跳过

本题的 ★★★ 必掌握清单：
    1. 三种缺失机制：MCAR / MAR / MNAR，以及怎么用数据判断是哪一种
    2. MNAR 下“缺失指示特征”为什么有预测力（缺失本身携带信息）
    3. 任何填充都会引入偏差，要量化出来；填充参数只能从训练集算
    4. 不同处理方案的评估必须“同一批人”才可比；数据字典要记录假设与偏差
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
RNG = np.random.default_rng(SEED)
N = 5000
TARGET = "default"
MISSING_RATE_TARGET = 0.30


# ==========================================================================
# ★ 脚手架（可跳过）：造一份“收入按 MNAR 缺失”的数据
#   _solve_intercept 用二分法调截距，把缺失率凑到目标值（30%），纯属造数据的技巧。
#   你只需知道它实现了“收入越高 -> 越容易缺失”，从而模拟出 MNAR 机制。
# ==========================================================================
def _solve_intercept(z: np.ndarray, target_rate: float) -> float:
    """二分法求截距 b，使 sigmoid(a*z+b) 的均值≈target_rate（a 固定为 1）。"""
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(z + mid)))).mean()
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_data(n: int = N) -> pd.DataFrame:
    city = RNG.choice(["北京", "上海", "广州", "深圳", "杭州", "成都"], size=n,
                      p=[0.22, 0.20, 0.18, 0.15, 0.13, 0.12])
    occupation = RNG.choice(["技术", "销售", "教师", "医生", "个体"], size=n,
                            p=[0.30, 0.25, 0.18, 0.12, 0.15])
    age = RNG.normal(38, 10, size=n)
    tenure = RNG.integers(0, 130, size=n)

    city_inc = {"北京": 11.3, "上海": 11.25, "广州": 11.0, "深圳": 11.15, "杭州": 11.05, "成都": 10.7}
    occ_inc = {"技术": 0.20, "销售": 0.05, "教师": -0.05, "医生": 0.10, "个体": -0.15}
    mu = np.array([city_inc[c] for c in city]) + np.array([occ_inc[o] for o in occupation])
    true_income = RNG.lognormal(mean=mu, sigma=0.35, size=n)  # 真实月收入（元/月量级视 mu 指数）

    # 违约概率：收入越低越容易违约
    income_std = (np.log(true_income) - np.log(true_income).mean()) / np.log(true_income).std()
    logit = -0.8 * income_std + 0.02 * (age - 38) + 0.4 * (occupation == "个体") - 0.5
    prob = 1 / (1 + np.exp(-logit))
    default = (RNG.random(n) < prob).astype(int)

    # MNAR：收入越高，越不愿意填
    b = _solve_intercept(income_std / income_std.std() * 0.9, MISSING_RATE_TARGET)
    miss_prob = 1 / (1 + np.exp(-((income_std / income_std.std() * 0.9) + b)))
    is_missing = RNG.random(n) < miss_prob

    income_obs = true_income.copy()
    income_obs[is_missing] = np.nan

    return pd.DataFrame(
        {"city": city, "occupation": occupation, "age": age, "tenure": tenure,
         "income": income_obs, "_true_income": true_income, "income_missing": is_missing.astype(int),
         TARGET: default}
    )


# ==========================================================================
# ★★★ 必掌握：缺失值处理的三个要点（本题核心）
#   1) 分组填充优于全局填充：按城市/职业分组算中位数，比一个全局均值更贴近个体；
#   2) 加“缺失指示特征”：把“这一格是否缺失”本身变成一列 0/1 特征。
#      MNAR 场景下“缺失”本身就带信息（收入高的人更不填），这列是有预测力的；
#   3) 防泄露：分组中位数、全局中位数都必须只在【训练折】里算（见 fit）。
# ★★ 工程规范：同样继承 BaseEstimator + TransformerMixin，
#   fit 只学训练数据、transform 只套用学到的映射。
# ==========================================================================
class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """按分组列计算中位数，填充目标列；并可选追加“是否缺失”指示列。"""

    def __init__(self, value_col: str, group_col: str, add_indicator: bool = True):
        self.value_col = value_col
        self.group_col = group_col
        self.add_indicator = add_indicator

    def fit(self, X, y=None) -> "GroupMedianImputer":
        X = pd.DataFrame(X).reset_index(drop=True)
        self.group_median_ = X.groupby(self.group_col)[self.value_col].median().to_dict()
        self.global_median_ = float(X[self.value_col].median())
        return self

    def transform(self, X) -> np.ndarray:
        X = pd.DataFrame(X).reset_index(drop=True)
        fill = X[self.group_col].map(self.group_median_).fillna(self.global_median_)
        filled = X[self.value_col].fillna(fill)
        cols = [filled.to_numpy(dtype=float)]
        if self.add_indicator:
            cols.append(X[self.value_col].isna().astype(int).to_numpy(dtype=float))
        return np.column_stack(cols)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
# ★ 脚手架：main 负责依次跑 ①~⑤ 并打印结论，流程可扫过；
#   五段打印出来的“结论文字”才是本体，建议精读（尤其 ① 的 MNAR 诊断和 ⑤ 的数据字典）
def main() -> None:
    df = build_data()
    # ★★★ 必掌握：缺失机制诊断 —— 比较“缺失组 vs 未缺失组”的真实分布差异
    #   MCAR（完全随机缺失）：缺失与任何变量都无关，两组分布应基本一致；
    #   MAR （随机缺失）    ：缺失与【其他可观测变量】有关，但与自身取值无关；
    #   MNAR（非随机缺失）  ：缺失与【变量自身取值】有关 —— 本题即“收入越高越不填”。
    print("=" * 76)
    print("① 缺失机制诊断：这是 MNAR 还是 MCAR / MAR？")
    print("=" * 76)
    print(f"总体缺失率            : {df['income_missing'].mean():.2%}")
    obs = df.loc[df["income_missing"] == 0, "_true_income"]
    mis = df.loc[df["income_missing"] == 1, "_true_income"]
    print(f"缺失组 真实收入均值   : {mis.mean():,.0f}")
    print(f"未缺失组 真实收入均值 : {obs.mean():,.0f}")
    print(f"→ 缺失组收入是未缺失组的 {mis.mean() / obs.mean():.2f} 倍（若为 MCAR 应≈1）")

    df["income_decile"] = pd.qcut(df["_true_income"], 10, labels=False, duplicates="drop")
    by_decile = df.groupby("income_decile").agg(
        真实收入中位数=("_true_income", "median"), 缺失率=("income_missing", "mean")
    )
    print("\n按真实收入十分位看缺失率（单调上升 => MNAR）：")
    print(by_decile.assign(真实收入中位数=lambda d: d["真实收入中位数"].round(0),
                           缺失率=lambda d: (d["缺失率"] * 100).round(1).astype(str) + "%").to_string())
    print("结论：缺失概率随收入升高而升高 => 属于 MNAR（非随机缺失），")
    print("      且缺失本身携带“收入偏高”的信息。\n")

    # ★★★ 必掌握：填充偏差 —— 任何填充都等于在“编造”数据，要把偏差量算出来给自己看
    print("=" * 76)
    print("② 填充策略的偏差（以真实收入为基准）")
    print("=" * 76)
    true_mean = df["_true_income"].mean()
    obs_mean = df["income"].mean()
    obs_median = df["income"].median()
    group_med = df.groupby("city")["income"].median()
    group_fill = df["city"].map(group_med).fillna(obs_median)
    print(f"真实收入均值              : {true_mean:,.0f}")
    print(f"观测样本均值              : {obs_mean:,.0f}  （低估 {(1 - obs_mean / true_mean) * 100:.1f}%）")
    print(f"全局中位数                : {obs_median:,.0f}")
    print(f"全局中位数填充后均值      : {df['income'].fillna(obs_median).mean():,.0f}"
          f"  （低估 {(1 - df['income'].fillna(obs_median).mean() / true_mean) * 100:.1f}%）")
    print(f"分组(城市)中位数填充后均值: {df['income'].fillna(group_fill).mean():,.0f}"
          f"  （低估 {(1 - df['income'].fillna(group_fill).mean() / true_mean) * 100:.1f}%）")
    print(f"填充前观测收入标准差      : {df['income'].std():,.0f}")
    print(f"全局中位数填充后标准差    : {df['income'].fillna(obs_median).std():,.0f}（方差被人为压缩）")

    # ★★★ 必掌握：缺失指示特征的价值 —— 它编码的是“采集行为”，而非真实属性
    print("\n" + "=" * 76)
    print("③ 缺失指示特征（income_missing）的价值")
    print("=" * 76)
    print(f"缺失 vs 未缺失 的违约率 : {df.groupby('income_missing')[TARGET].mean().round(3).to_dict()}")
    print(f"指示特征与真实收入的相关 : {df['income_missing'].corr(df['_true_income']):+.3f}"
          f"（正相关：缺失对应更高收入）")
    print("→ 该指示列本身有预测力，因为它编码了“收入高 => 不填”这一采集现象。\n")

    # ★★★ 必掌握：评估的公平性 —— 三种方案必须在“同一批人”上比，否则分数不可比。
    #   删行方案会丢掉约 30% 样本、也无法给缺失客户打分，天然吃亏/占便宜，
    #   所以这里固定一份公共子集来公平对比。
    print("=" * 76)
    print("④ 建模对比（固定同一测试集，指标 ROC-AUC，仅用可观测信息）")
    print("=" * 76)
    # 固定一份共同测试集，保证三种策略在“同一批人”上比较，避免评估人群不同导致误判
    feats_all = ["city", "occupation", "age", "tenure", "income"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[feats_all], df[TARGET], test_size=0.3, random_state=SEED, stratify=df[TARGET]
    )
    eval_mask = X_te["income"].notna().to_numpy()      # 三种策略都能评分的公共子集
    X_te_eval, y_te_eval = X_te[eval_mask], y_te[eval_mask]

    def auc_of(pipe: Pipeline, X_fit, y_fit, X_eval, y_eval) -> float:
        pipe.fit(X_fit, y_fit)
        return roc_auc_score(y_eval, pipe.predict_proba(X_eval)[:, 1])

    # A：直接删除含缺失的行（且无法给缺失客户打分）
    tr_mask = X_tr["income"].notna()
    pipeA = Pipeline([("prep", _ct(mode="global", include_indicator=False)), ("clf", LogisticRegression(max_iter=3000))])
    aucA = auc_of(pipeA, X_tr[tr_mask], y_tr[tr_mask], X_te_eval, y_te_eval)

    # B：全局中位数填充（SimpleImputer，不加指示列）
    pipeB = Pipeline([("prep", _ct(mode="global", include_indicator=False)), ("clf", LogisticRegression(max_iter=3000))])
    aucB = auc_of(pipeB, X_tr, y_tr, X_te_eval, y_te_eval)

    # C：分组中位数填充 + 缺失指示特征
    X_tr_c = X_tr.assign(income_missing=X_tr["income"].isna().astype(int))
    X_te_eval_c = X_te_eval.assign(income_missing=X_te_eval["income"].isna().astype(int))
    X_te_c = X_te.assign(income_missing=X_te["income"].isna().astype(int))
    pipeC = Pipeline([("prep", _ct(mode="group", include_indicator=True)), ("clf", LogisticRegression(max_iter=3000))])
    aucC = auc_of(pipeC, X_tr_c, y_tr, X_te_eval_c, y_te_eval)
    aucC_full = auc_of(pipeC, X_tr_c, y_tr, X_te_c, y_te)   # C 能给全部客户打分

    print(f"公共评估子集样本数 : {eval_mask.sum()}（测试集中无缺失收入的那部分）")
    print(f"A. 删除缺失行（训练集仅剩 {tr_mask.sum()}/{len(X_tr)} 行，丢弃 {len(X_tr) - tr_mask.sum()} 行） : AUC = {aucA:.4f}"
          f"  [无法给 {int((~eval_mask).sum())} 位缺失收入的客户打分]")
    print(f"B. 全局中位数填充                                      : AUC = {aucB:.4f}")
    print(f"C. 分组中位数填充 + 缺失指示特征                       : AUC = {aucC:.4f}")
    print(f"   （C 在全测试集上的 AUC = {aucC_full:.4f}，说明它还能覆盖缺失收入的客户）")
    print("→ 删行看似分数不低，但它丢了约 31% 训练样本、且对 31% 的客户无输出；")
    print("  填充+指示特征能覆盖全量客户，尽管单看公共子集未必最高。")

    # ★★★ 必掌握：数据字典 —— 真实项目里，缺失处理必须写清“机制假设 + 方案 + 已知偏差”
    print("\n" + "=" * 76)
    print("⑤ 数据字典 / 假设记录（建议写入项目文档）")
    print("=" * 76)
    print(
        "字段: income（月收入）\n"
        "  - 缺失率: 约 30%\n"
        "  - 缺失机制假设: MNAR。证据——缺失组的真实收入显著高于未缺失组，且缺失率随收入\n"
        "    分位单调上升；说明“收入越高越不愿填”。\n"
        "  - 处理方案: ① 新增指示特征 income_missing(0/1) 保留“缺失态”语义；\n"
        "              ② 对 income 采用按城市/职业的分组中位数填充，而非全局均值/中位数；\n"
        "              ③ 不删除缺失样本（30% 比例过高，删行会损失样本并加剧偏差）。\n"
        "  - 已知偏差: 任何填充值都偏低，模型会系统性低估高收入人群的风险；\n"
        "              且模型可能把“缺失”学成“高风险”，这是采集偏差的映射而非真实因果。\n"
        "  - 上线监控: 持续监控缺失率与收入分布漂移；若采集口径变化需重估填充模型。\n"
        "  - 迭代方向: 若业务能补采，优先补采；否则可探索多列联合的模型预测填充(MICE)，\n"
        "              但填充模型必须只在训练集内训练，防止标签/信息泄露。"
    )


# ★★ 工程规范：ColumnTransformer 的组装 —— 数值列 / 低基数列 / income 填充器
#   通过 mode 参数切换“全局填充”或“分组填充 + 指示列”，用于对比不同方案。
def _ct(mode: str = "group", include_indicator: bool = True) -> ColumnTransformer:
    """构造 ColumnTransformer：数值列 + 低基数列独热 + income 填充器。

    mode="global"：income 用全局中位数填充（SimpleImputer）；
    mode="group" ：income 用分组（城市）中位数填充，可选追加缺失指示列。
    """
    num_cols = ["age", "tenure"]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    if mode == "group":
        income_transformer = GroupMedianImputer("income", "city", add_indicator=include_indicator)
    else:
        income_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    transformers = [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, ["city", "occupation"]),
        ("income", income_transformer, ["income", "city"] if mode == "group" else ["income"]),
    ]
    if include_indicator:
        transformers.append(("ind", "passthrough", ["income_missing"]))
    return ColumnTransformer(transformers=transformers)


if __name__ == "__main__":
    main()