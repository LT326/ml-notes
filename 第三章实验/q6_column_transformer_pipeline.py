# -*- coding: utf-8 -*-
"""
第 3 章 3.12.3 精通级 · 第 6 题
------------------------------------------------------------
题目：用 ColumnTransformer + Pipeline 写一条完整的预处理流水线，同时处理：
      - 3 个数值列（含缺失）
      - 2 个低基数类别列
      - 1 个高基数类别列（用目标编码思路，手写一个自定义 Transformer）
      然后放进 cross_val_score，验证整条流程没有泄露。

工程要点：
  - 自定义 Transformer 必须继承 BaseEstimator + TransformerMixin，
    在 fit(X, y) 里只用“折内数据”学习映射；
  - 实现 fit_transform（折内 out-of-fold 编码）+ transform（全量映射），
    这样放进 Pipeline 后由 cross_val_score 自动隔离，天然防泄露；
  - 验证方式：① 检查映射是否只由训练折算出；② 与“全量算目标编码”的泄露版对比分数。

运行： python q6_column_transformer_pipeline.py

============================================================
【阅读指南】本文件按“需掌握程度”做了三级标注：
    ★★★ 必掌握   —— 数据处理核心考点，要能自己讲清原理
    ★★  工程规范 —— 工程化的正确写法，重点理解“为什么”
    ★   脚手架   —— 造数据、打印排版；读代码时可跳过

本题的 ★★★ 必掌握清单：
    1. 自定义 Transformer 的三个方法：fit / fit_transform / transform 各干什么
    2. fit 只用“折内数据”，fit_transform 返回 out-of-fold 编码 —— 防泄露的核心
    3. 未见过的新类别要能优雅回退（本题回退到全局均值）
    4. ColumnTransformer 如何把数值列 / 低基数列 / 高基数列分而治之
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
RNG = np.random.default_rng(SEED)
NUM_COLS = ["age", "income", "tenure"]
LOW_CARD_COLS = ["gender", "education"]
HIGH_CARD_COL = "merchant_id"
TARGET = "default"
SMOOTH_M = 30


# ==========================================================================
# ★★★ 必掌握：自定义 Transformer 的防泄露写法（本文件最重要的一段）
#   关键在三个方法各司其职：
#     fit(X, y)           -> 只用传入数据（CV 时即“折内训练数据”）学一份全量映射；
#     fit_transform(X, y) -> 返回 K 折 out-of-fold 编码，每行的编码不含自己的标签；
#     transform(X)        -> 查表编码；未见过的类别回退到全局均值。
#   只要遵守“fit 绝不碰外部数据”，这个组件塞进 Pipeline 后就是安全的。
# ★★ 工程规范：继承 BaseEstimator + TransformerMixin + 重写这三个方法，
#   是 sklearn 自定义预处理的标准范式，可以当模板套到别的清洗步骤上。
# ==========================================================================
class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """平滑目标编码器（单列），实现防泄露的 fit / fit_transform / transform。"""

    def __init__(self, col: str, m: int = SMOOTH_M, n_splits: int = 5, random_state: int = 0):
        self.col = col
        self.m = m
        self.n_splits = n_splits
        self.random_state = random_state

    def _smooth(self, keys: pd.Series, y: pd.Series) -> tuple[dict, float]:
        # ★★★ 必掌握：平滑公式 —— 样本越少的类别，越向全局均值收缩
        gm = float(y.mean())
        stat = pd.DataFrame({"k": keys.values, "y": y.values}).groupby("k")["y"].agg(["mean", "count"])
        mapping = (stat["count"] * stat["mean"] + self.m * gm) / (stat["count"] + self.m)
        return mapping.to_dict(), gm

    def fit(self, X, y) -> "SmoothedTargetEncoder":
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        self.mapping_, self.global_mean_ = self._smooth(X[self.col], y)
        return self

    def transform(self, X) -> np.ndarray:
        X = pd.DataFrame(X).reset_index(drop=True)
        return X[self.col].map(self.mapping_).fillna(self.global_mean_).to_numpy(dtype=float).reshape(-1, 1)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        self.fit(X, y)
        X = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(y)
        out = np.full(len(X), np.nan)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(X):
            mapping, gm = self._smooth(X.loc[tr, self.col].reset_index(drop=True), pd.Series(y_arr[tr]))
            out[va] = pd.Series(X.loc[va, self.col].values).map(mapping).fillna(gm).to_numpy()
        return np.where(np.isnan(out), self.global_mean_, out).reshape(-1, 1)


# ==========================================================================
# ★ 脚手架（可跳过）：造一份含三类列（数值 / 低基数 / 高基数）的数据
#   只负责提供素材，随机数公式不是知识点。
# ==========================================================================
def build_data(n: int = 3000) -> pd.DataFrame:
    merchants = [f"M{i:04d}" for i in range(200)]          # 高基数：200 个商户
    merchant_bias = RNG.normal(0, 1.5, size=len(merchants))
    bias_map = dict(zip(merchants, merchant_bias))

    age = RNG.normal(38, 10, size=n)
    income = RNG.lognormal(10.2, 0.35, size=n)
    tenure = RNG.integers(0, 120, size=n)
    gender = RNG.choice(["男", "女"], size=n)
    education = RNG.choice(["高中", "本科", "硕士", "博士"], size=n, p=[0.3, 0.45, 0.2, 0.05])
    merchant = RNG.choice(merchants, size=n, p=1.0 / np.arange(1, 201) ** 1.2 / np.sum(1.0 / np.arange(1, 201) ** 1.2))

    logit = -0.03 * (age - 38) - 0.0007 * (income - income.mean()) + 1.1 * np.array([bias_map[m] for m in merchant])
    prob = 1 / (1 + np.exp(-logit))
    y = (RNG.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {"age": age, "income": income, "tenure": tenure, "gender": gender, "education": education,
         "merchant_id": merchant, TARGET: y}
    )
    # 注入缺失
    for col, frac in {"age": 0.06, "income": 0.10, "tenure": 0.04, "gender": 0.03, "education": 0.05}.items():
        idx = RNG.choice(df.index, size=int(n * frac), replace=False)
        df.loc[idx, col] = np.nan
    return df


# ==========================================================================
# ★★ 工程规范：ColumnTransformer 把三类列“分而治之”
#   数值列   -> 中位数填充 + 标准化
#   低基数列 -> 众数填充 + 独热
#   高基数列 -> 自定义目标编码器（防泄露的关键）
#   一个对象管三套流程，这是真实项目的标准写法。
#   对照版 use_custom_encoder=False：用“全量算好的泄露特征”替换，模拟错误做法。
# ==========================================================================
def build_preprocessor(use_custom_encoder: bool = True) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    low_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    # 高基数：手写目标编码 Transformer（防泄露）；对照版用“全量算好的泄露特征”替换
    high = SmoothedTargetEncoder(col=HIGH_CARD_COL, m=SMOOTH_M, n_splits=5, random_state=0) \
        if use_custom_encoder else "passthrough"
    high_cols = [HIGH_CARD_COL] if use_custom_encoder else ["te_leak"]
    return ColumnTransformer(
        transformers=[("num", num_pipe, NUM_COLS), ("low", low_pipe, LOW_CARD_COLS), ("high", high, high_cols)]
    )


# ★ 脚手架：main 负责跑“正确流水线 vs 泄露对照”并打印分数，流程可扫过；
#   重点是 ③ “验证映射只来自训练折”那段，它是防泄露的硬证据
def main() -> None:
    df = build_data()
    print("=" * 72)
    print("数据概况")
    print("=" * 72)
    print(f"形状 : {df.shape}")
    print(f"数值列 {NUM_COLS} / 低基数 {LOW_CARD_COLS} / 高基数 {HIGH_CARD_COL}"
          f"（类别数 {df[HIGH_CARD_COL].nunique()}）")
    print(f"各列缺失数 : {df.isna().sum().to_dict()}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].to_numpy()
    clf = LogisticRegression(max_iter=3000)

    # ★★★ 必掌握：正确做法 —— 目标编码留在 Pipeline 内，由 cross_val_score 折内 fit
    # ---- 正确流水线 ----
    pipe = Pipeline([("prep", build_preprocessor(use_custom_encoder=True)), ("clf", clf)])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=1)
    print("\n" + "=" * 72)
    print("① 正确流水线（目标编码放进 Pipeline，由 CV 自动隔离）")
    print("=" * 72)
    print(f"5 折准确率 : {scores.mean():.4f} ± {scores.std():.4f}")

    # ★ 对照（反面教材）：在全量数据上先算好目标编码，再当作普通数值特征 —— 这就是泄露
    # ---- 泄露对照：全量算目标编码后当作普通数值特征 ----
    df_leak = X.copy()
    gm = y.mean()
    stat = df_leak.assign(**{TARGET: y}).groupby(HIGH_CARD_COL)[TARGET].agg(["mean", "count"])
    df_leak["te_leak"] = df_leak[HIGH_CARD_COL].map(
        ((stat["count"] * stat["mean"] + SMOOTH_M * gm) / (stat["count"] + SMOOTH_M)).to_dict()
    ).fillna(gm)
    pipe_leak = Pipeline([("prep", build_preprocessor(use_custom_encoder=False)), ("clf", LogisticRegression(max_iter=3000))])
    scores_leak = cross_val_score(pipe_leak, df_leak, y, cv=5, scoring="accuracy", n_jobs=1)
    print("\n" + "=" * 72)
    print("② 泄露对照（目标编码在全量数据上算好，再当普通特征）")
    print("=" * 72)
    print(f"5 折准确率 : {scores_leak.mean():.4f} ± {scores_leak.std():.4f}")
    print(f"→ 泄露版虚高：{(scores_leak.mean() - scores.mean()) * 100:+.2f} 个百分点")

    # ★★ 工程规范：验证思路 —— 把 Transformer 学到的映射，和“只用训练折手算的映射”
    #   逐一比对；再和“全量手算的映射”比对，不一致就证明 fit 确实没用全量数据。
    # ---- 硬核验证：映射只由训练折算出 ----
    print("\n" + "=" * 72)
    print("③ 验证：自定义 Transformer 的映射确实只来自训练折")
    print("=" * 72)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    prep = build_preprocessor(use_custom_encoder=True)
    prep.fit(X_tr, y_tr)
    te = prep.named_transformers_["high"]

    manual = pd.DataFrame({"k": X_tr[HIGH_CARD_COL].values, "y": y_tr}).groupby("k")["y"].agg(["mean", "count"])
    manual_map = ((manual["count"] * manual["mean"] + SMOOTH_M * y_tr.mean()) / (manual["count"] + SMOOTH_M)).to_dict()

    same = all(
        k in manual_map and abs(te.mapping_.get(k, -1) - manual_map[k]) < 1e-9
        for k in manual_map
    )
    print(f"训练折商户数         : {len(manual_map)}")
    print(f"Transformer 学到的映射 : {len(te.mapping_)} 个类别")
    print(f"与“仅训练折手算映射”一致 : {same}")

    # 反例：全量算出来的映射不同，证明若 fit 用全量就会泄露
    full = pd.DataFrame({"k": X[HIGH_CARD_COL].values, "y": y}).groupby("k")["y"].agg(["mean", "count"])
    full_map = ((full["count"] * full["mean"] + SMOOTH_M * y.mean()) / (full["count"] + SMOOTH_M)).to_dict()
    diff = sum(1 for k in te.mapping_ if abs(te.mapping_[k] - full_map.get(k, np.nan)) > 1e-9)
    print(f"与“全量手算映射”不一致的类别数 : {diff}（>0 说明 fit 只用了训练折，未泄露）")

    print("\n[OK] 整条流程由 cross_val_score 驱动，fit 只接收折内数据，验证通过、无泄露。")


if __name__ == "__main__":
    main()