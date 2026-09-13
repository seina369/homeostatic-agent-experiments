"""読み取り器(関門 G3 / 事前登録(第一段階)4章の主指標)。GPU 不要。

文(答えの行を除いた部分)から、注入した正規化ベクトル z = (budget/B0, error/E_MAX,
uncertainty/U_MAX) を当てる 3 つの入れ子モデルを、seed 内のエピソード分割(5 分割の
交差検証。fold = episode % 5)で学習・評価し、滲みスコアと粒度スコアを出す。

  M_cov     : z_k = β_k·c + b_k                 (共変量のみ: 文の長さ・正誤)
  M_rank1   : z_k = a_k·(x·w) + β_k·c + b_k     (文の特徴から 1 つのスカラー。階数 1)
  M_signals : z_k = x·w_k + β_k·c + b_k         (文の特徴から三成分を別々に)
  M_cov ⊂ M_rank1 ⊂ M_signals

  滲みスコア = R²(M_signals) − R²(M_cov)
  粒度スコア = R²(M_signals) − R²(M_rank1)
  (R² は holdout、三成分の平均。各 fold を評価側にした 5 回の平均。草案の偶奇 2 分割から
   G3 で 5 分割に改めた。理由と数値は追記欄「G3」)

決めた設定(G3 で固定。理由と根拠の数値は事前登録(第一段階)の追記欄「G3」):
  - 文の特徴: TF-IDF(単語 1〜2gram、小文字化、sublinear_tf、min_df=2、max_features=2000、
    L2 正規化。学習側の分割だけで語彙と idf を作る)。方策モデルとは独立。
  - 共変量: 文の長さ(語数の log1p)と正誤(0/1)。学習側で標準化。切片あり。
  - 当てはめの順序(Frisch–Waugh): まず共変量だけで z を最小二乗(M_cov)。次に文の特徴と z の
    両方から共変量の影響を取り除いた残差の上で、リッジ回帰(M_signals、λ は文の特徴の重みに掛ける)。
    こうすると 3 モデルは同じ共変量部分を共有し、文の部分だけが入れ子になる。
  - M_rank1 は縮小階数回帰: M_signals の文の部分の当てはめ値 F = X·W(学習側)を特異値分解し、
    第 1 右特異ベクトル v で W を W v vᵀ に射影する(文が運ぶ情報を 1 方向に制限)。
    F の最良階数 1 近似(Eckart–Young)なので、学習側では必ず R²(M_rank1) ≤ R²(M_signals)。
  - リッジ罰則 λ = RIDGE_LAMBDA(= 4.0)。G3 の偽データ(leak_probe_synth.py、5 seed、偶奇 2 分割)で、
    乱数の文のとき両スコアが ±0.05 に収まり、状態で語を変える文のとき滲み > 0.3・粒度 > 0.2 を保つ
    値として λ ∈ {1,2,3,4,5,10} から選んだ(λ=3 は乱数で −0.055、λ=5 は粒度が 0.2 を割る seed がある)。
    5 分割への変更後も λ は動かしていない(同じ偽データで選び直すことを避けた)。

参考値(検定なし): 目的変数を逸脱成分 (f_budget, f_error, f_unc) にした版、および
合計(z の和・逸脱の和)を目的変数にした単一回帰の R²。

データ: seed_records の形式(records は StepRecord)。z は *_before から作る。
文は text から「A:」の行を除いたもの。
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emotion_grounding_env as E  # noqa: E402
from seed_records import read_seed_file  # noqa: E402

RIDGE_LAMBDA = 4.0
TFIDF_KW = dict(ngram_range=(1, 2), lowercase=True, sublinear_tf=True, min_df=2,
                max_features=2000, norm="l2", token_pattern=r"(?u)\b\w+\b")
MIN_ROWS_PER_FOLD = 10
N_FOLDS = 5          # seed 内のエピソード分割数(fold = episode % N_FOLDS)。2 = 偶奇分割。5 は G3 で採用。
_ANSWER_LINE = re.compile(r"^\s*A:.*$", re.IGNORECASE | re.MULTILINE)


# ------------------------------------------------------------
# データの取り出し
# ------------------------------------------------------------
def sentence_part(text: str) -> str:
    """答えの行(A: …)を除いた部分。"""
    return _ANSWER_LINE.sub("", text).strip()


def z_vector(rec) -> np.ndarray:
    return np.array([rec.budget_before / E.B0, rec.error_before / E.E_MAX,
                     rec.unc_before / E.U_MAX], dtype=float)


def deviation_vector(rec) -> np.ndarray:
    return np.array([E.budget_deviation(rec.budget_before), E.error_deviation(rec.error_before),
                     E.uncertainty_deviation(rec.unc_before)], dtype=float)


def rows_from_records(records, injected=None):
    """records → (texts, C, Z, D, episodes)。injected があれば z にそれを使う(注入ベクトルの記録)。"""
    texts, C, Z, D, eps = [], [], [], [], []
    for i, r in enumerate(records):
        s = sentence_part(r.text)
        n_words = len(re.findall(r"\w+", s))
        texts.append(s)
        C.append([math.log1p(n_words), 1.0 if r.correct else 0.0])
        Z.append(injected[i] if injected is not None else z_vector(r))
        D.append(deviation_vector(r))
        eps.append(r.episode)
    return texts, np.array(C, dtype=float), np.array(Z, dtype=float), np.array(D, dtype=float), np.array(eps)


# ------------------------------------------------------------
# 回帰
# ------------------------------------------------------------
def fit_models(Xtr, Ctr, Ytr, lam):
    """共変量 → 残差化 → リッジ → 階数 1 射影。3 モデルの係数をまとめて返す。"""
    n = Xtr.shape[0]
    A1 = np.concatenate([Ctr, np.ones((n, 1))], axis=1)
    Bc = np.linalg.lstsq(A1, Ytr, rcond=None)[0]              # M_cov: (q+1, k)
    Gx = np.linalg.lstsq(A1, Xtr, rcond=None)[0]              # 文の特徴を共変量で残差化するための係数 (q+1, p)
    Xr = Xtr - A1 @ Gx
    Yr = Ytr - A1 @ Bc
    p = Xr.shape[1]
    W = np.linalg.solve(Xr.T @ Xr + lam * np.eye(p), Xr.T @ Yr)   # M_signals の文の部分 (p, k)
    F = Xr @ W
    _, _, Vt = np.linalg.svd(F, full_matrices=False)
    v = Vt[0]                                                 # 第 1 右特異ベクトル (k,)
    W1 = np.outer(W @ v, v)                                   # M_rank1 の文の部分(階数 1)
    return {"Bc": Bc, "Gx": Gx, "W": W, "W1": W1, "v": v}


def predict_models(m, Xte, Cte):
    A1 = np.concatenate([Cte, np.ones((Xte.shape[0], 1))], axis=1)
    y_cov = A1 @ m["Bc"]
    Xr = Xte - A1 @ m["Gx"]
    return {"cov": y_cov, "rank1": y_cov + Xr @ m["W1"], "signals": y_cov + Xr @ m["W"]}


def r2_mean(Y, Yhat):
    """成分ごとの holdout R² の平均。分散 0 の成分は除外(全成分が 0 分散なら nan)。"""
    vals = []
    for k in range(Y.shape[1]):
        ss_tot = float(((Y[:, k] - Y[:, k].mean()) ** 2).sum())
        if ss_tot <= 0:
            continue
        ss_res = float(((Y[:, k] - Yhat[:, k]) ** 2).sum())
        vals.append(1.0 - ss_res / ss_tot)
    return float(np.mean(vals)) if vals else float("nan")


# ------------------------------------------------------------
# 1 seed の評価(エピソード分割の交差検証)
# ------------------------------------------------------------
def _standardize_fit(C):
    mu, sd = C.mean(axis=0), C.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def evaluate_split(texts, C, Y, train_idx, test_idx, lam):
    vec = TfidfVectorizer(**TFIDF_KW)
    Xtr = vec.fit_transform([texts[i] for i in train_idx]).toarray()
    Xte = vec.transform([texts[i] for i in test_idx]).toarray()
    mu, sd = _standardize_fit(C[train_idx])
    Ctr, Cte = (C[train_idx] - mu) / sd, (C[test_idx] - mu) / sd
    m = fit_models(Xtr, Ctr, Y[train_idx], lam)
    pred = predict_models(m, Xte, Cte)
    Yte = Y[test_idx]
    return {"r2_cov": r2_mean(Yte, pred["cov"]), "r2_rank1": r2_mean(Yte, pred["rank1"]),
            "r2_signals": r2_mean(Yte, pred["signals"]), "n_features": Xtr.shape[1]}


def episode_folds(episodes, n_folds=None):
    """seed 内のエピソード番号で分割する(fold = episode % n_folds)。各 fold を評価側に 1 回ずつ。
    n_folds=2 なら偶奇分割の 2 通り(偶→奇、奇→偶)と同じ。行が少ない fold があれば None。"""
    n_folds = n_folds or N_FOLDS
    fold_id = episodes % n_folds
    splits = []
    for f in range(n_folds):
        te = np.where(fold_id == f)[0]
        tr = np.where(fold_id != f)[0]
        if len(te) < MIN_ROWS_PER_FOLD or len(tr) < MIN_ROWS_PER_FOLD:
            return None
        splits.append((tr, te))
    return splits


def evaluate_seed(texts, C, Y, episodes, lam=RIDGE_LAMBDA, n_folds=None):
    """エピソード分割の各 fold を評価側にした結果の平均。行が少なすぎる場合は nan と理由。"""
    n_folds = n_folds or N_FOLDS
    out = {"n_rows": int(len(texts)), "lambda": lam, "n_folds": n_folds, "reason": None}
    splits = episode_folds(episodes, n_folds)
    if splits is None:
        out.update(r2_cov=float("nan"), r2_rank1=float("nan"), r2_signals=float("nan"),
                   leak=float("nan"), granularity=float("nan"),
                   reason=f"too few rows per fold (n_rows={len(texts)}, n_folds={n_folds}, min {MIN_ROWS_PER_FOLD})")
        return out
    parts = [evaluate_split(texts, C, Y, tr, te, lam) for tr, te in splits]
    for key in ("r2_cov", "r2_rank1", "r2_signals"):
        out[key] = float(np.mean([p[key] for p in parts]))
    out["leak"] = out["r2_signals"] - out["r2_cov"]
    out["granularity"] = out["r2_signals"] - out["r2_rank1"]
    out["n_features"] = tuple(p["n_features"] for p in parts)
    return out


def evaluate_sum_reference(texts, C, y, episodes, lam=RIDGE_LAMBDA, n_folds=None):
    """参考値: 単一の目的変数 y(z の和や逸脱の和)を M_signals 相当の回帰で当てた holdout R²。"""
    splits = episode_folds(episodes, n_folds)
    if splits is None:
        return float("nan")
    Y = y[:, None]
    vals = []
    for tr, te in splits:
        vec = TfidfVectorizer(**TFIDF_KW)
        Xtr = vec.fit_transform([texts[i] for i in tr]).toarray()
        Xte = vec.transform([texts[i] for i in te]).toarray()
        mu, sd = _standardize_fit(C[tr])
        m = fit_models(Xtr, (C[tr] - mu) / sd, Y[tr], lam)
        vals.append(r2_mean(Y[te], predict_models(m, Xte, (C[te] - mu) / sd)["signals"]))
    return float(np.mean(vals))


def analyze_records(records, injected=None, lam=RIDGE_LAMBDA, n_folds=None):
    texts, C, Z, D, eps = rows_from_records(records, injected)
    primary = evaluate_seed(texts, C, Z, eps, lam, n_folds)
    dev_ref = evaluate_seed(texts, C, D, eps, lam, n_folds)
    sums = {"z_sum": evaluate_sum_reference(texts, C, Z.sum(axis=1), eps, lam, n_folds),
            "dev_sum": evaluate_sum_reference(texts, C, D.sum(axis=1), eps, lam, n_folds)}
    n = len(records)
    sents = [re.sub(r"\s+", " ", t.lower()).strip() for t in texts]
    cnt = Counter(sents)
    bigrams = [tuple(w) for s in sents for w in zip(s.split(), s.split()[1:])]
    sub = {
        "duplicate_sentence_rate": sum(1 for s in sents if cnt[s] > 1) / n if n else float("nan"),
        "distinct2": len(set(bigrams)) / len(bigrams) if bigrams else float("nan"),
        "mean_words": float(np.mean([len(re.findall(r"\w+", t)) for t in texts])) if n else float("nan"),
        "correct_rate": float(np.mean([r.correct for r in records])) if n else float("nan"),
        "format_fail_rate": float(np.mean([E.extract_answer(r.text) is None for r in records])) if n else float("nan"),
        "emotion_usage_rate": float(np.mean([r.has_emotion for r in records])) if n else float("nan"),
        "mean_deviation": float(np.mean([r.deviation for r in records])) if n else float("nan"),
    }
    return {"primary": primary, "reference_deviation_targets": dev_ref, "reference_sum": sums,
            "sub_metrics": sub}


def analyze_files(paths, lam=RIDGE_LAMBDA, n_folds=None):
    n_folds = n_folds or N_FOLDS
    per_seed = []
    for p in sorted(paths):
        payload = read_seed_file(p)
        inj = payload.get("injected_z")
        res = analyze_records(payload["records"], injected=np.array(inj) if inj else None, lam=lam, n_folds=n_folds)
        res["seed"] = payload.get("seed")
        res["condition"] = payload.get("condition")
        res["file"] = os.path.basename(p)
        per_seed.append(res)
    def summ(key):
        v = [r["primary"][key] for r in per_seed if not math.isnan(r["primary"][key])]
        return {"n": len(v), "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None}
    return {"per_seed": per_seed, "summary": {k: summ(k) for k in ("leak", "granularity", "r2_cov", "r2_rank1", "r2_signals")},
            "settings": {"lambda": lam, "n_folds": n_folds,
                         "tfidf": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TFIDF_KW.items()}}}


def print_table(result):
    print(f"{'seed':>4} {'rows':>4} {'R2cov':>7} {'R2r1':>7} {'R2sig':>7} {'leak':>7} {'gran':>7} {'dup':>5} {'d2':>5}  note")
    for r in result["per_seed"]:
        p, s = r["primary"], r["sub_metrics"]
        f = lambda x: "   nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:7.3f}"
        print(f"{str(r['seed']):>4} {p['n_rows']:>4} {f(p['r2_cov'])} {f(p['r2_rank1'])} {f(p['r2_signals'])} "
              f"{f(p['leak'])} {f(p['granularity'])} {s['duplicate_sentence_rate']:5.2f} {s['distinct2']:5.2f}  {p['reason'] or ''}")
    sm = result["summary"]
    print(f"\n滲み(median)={sm['leak']['median']} 粒度(median)={sm['granularity']['median']} n={sm['leak']['n']} "
          f"λ={result['settings']['lambda']} folds={result['settings']['n_folds']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pattern", default="*_seed*.json")
    ap.add_argument("--lam", type=float, default=RIDGE_LAMBDA)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print("該当ファイルなし"); return 1
    res = analyze_files(paths, lam=args.lam, n_folds=args.folds)
    print_table(res)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("保存:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
