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

並べ替え帰無基準(採用。N_PERM=100、PERM_MODE="rows"): 条件・seed ごとに z の行を行の間で
並べ替えて同じ読み取り器にかけた滲み・粒度の平均を「その条件のゼロ」とし、生の値から引いた
adjusted を主指標にする(raw と null も併記)。理由と検算は追記欄「G3 追加検討」。エピソード塊の
並べ替え("episode")は比較用に残すが、実データでは状態軌跡がエピソード間でほぼ同形なので
本物の滲みまで帰無に入ってしまう(同追記に確認結果)。

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
N_PERM = 100         # 並べ替え帰無基準の回数(0 = 計算しない)。100 回・行の並べ替えを採用(追記欄「G3 追加検討」)。
PERM_MODE = "rows"   # 並べ替えの単位: "rows"(行。採用)か "episode"(エピソードの塊。比較用。実データでは本物の滲みを引いてしまう)。
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


class FoldDesign:
    """1 fold 分の、目的変数 Y に依存しない前計算(TF-IDF、共変量の標準化、残差化、リッジの解)。
    fit_models / predict_models と同じ式だが、Y だけを何度も差し替える(並べ替え帰無基準)ために
    X 側の計算を一度で済ませる。リッジは双対形 (XᵀX+λI)⁻¹Xᵀ = Xᵀ(XXᵀ+λI)⁻¹ で解く(n < p のとき安い)。"""

    def __init__(self, texts, C, train_idx, test_idx, lam):
        vec = TfidfVectorizer(**TFIDF_KW)
        Xtr = vec.fit_transform([texts[i] for i in train_idx]).toarray()
        Xte = vec.transform([texts[i] for i in test_idx]).toarray()
        mu, sd = _standardize_fit(C[train_idx])
        Ctr, Cte = (C[train_idx] - mu) / sd, (C[test_idx] - mu) / sd
        self.A1tr = np.concatenate([Ctr, np.ones((len(train_idx), 1))], axis=1)
        self.A1te = np.concatenate([Cte, np.ones((len(test_idx), 1))], axis=1)
        self.P = np.linalg.pinv(self.A1tr)                        # (q+1, n): 最小二乗の係数 = P @ Y
        Gx = self.P @ Xtr
        self.Xr_tr = Xtr - self.A1tr @ Gx
        self.Xr_te = Xte - self.A1te @ Gx
        n = self.Xr_tr.shape[0]
        self.K = self.Xr_tr.T @ np.linalg.inv(self.Xr_tr @ self.Xr_tr.T + lam * np.eye(n))   # (p, n): W = K @ Yr
        self.n_features = Xtr.shape[1]
        self.train_idx, self.test_idx = train_idx, test_idx

    def fit_predict(self, Ytr):
        Bc = self.P @ Ytr
        Yr = Ytr - self.A1tr @ Bc
        W = self.K @ Yr
        F = self.Xr_tr @ W
        _, _, Vt = np.linalg.svd(F, full_matrices=False)
        v = Vt[0]
        W1 = np.outer(W @ v, v)
        y_cov = self.A1te @ Bc
        return {"cov": y_cov, "rank1": y_cov + self.Xr_te @ W1, "signals": y_cov + self.Xr_te @ W}


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


def _scores_from_designs(designs, Y):
    """各 fold の holdout R²(cov / rank1 / signals)を fold 平均し、滲み・粒度を返す。"""
    acc = {"r2_cov": [], "r2_rank1": [], "r2_signals": []}
    for d in designs:
        pred = d.fit_predict(Y[d.train_idx])
        Yte = Y[d.test_idx]
        acc["r2_cov"].append(r2_mean(Yte, pred["cov"]))
        acc["r2_rank1"].append(r2_mean(Yte, pred["rank1"]))
        acc["r2_signals"].append(r2_mean(Yte, pred["signals"]))
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["leak"] = out["r2_signals"] - out["r2_cov"]
    out["granularity"] = out["r2_signals"] - out["r2_rank1"]
    return out


def evaluate_split(texts, C, Y, train_idx, test_idx, lam):
    d = FoldDesign(texts, C, train_idx, test_idx, lam)
    pred = d.fit_predict(Y[train_idx])
    Yte = Y[test_idx]
    return {"r2_cov": r2_mean(Yte, pred["cov"]), "r2_rank1": r2_mean(Yte, pred["rank1"]),
            "r2_signals": r2_mean(Yte, pred["signals"]), "n_features": d.n_features}


def _permute_rows(Y, episodes, rng, mode):
    """mode="rows": 行を自由に並べ替える(採用。追記欄「G3 追加検討」)。
    mode="episode": エピソードの塊ごと(z の並びを塊の中では保ったまま)、同じ長さのエピソードの
    間で入れ替える(長さが 1 つしかない塊は動かない)。比較検討用に残す。"""
    if mode == "episode":
        ids = np.unique(episodes)
        blocks = [np.where(episodes == e)[0] for e in ids]
        Yp = Y.copy()
        by_len = {}
        for b in blocks:
            by_len.setdefault(len(b), []).append(b)
        for group in by_len.values():
            order = rng.permutation(len(group))
            for dst, src in zip(group, order):
                Yp[dst] = Y[group[src]]
        return Yp
    return Y[rng.permutation(Y.shape[0])]


def permutation_null(designs, Y, n_perm, seed=0, episodes=None, mode="rows"):
    """並べ替え帰無基準: z の行を(seed 内で)並べ替えて同じ読み取り器にかけた滲み・粒度の分布。
    文と z の対応を壊すので「文が何も運ばないときにこの読み取り器が出す値」(過適合による負の
    ずれを含む)の推定になる。戻り値は平均・標準偏差・n_perm・mode。
    mode="rows" は行の自由な並べ替え(z の行が独立なら正確)。mode="episode" はエピソードの塊ごとの
    並べ替え(z がエピソード内で時間相関を持つ実データ向け。塊の長さが揃っている必要がある)。
    注意: どちらも共変量と z の対応を同時に壊す(偽データでは共変量は z と独立なので影響なし。
    実データでは R²(M_cov) が帰無で 0 付近になる分、ずれの推定がわずかに変わりうる)。"""
    rng = np.random.default_rng(seed)
    leaks, grans = [], []
    for _ in range(n_perm):
        Yp = _permute_rows(Y, episodes, rng, mode) if episodes is not None else Y[rng.permutation(Y.shape[0])]
        s = _scores_from_designs(designs, Yp)
        leaks.append(s["leak"])
        grans.append(s["granularity"])
    return {"leak": float(np.mean(leaks)), "granularity": float(np.mean(grans)),
            "leak_sd": float(np.std(leaks)), "granularity_sd": float(np.std(grans)),
            "n_perm": int(n_perm), "mode": mode}


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


def evaluate_seed(texts, C, Y, episodes, lam=RIDGE_LAMBDA, n_folds=None, n_perm=None, perm_mode=None):
    """エピソード分割の各 fold を評価側にした結果の平均。行が少なすぎる場合は nan と理由。
    n_perm > 0 なら並べ替え帰無基準(null)と、それを引いた値(adjusted)も返す。"""
    n_folds = n_folds or N_FOLDS
    n_perm = N_PERM if n_perm is None else n_perm
    perm_mode = perm_mode or PERM_MODE
    out = {"n_rows": int(len(texts)), "lambda": lam, "n_folds": n_folds, "reason": None}
    splits = episode_folds(episodes, n_folds)
    if splits is None:
        out.update(r2_cov=float("nan"), r2_rank1=float("nan"), r2_signals=float("nan"),
                   leak=float("nan"), granularity=float("nan"),
                   reason=f"too few rows per fold (n_rows={len(texts)}, n_folds={n_folds}, min {MIN_ROWS_PER_FOLD})")
        if n_perm:
            out["null"] = None
            out["adjusted"] = {"leak": float("nan"), "granularity": float("nan")}
        return out
    designs = [FoldDesign(texts, C, tr, te, lam) for tr, te in splits]
    out.update(_scores_from_designs(designs, Y))
    out["n_features"] = tuple(d.n_features for d in designs)
    if n_perm:
        null = permutation_null(designs, Y, n_perm, episodes=episodes, mode=perm_mode)
        out["null"] = null
        out["adjusted"] = {"leak": out["leak"] - null["leak"],
                           "granularity": out["granularity"] - null["granularity"]}
    return out


def evaluate_sum_reference(texts, C, y, episodes, lam=RIDGE_LAMBDA, n_folds=None):
    """参考値: 単一の目的変数 y(z の和や逸脱の和)を M_signals 相当の回帰で当てた holdout R²。"""
    splits = episode_folds(episodes, n_folds)
    if splits is None:
        return float("nan")
    Y = y[:, None]
    vals = []
    for tr, te in splits:
        vals.append(evaluate_split(texts, C, Y, tr, te, lam)["r2_signals"])
    return float(np.mean(vals))


def analyze_records(records, injected=None, lam=RIDGE_LAMBDA, n_folds=None, n_perm=None, perm_mode=None):
    texts, C, Z, D, eps = rows_from_records(records, injected)
    primary = evaluate_seed(texts, C, Z, eps, lam, n_folds, n_perm, perm_mode)
    dev_ref = evaluate_seed(texts, C, D, eps, lam, n_folds, n_perm, perm_mode)
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


def analyze_files(paths, lam=RIDGE_LAMBDA, n_folds=None, n_perm=None, perm_mode=None):
    n_folds = n_folds or N_FOLDS
    n_perm = N_PERM if n_perm is None else n_perm
    perm_mode = perm_mode or PERM_MODE
    per_seed = []
    for p in sorted(paths):
        payload = read_seed_file(p)
        inj = payload.get("injected_z")
        res = analyze_records(payload["records"], injected=np.array(inj) if inj else None, lam=lam,
                              n_folds=n_folds, n_perm=n_perm, perm_mode=perm_mode)
        res["seed"] = payload.get("seed")
        res["condition"] = payload.get("condition")
        res["file"] = os.path.basename(p)
        per_seed.append(res)
    def summ(vals):
        v = [x for x in vals if not math.isnan(x)]
        return {"n": len(v), "mean": float(np.mean(v)) if v else None, "median": float(np.median(v)) if v else None}
    summary = {k: summ([r["primary"][k] for r in per_seed]) for k in ("leak", "granularity", "r2_cov", "r2_rank1", "r2_signals")}
    if n_perm:
        for k in ("leak", "granularity"):
            summary[f"adjusted_{k}"] = summ([r["primary"]["adjusted"][k] for r in per_seed])
            summary[f"null_{k}"] = summ([r["primary"]["null"][k] if r["primary"]["null"] else float("nan") for r in per_seed])
    return {"per_seed": per_seed, "summary": summary,
            "settings": {"lambda": lam, "n_folds": n_folds, "n_perm": n_perm, "perm_mode": perm_mode,
                         "tfidf": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TFIDF_KW.items()}}}


def print_table(result):
    with_null = bool(result["settings"].get("n_perm"))
    extra = f" {'nullL':>7} {'nullG':>7} {'adjL':>7} {'adjG':>7}" if with_null else ""
    print(f"{'seed':>4} {'rows':>4} {'R2cov':>7} {'R2r1':>7} {'R2sig':>7} {'leak':>7} {'gran':>7}{extra} {'dup':>5} {'d2':>5}  note")
    f = lambda x: "    nan" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:7.3f}"
    for r in result["per_seed"]:
        p, s = r["primary"], r["sub_metrics"]
        ex = ""
        if with_null:
            nl = p.get("null") or {}
            ex = f" {f(nl.get('leak'))} {f(nl.get('granularity'))} {f(p['adjusted']['leak'])} {f(p['adjusted']['granularity'])}"
        print(f"{str(r['seed']):>4} {p['n_rows']:>4} {f(p['r2_cov'])} {f(p['r2_rank1'])} {f(p['r2_signals'])} "
              f"{f(p['leak'])} {f(p['granularity'])}{ex} {s['duplicate_sentence_rate']:5.2f} {s['distinct2']:5.2f}  {p['reason'] or ''}")
    sm = result["summary"]
    line = (f"\n滲み(median)={sm['leak']['median']} 粒度(median)={sm['granularity']['median']} n={sm['leak']['n']} "
            f"λ={result['settings']['lambda']} folds={result['settings']['n_folds']}")
    if with_null:
        line += (f" perm={result['settings']['n_perm']} 帰無補正後: 滲み(median)={sm['adjusted_leak']['median']} "
                 f"粒度(median)={sm['adjusted_granularity']['median']}")
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pattern", default="*_seed*.json")
    ap.add_argument("--lam", type=float, default=RIDGE_LAMBDA)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--perm", type=int, default=N_PERM, help="並べ替え帰無基準の回数(0 で無効)")
    ap.add_argument("--perm-mode", default=PERM_MODE, choices=("rows", "episode"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print("該当ファイルなし"); return 1
    res = analyze_files(paths, lam=args.lam, n_folds=args.folds, n_perm=args.perm, perm_mode=args.perm_mode)
    print_table(res)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("保存:", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
