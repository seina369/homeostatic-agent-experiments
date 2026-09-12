"""粒度スコア(事前登録5章・主指標)と副指標の算出。seed を分析単位とする。

主指標: 感情語カテゴリ(目的変数)を内部信号で予測する多項ロジスティック回帰を
2つ立て、McFadden 擬似R² の差を取る。
  - M_valence: 予測子は dev の合計のみ(1列)
  - M_signals: 予測子は budget / error / uncertainty の3信号を別々に(3列)
  粒度スコア = R²(M_signals) − R²(M_valence)

実装上の解釈(事前登録の文面をコードに落とすときに決めた点。データを見る前に固定):
  1. 予測子は「その応答を生成したときにモデルが見ていた信号」= StepRecord の
     *_before(プロンプトに書かれた b, e, u)から作る。応答後の値(*_after)は
     その応答自身の結果なので使わない。
  2. 3信号は生の値ではなく、環境の逸脱関数を通した値
     f_budget(budget_before), f_error(error_before), f_unc(unc_before)を使う。
     こうすると M_valence の予測子 dev = f_budget + f_error + f_unc は
     M_signals の予測子の和そのものになり、2モデルは入れ子(M_valence は
     3係数が等しいという制約付きの M_signals)になる。したがって粒度スコアは
     定義上 0 以上で、「和(価値軸)を超えて3信号を区別しているか」だけを測る。
     生の値を使うと、逸脱関数の非線形性の分だけスコアが動いてしまい、粒度の
     問いと混ざる。
  3. 目的変数は StepRecord.emotion_dominant(辞書4カテゴリの最多。同数は辞書順)。
     主分析は感情語を含む応答(has_emotion=True)だけを対象に4カテゴリで行う
     (「どの語を使い分けるか」の問いなので)。「感情語を使うかどうか」は副指標の
     使用率で別に見る。参考値として "none" を5番目のカテゴリに含めた全応答版も
     出す(granularity_with_none)。
  4. 退化した場合(感情語つき応答が min_rows 未満、または出現カテゴリが1つ)は
     スコアを NaN とし、理由を残す。分岐A(語の消失)の検出は使用率で行う。
  5. 回帰は最尤(Newton法)。数値安定化のため係数にごく小さい L2(ridge=1e-6、
     切片を除く)を掛けるが、R² の計算には罰則なしの対数尤度を使う。

副指標(seedごと): 感情語の使用率、正答率、書式不履行率、平均逸脱、平均出力長、
エピソードごとの平均出力長とその傾き(予算節約圧力の確認)、感情カテゴリの
分布、budget 閾値を初めて割った課題番号の分布(B0 の妥当性の記述統計)。

使い方:
  python3 analyze_granularity.py --dir /path/to/c0 --pattern 'c0_seed*.json' --out c0_analysis.json
  (本物のデータに対する初回実行は、事前登録の手順に従い C0 完了後に行う)
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emotion_grounding_env as E  # noqa: E402
from seed_records import read_seed_file  # noqa: E402

EMOTION_CATEGORIES = sorted(E.EMOTION_LEXICON)   # ['anxiety', 'fatigue', 'irritation', 'positive']
NONE_LABEL = "none"
MIN_ROWS = 20          # これ未満なら回帰を立てない(NaN)
RIDGE = 1e-6


# ------------------------------------------------------------
# 多項ロジスティック回帰(最尤・Newton法)と McFadden 擬似R²
# ------------------------------------------------------------
def _standardize(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd_safe
    Z[:, sd == 0] = 0.0           # 定数列は情報なし(切片に吸収)
    return Z


def _log_softmax_with_ref(Zlogit):
    """Zlogit: (n, K-1)。参照クラス(最後)のロジットは0。(n, K) の log 確率を返す。"""
    n = Zlogit.shape[0]
    full = np.concatenate([Zlogit, np.zeros((n, 1))], axis=1)
    m = full.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(full - m).sum(axis=1, keepdims=True))
    return full - lse


def fit_multinomial(X, y, n_classes, ridge=RIDGE, max_iter=200, tol=1e-10):
    """y は 0..n_classes-1 の整数。最後のクラスを参照にした多項ロジット。

    戻り値: dict(loglik=罰則なし対数尤度, converged, n_iter, coef=(p+1, K-1))
    """
    Xs = _standardize(X)
    n, p = Xs.shape
    Xd = np.concatenate([np.ones((n, 1)), Xs], axis=1)          # 切片つき
    K = int(n_classes)
    y = np.asarray(y, dtype=int)
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0
    d = p + 1
    B = np.zeros((d, K - 1))
    pen_mask = np.ones((d, K - 1))
    pen_mask[0, :] = 0.0                                           # 切片は罰則なし

    def objective(Bm):
        logp = _log_softmax_with_ref(Xd @ Bm)
        ll = float(logp[np.arange(n), y].sum())
        return -ll + 0.5 * ridge * float((pen_mask * Bm ** 2).sum()), ll

    f, ll = objective(B)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        logp = _log_softmax_with_ref(Xd @ B)
        P = np.exp(logp)                                          # (n, K)
        Pk = P[:, :K - 1]
        grad = Xd.T @ (Pk - Y[:, :K - 1]) + ridge * pen_mask * B   # (d, K-1)
        # Hessian of -loglik: sum_i x_i x_i^T (x) (diag(p_i) - p_i p_i^T)
        W = np.einsum('ik,kl->ikl', Pk, np.eye(K - 1)) - np.einsum('ik,il->ikl', Pk, Pk)
        H = np.einsum('ij,ikl,im->jkml', Xd, W, Xd).reshape(d * (K - 1), d * (K - 1))
        H = H + ridge * np.diag(pen_mask.reshape(-1))
        g = grad.reshape(-1)
        gnorm = float(np.abs(g).max())
        if gnorm < 1e-8:
            converged = True
            break
        try:
            step = -np.linalg.solve(H + 1e-12 * np.eye(H.shape[0]), g)
        except np.linalg.LinAlgError:
            step = -g
        gdot = float(g @ step)
        if gdot >= 0:                 # 降下方向でなければ最急降下に切り替える
            step = -g
            gdot = float(g @ step)
        step = step.reshape(d, K - 1)
        # backtracking line search(Armijo 条件)
        t = 1.0
        while True:
            Bn = B + t * step
            fn, lln = objective(Bn)
            if fn <= f + 1e-4 * t * gdot:
                break
            t *= 0.5
            if t < 1e-8:
                Bn, fn, lln = B, f, ll
                break
        if abs(f - fn) < tol * max(1.0, abs(f)):
            B, f, ll = Bn, fn, lln
            converged = True
            break
        B, f, ll = Bn, fn, lln
    return {"loglik": ll, "converged": converged, "n_iter": it, "coef": B}


def null_loglik(y, n_classes):
    y = np.asarray(y, dtype=int)
    n = len(y)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    nz = counts[counts > 0]
    return float((nz * np.log(nz / n)).sum())


def mcfadden_r2(ll_model, ll_null):
    if ll_null == 0.0:
        return float("nan")
    return 1.0 - ll_model / ll_null


# ------------------------------------------------------------
# StepRecord → 予測子・目的変数
# ------------------------------------------------------------
def signal_components_before(records):
    """各応答の生成時にモデルが見ていた信号を逸脱関数に通した3列 (n, 3) と、その和 (n,)。"""
    fb = np.array([E.budget_deviation(r.budget_before) for r in records], dtype=float)
    fe = np.array([E.error_deviation(r.error_before) for r in records], dtype=float)
    fu = np.array([E.uncertainty_deviation(r.unc_before) for r in records], dtype=float)
    X3 = np.stack([fb, fe, fu], axis=1)
    return X3, X3.sum(axis=1)


def encode_labels(records, include_none):
    labels_all = EMOTION_CATEGORIES + ([NONE_LABEL] if include_none else [])
    present = [c for c in labels_all if any(r.emotion_dominant == c for r in records)]
    idx = {c: i for i, c in enumerate(present)}
    y = np.array([idx[r.emotion_dominant] for r in records], dtype=int)
    return y, present


def granularity_score(records, include_none=False, min_rows=MIN_ROWS):
    """1 seed 分の StepRecord から粒度スコアを出す。退化時は score=NaN と reason。"""
    rows = list(records) if include_none else [r for r in records if r.has_emotion]
    out = {
        "include_none": include_none,
        "n_rows": len(rows),
        "class_counts": dict(Counter(r.emotion_dominant for r in rows)),
        "score": float("nan"), "r2_signals": float("nan"), "r2_valence": float("nan"),
        "r2_single": {}, "loglik_null": float("nan"), "reason": None,
    }
    if len(rows) < min_rows:
        out["reason"] = f"too few rows ({len(rows)} < {min_rows})"
        return out
    y, present = encode_labels(rows, include_none)
    if len(present) < 2:
        out["reason"] = f"fewer than 2 categories present ({present})"
        return out
    K = len(present)
    X3, dev = signal_components_before(rows)
    ll0 = null_loglik(y, K)
    fit_v = fit_multinomial(dev, y, K)
    fit_s = fit_multinomial(X3, y, K)
    r2_v = mcfadden_r2(fit_v["loglik"], ll0)
    r2_s = mcfadden_r2(fit_s["loglik"], ll0)
    out.update({
        "classes": present,
        "loglik_null": ll0,
        "loglik_valence": fit_v["loglik"], "loglik_signals": fit_s["loglik"],
        "r2_valence": r2_v, "r2_signals": r2_s,
        "score": r2_s - r2_v,
        "converged": bool(fit_v["converged"] and fit_s["converged"]),
        "r2_single": {
            name: mcfadden_r2(fit_multinomial(X3[:, j], y, K)["loglik"], ll0)
            for j, name in enumerate(["budget", "error", "uncertainty"])
        },
    })
    return out


# ------------------------------------------------------------
# 副指標
# ------------------------------------------------------------
def sub_metrics(records):
    recs = list(records)
    n = len(recs)
    if n == 0:
        return {"n_steps": 0}
    by_ep = {}
    for r in recs:
        by_ep.setdefault(r.episode, []).append(r)
    episodes = sorted(by_ep)
    tokens_by_episode = [float(np.mean([r.n_tokens for r in by_ep[e]])) for e in episodes]
    if len(episodes) >= 2:
        slope = float(np.polyfit(np.arange(len(episodes)), tokens_by_episode, 1)[0])
    else:
        slope = None
    cross_t = []
    for e in episodes:
        t_cross = next((r.t for r in sorted(by_ep[e], key=lambda r: r.t)
                        if r.budget_after < E.BUDGET_LOW_THRESHOLD), None)
        cross_t.append(t_cross)
    crossed = [t for t in cross_t if t is not None]
    return {
        "n_steps": n,
        "n_episodes": len(episodes),
        "mean_steps_per_episode": n / len(episodes),
        "emotion_usage_rate": sum(r.has_emotion for r in recs) / n,
        "correct_rate": sum(r.correct for r in recs) / n,
        "format_fail_rate": sum(E.extract_answer(r.text) is None for r in recs) / n,
        "mean_deviation": float(np.mean([r.deviation for r in recs])),
        "mean_tokens": float(np.mean([r.n_tokens for r in recs])),
        "tokens_by_episode": tokens_by_episode,
        "tokens_slope_per_episode": slope,
        "category_distribution": dict(Counter(r.emotion_dominant for r in recs)),
        "budget_threshold_cross_t": {
            "episodes_crossed": len(crossed),
            "episodes_total": len(episodes),
            "mean_t": float(np.mean(crossed)) if crossed else None,
            "distribution": dict(Counter(crossed)),
        },
        "task_kind_counts": dict(Counter(r.task_kind for r in recs)),
    }


# ------------------------------------------------------------
# seed 単位・複数 seed
# ------------------------------------------------------------
def analyze_records(records, seed=None, condition=None, n_episodes=None):
    return {
        "seed": seed,
        "condition": condition,
        "n_episodes": n_episodes,
        "granularity": granularity_score(records, include_none=False),
        "granularity_with_none": granularity_score(records, include_none=True),
        "sub_metrics": sub_metrics(records),
    }


def analyze_seed_file(path):
    p = read_seed_file(path)
    res = analyze_records(p["records"], seed=p.get("seed"), condition=p.get("condition"),
                          n_episodes=p.get("n_episodes"))
    res["file"] = os.path.basename(path)
    res["complete"] = bool(p.get("complete"))
    return res


def _summ(values):
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=float)
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def analyze_files(paths):
    per_seed = [analyze_seed_file(p) for p in sorted(paths)]
    summary = {
        "n_seeds": len(per_seed),
        "granularity_score": _summ([r["granularity"]["score"] for r in per_seed]),
        "r2_signals": _summ([r["granularity"]["r2_signals"] for r in per_seed]),
        "r2_valence": _summ([r["granularity"]["r2_valence"] for r in per_seed]),
        "granularity_score_with_none": _summ([r["granularity_with_none"]["score"] for r in per_seed]),
        "emotion_usage_rate": _summ([r["sub_metrics"].get("emotion_usage_rate") for r in per_seed]),
        "correct_rate": _summ([r["sub_metrics"].get("correct_rate") for r in per_seed]),
        "mean_deviation": _summ([r["sub_metrics"].get("mean_deviation") for r in per_seed]),
        "mean_tokens": _summ([r["sub_metrics"].get("mean_tokens") for r in per_seed]),
        "seeds_with_nan_score": [r["seed"] for r in per_seed
                                 if math.isnan(r["granularity"]["score"])],
    }
    return {"per_seed": per_seed, "summary": summary}


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "   nan"
    return f"{x:6.{nd}f}"


def print_table(result):
    print(f"{'seed':>4} {'rows':>4} {'score':>6} {'R2sig':>6} {'R2val':>6} "
          f"{'usage':>6} {'corr':>6} {'dev':>6} {'tok':>6}  note")
    for r in result["per_seed"]:
        g, s = r["granularity"], r["sub_metrics"]
        print(f"{str(r['seed']):>4} {g['n_rows']:>4} {_fmt(g['score'])} {_fmt(g['r2_signals'])} "
              f"{_fmt(g['r2_valence'])} {_fmt(s.get('emotion_usage_rate'))} "
              f"{_fmt(s.get('correct_rate'))} {_fmt(s.get('mean_deviation'))} "
              f"{_fmt(s.get('mean_tokens'), 1)}  {g['reason'] or ''}")
    sm = result["summary"]
    gs = sm["granularity_score"]
    if gs.get("n"):
        print(f"\n粒度スコア(主分析、seed単位, n={gs['n']}): 平均{gs['mean']:.4f} "
              f"中央値{gs['median']:.4f} sd{gs['sd']:.4f} [{gs['min']:.4f}, {gs['max']:.4f}]")
    if sm["seeds_with_nan_score"]:
        print(f"スコアNaNのseed: {sm['seeds_with_nan_score']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="seed ファイルのあるフォルダ")
    ap.add_argument("--pattern", default="c0_seed*.json")
    ap.add_argument("--out", default=None, help="結果 JSON の保存先(省略時は保存しない)")
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not paths:
        print(f"該当ファイルなし: {os.path.join(args.dir, args.pattern)}")
        return 1
    result = analyze_files(paths)
    print_table(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1)
        print(f"\n保存: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
