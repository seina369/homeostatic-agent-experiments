"""B0 をエピソードごとに振る案の偽データ検証(検討・不採用。環境のコードは変更しない)。

事前登録(第一段階)追記欄「2026-09-13 B0 をエピソードごとに振る案(検討・不採用)」の
表と変種を再現する。GPU 不要。

案: 粒度の天井(約 0.09)が環境側の測定限界(budget と error の相関、budget が課題番号で
ほぼ決まること)によるので、B0 をエピソードごとに一様に [240, 440] から引き(正規化は固定
定数 440 で割る。逸脱の閾値はそのエピソードの B0 の 25% だが z には無関係)、budget を
課題番号から切り離す。採用の目安は粒度 0.2 以上。結果は 0.097 → 0.101 で不採用。

方法: v1 C0 の実データ(data/c0_v1、15 seed)から seed ごとにステップ単位の動態
(n_tokens・正誤・unc_after)を借り、30 エピソードを模擬する(実エピソードの並びを再利用し、
足りない分は同じ seed のステップを復元抽出。終了条件は実環境と同じ t ≥ N_TASKS または
budget ≤ 0)。現行(B0=340 固定、/340)と案を同じ機構で作り、signals 型偽文
(leak_probe_synth.make_records_from_z)を付けて比べる:
  - budget–error 相関、t だけで z を当てた R²、z_b の標準偏差
  - 神託特徴(3 語 × 5 段階 one-hot、罰則 ≈ 0)での滲み・粒度の天井
  - TF-IDF の読み取り器(leak_probe: λ=4、5 分割、行の並べ替え帰無 100 回)での滲み・粒度(補正後)

実行:
  python3 b0_variation_check.py                      # 現行 と 案 U[240,440]/440 の比較表
  python3 b0_variation_check.py --lo 140 --hi 540    # 範囲を広げた変種
  python3 b0_variation_check.py --norm own           # 正規化をエピソード自身の B0_e にする変種
"""

import argparse
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emotion_grounding_env as E  # noqa: E402
import leak_probe as P  # noqa: E402
import leak_probe_synth as S  # noqa: E402
from seed_records import read_seed_file  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "c0_v1")
N_EP = 30


def load_dynamics(data_dir=DATA_DIR):
    """seed → エピソードのリスト。エピソード = [(n_tokens, correct, unc_after), ...]。"""
    out = {}
    for p in sorted(glob.glob(os.path.join(data_dir, "c0_seed*.json"))):
        pay = read_seed_file(p)
        eps = {}
        for r in pay["records"]:
            eps.setdefault(r.episode, []).append((r.n_tokens, r.correct, r.unc_after))
        out[pay["seed"]] = [eps[e] for e in sorted(eps)]
    return out


def simulate(episodes, rng, lo=None, hi=None, norm="max"):
    """z の列 (n, 3)、エピソード番号、課題番号 t を返す。
    lo/hi が None なら現行(B0=E.B0 固定、z_b = budget/E.B0)。
    そうでなければ B0_e ~ U[lo, hi]、z_b = budget/hi(norm="max")または budget/B0_e(norm="own")。"""
    pool = [s for ep in episodes for s in ep]
    Z, EP, T = [], [], []
    for e in range(N_EP):
        base = list(rng.choice(episodes))
        if lo is None:
            b0, denom = float(E.B0), float(E.B0)
        else:
            b0 = rng.uniform(lo, hi)
            denom = b0 if norm == "own" else float(hi)
        budget, err, unc, t = b0, 0.0, E.U_OPT, 0
        while t < E.N_TASKS and budget > 0:
            n_tok, correct, unc_after = base[t] if t < len(base) else rng.choice(pool)
            Z.append([budget / denom, err / E.E_MAX, unc / E.U_MAX])
            EP.append(e)
            T.append(t)
            budget = max(budget - n_tok, 0.0)
            if not correct:
                err += 1.0
            unc = unc_after
            t += 1
    return np.array(Z), np.array(EP), np.array(T)


def r2_from_t(Z, T):
    """課題番号 t の平均だけで z の各成分を当てたときの R²(エピソード間の同形性の度合い)。"""
    out = []
    for k in range(Z.shape[1]):
        pred = np.array([Z[T == t, k].mean() for t in T])
        ss_tot = ((Z[:, k] - Z[:, k].mean()) ** 2).sum()
        out.append(1 - ((Z[:, k] - pred) ** 2).sum() / ss_tot if ss_tot > 0 else float("nan"))
    return out


def oracle_features(texts):
    """signals 型偽文の目印語(3 語 × 5 段階)を完全に読む one-hot(検出力の天井用)。"""
    X = np.zeros((len(texts), 15))
    for i, t in enumerate(texts):
        ws = set(t.lower().split())
        for j, lst in enumerate((S.GRADED_B, S.GRADED_E, S.GRADED_U)):
            for k, w in enumerate(lst):
                if w in ws:
                    X[i, 5 * j + k] = 1.0
    return X


def oracle_scores(recs, n_folds=5):
    texts, C, Z, D, eps = P.rows_from_records(recs)
    X = oracle_features(texts)
    r2 = {"cov": [], "rank1": [], "signals": []}
    for tr, te in P.episode_folds(eps, n_folds):
        mu, sd = P._standardize_fit(C[tr])
        m = P.fit_models(X[tr], (C[tr] - mu) / sd, Z[tr], 1e-6)
        pred = P.predict_models(m, X[te], (C[te] - mu) / sd)
        for k in r2:
            r2[k].append(P.r2_mean(Z[te], pred[k]))
    r2 = {k: float(np.mean(v)) for k, v in r2.items()}
    return r2["signals"] - r2["cov"], r2["signals"] - r2["rank1"]


def evaluate(dyn, lo=None, hi=None, norm="max", lam=4.0, n_perm=100):
    rows = []
    for sd in sorted(dyn):
        rng = random.Random(1000 + sd)
        Z, EP, T = simulate(dyn[sd], rng, lo, hi, norm)
        recs = S.make_records_from_z(Z, EP, T, kind="signals", seed=sd)
        o_leak, o_gran = oracle_scores(recs)
        pr = P.analyze_records(recs, lam=lam, n_perm=n_perm)["primary"]
        corr = np.corrcoef(Z.T)
        rows.append(dict(rows=len(Z), corr_be=corr[0, 1], corr_bu=corr[0, 2], corr_eu=corr[1, 2],
                         r2t_b=r2_from_t(Z, T)[0], r2t_e=r2_from_t(Z, T)[1], r2t_u=r2_from_t(Z, T)[2],
                         zb_sd=float(Z[:, 0].std()), o_leak=o_leak, o_gran=o_gran,
                         leak_raw=pr["leak"], gran_raw=pr["granularity"],
                         leak=pr["adjusted"]["leak"], gran=pr["adjusted"]["granularity"]))
    return rows


ROWS = (("行数/seed", "rows"), ("budget–error 相関", "corr_be"), ("budget–unc 相関", "corr_bu"),
        ("error–unc 相関", "corr_eu"), ("R²(t) budget", "r2t_b"), ("R²(t) error", "r2t_e"),
        ("R²(t) unc", "r2t_u"), ("z_b の SD", "zb_sd"), ("神託 滲み(天井)", "o_leak"),
        ("神託 粒度(天井)", "o_gran"), ("TF-IDF 滲み raw", "leak_raw"), ("TF-IDF 滲み adjusted", "leak"),
        ("TF-IDF 粒度 raw", "gran_raw"), ("TF-IDF 粒度 adjusted", "gran"))


def _agg(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return f"{v.mean():+.3f} [{v.min():+.3f}, {v.max():+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=240.0)
    ap.add_argument("--hi", type=float, default=440.0)
    ap.add_argument("--norm", choices=("max", "own"), default="max")
    ap.add_argument("--lam", type=float, default=P.RIDGE_LAMBDA)
    ap.add_argument("--perm", type=int, default=100)
    args = ap.parse_args()
    dyn = load_dynamics()
    if not dyn:
        print("data/c0_v1 の seed ファイルが見つからない"); return 1
    fixed = evaluate(dyn, None, None, "max", args.lam, args.perm)
    varied = evaluate(dyn, args.lo, args.hi, args.norm, args.lam, args.perm)
    label = f"案 B0~U[{args.lo:.0f},{args.hi:.0f}] /{'B0_e' if args.norm == 'own' else f'{args.hi:.0f}'}"
    print(f"{'指標(15 seed 平均 [最小, 最大])':30s} {'現行 B0=' + str(E.B0) + ' 固定':30s} {label:30s}")
    for name, key in ROWS:
        print(f"{name:30s} {_agg(fixed, key):30s} {_agg(varied, key):30s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
