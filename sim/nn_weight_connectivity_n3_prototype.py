"""
感情AIプロジェクト フェーズ14 プロトタイプ(要件5・NN時代):
PartCの連結度低下の再現性確認(n=3)
================================================================

目的: フェーズ13でn=1(seed=0)のみ観察された「Part C(grokking探索、
8000ep)の代数的連結度がランダム初期化ベースラインから大幅に低下する」
という結果が、学習量・強度に対する再現性のある傾向か、単なる1系統の
偶然かを確かめる。

実施内容: Part A・B・Cとも既に要件7 NN移行実験(タスク#63/#65/#68付近)で
n=3(seed=0,11,22)またはn=15まで学習済みで、その時間区切り式再開state
pickleが一時ディレクトリに現存していることを確認した(nn_partA/B/C_state_
seed{0,11,22}.pkl)。よって新規学習は不要で、既存のseed=11,22の重みを
追加抽出しseed=0の結果と合わせてn=3で集計する。フェーズ13と全く同じ
algebraic_connectivity()・mlp_layer_graph()・random_init_baseline()を
再利用し、Part A/B/Cそれぞれseed0,11,22の正規化Fiedler値を計算した上で、
(1)各Partの平均がランダム初期化ベースライン(10シード平均)と有意に
異なるか(1標本t検定)、(2)Part C が Part A・Part B と有意に異なるか
(対応なし2標本t検定)を確認する。

要件4側(エルダー・サクセサー)の本来重みでの追検証は、既存pickleが
残っていないため新規学習(時間区切り式で複数チャンクの再学習)が必要と
なりコストが高いため、ユーザーの指示に従い今回は見送り、Part Cの
再現性確認のみに絞る。

判定基準: n=3でも同程度の低下が一貫して見られれば再現性のある傾向として
扱う。ばらつきが大きく1系統目だけの特異な結果だった場合は単発の偶然として
記録し、この方向の深追いを打ち切る。
"""

import pickle
import json
import numpy as np
from scipy import stats

import __main__


class _Blank:
    pass


for _name in [
    "MLPParamsGen", "NNAgentGeneric", "SuccessorNet",
    "NNAgentSplit", "SuccessorNetSplit",
    "NNMoveAgent", "NNGuessAgent",
    "DQNAgent", "MLPParams", "ReplayBuffer", "AdamState",
    "EvalPolicy", "NNMoveEvalPolicy", "NNGuessEvalPolicy",
]:
    setattr(__main__, _name, type(_name, (_Blank,), {}))


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def algebraic_connectivity(W):
    Wz = W.copy()
    np.fill_diagonal(Wz, 0.0)
    Wsym = (Wz + Wz.T) / 2.0
    deg = np.diag(Wsym.sum(axis=1))
    L = deg - Wsym
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


def mlp_layer_graph(weight_matrices, normalize=False):
    mats = [np.abs(np.asarray(w)) for w in weight_matrices]
    if normalize:
        all_vals = np.concatenate([m.flatten() for m in mats])
        scale = all_vals.mean()
        if scale > 0:
            mats = [m / scale for m in mats]
    layer_sizes = [mats[0].shape[0]] + [m.shape[1] for m in mats]
    n_total = sum(layer_sizes)
    W = np.zeros((n_total, n_total))
    offsets = np.cumsum([0] + layer_sizes)
    for li, m in enumerate(mats):
        r0, r1 = offsets[li], offsets[li + 1]
        c0, c1 = offsets[li + 1], offsets[li + 2]
        W[r0:r1, c0:c1] = m
        W[c0:c1, r0:r1] = m.T
    return W, n_total


def random_init_baseline(layer_sizes, n_seeds=10):
    raws, norms = [], []
    for seed in range(n_seeds):
        rng = np.random.RandomState(1000 + seed)
        mats = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            mats.append(rng.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / fan_in))
        W_norm, _ = mlp_layer_graph(mats, normalize=True)
        norms.append(algebraic_connectivity(W_norm))
    return norms


SEEDS = [0, 11, 22]


def fiedler_for_part(part, seed):
    st = load_pickle(f"nn_part{part}_state_seed{seed}.pkl")
    agent = st["agent"]
    mats = [agent.params.W1, agent.params.W2, agent.params.W3]
    _, _ = mlp_layer_graph(mats, normalize=False)
    W_norm, _ = mlp_layer_graph(mats, normalize=True)
    return algebraic_connectivity(W_norm)


def main():
    per_part = {}
    for part in ["A", "B", "C"]:
        vals = [fiedler_for_part(part, s) for s in SEEDS]
        per_part[part] = vals
        print(f"Part{part}: seeds={SEEDS} -> Fiedler(正規化)={[round(v,4) for v in vals]}, "
              f"平均={np.mean(vals):.4f}±{np.std(vals):.4f}")

    baseline = random_init_baseline([9, 32, 32, 5], n_seeds=10)
    baseline_mean, baseline_std = float(np.mean(baseline)), float(np.std(baseline))
    print(f"ランダム初期化ベースライン(9-32-32-5, n=10): {baseline_mean:.4f}±{baseline_std:.4f}")

    results = {"seeds": SEEDS, "per_part": per_part,
               "baseline_mean": baseline_mean, "baseline_std": baseline_std,
               "baseline_samples": baseline, "ttests": {}}

    for part in ["A", "B", "C"]:
        t, p = stats.ttest_1samp(per_part[part], baseline_mean)
        results["ttests"][f"{part}_vs_baseline"] = {"t": float(t), "p": float(p)}
        print(f"Part{part} vs ベースライン: t={t:.4f}, p={p:.4f}")

    for pair in [("C", "A"), ("C", "B")]:
        t, p = stats.ttest_ind(per_part[pair[0]], per_part[pair[1]])
        results["ttests"][f"{pair[0]}_vs_{pair[1]}"] = {"t": float(t), "p": float(p)}
        print(f"Part{pair[0]} vs Part{pair[1]}: t={t:.4f}, p={p:.4f}")

    with open("nn_weight_connectivity_n3_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n保存: nn_weight_connectivity_n3_results.json")


if __name__ == "__main__":
    main()
