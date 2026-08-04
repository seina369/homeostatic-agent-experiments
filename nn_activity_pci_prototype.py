"""
感情AIプロジェクト フェーズ15 プロトタイプ(要件5・NN時代 最終プローブ):
実在するNNの活動パターンにPCIを適用
================================================================

目的: フェーズ13・14で重み構造の代数的連結度からは学習ダイナミクスとの
明確な対応関係が得られなかったため、動的な指標(PCI、摂動複雑性指標)を
実在の学習済みネットワークに初めて適用する。要件5のNN時代探索における
最後の一手。

実施内容: フェーズ14と同じ既存pickle(要件7 PartA/B/C、各n=3、
seed0/11/22)を再利用し、新規学習は行わない。各ネットワークについて:
  1. 学習済み方策(eps=0.1のほぼ貪欲方策)で環境を実際にT=60ステップ
     ロールアウトし、代表的な入力状態系列(9次元、環境から得られる実際の
     状態)を記録する(=「環境から得られる代表的な入力系列」)。
  2. この入力系列を素通しした基準の隠れ層活動(h1:32ユニット、
     h2:32ユニット)の軌跡(T×64)を記録する。
  3. h1のユニットのうち8個(ランダムに選んだ1グループ)を全時刻で
     0にクランプする「短い擾乱」を与え、同じ入力系列を再度通した際の
     活動(擾乱後のh1・h2)を記録する。
  4. 基準と擾乱後の差分(|Δh1|, |Δh2|)を、各ユニットについて0より大きいか
     どうかで2値化し、ユニット×時刻のビット列(既存PCI実装
     `pci_score_for_perturbation`と全く同じチャンネル優先の並び順)に
     直列化し、既存のLempel-Ziv圧縮ベースのPCI実装
     (`iit_pci_scale_prototype.py`の`lz_complexity`と正規化式
     `c*log2(L)/L`)をそのまま再利用して複雑性を算出する。
  5. 異なる8ユニット擾乱グループを5回試行し、その平均を1系統のPCI値とする。

【設計上の重要な注記】既存のPCI実装はブール的な力学系(各ノードが確率的に
状態遷移する再帰的なネットワーク)を対象に設計されていた。要件7の
NN方策ネットワークは層間に再帰(リカレント結合)を持たないフィード
フォワードMLPであり、擾乱がタイムステップをまたいで自己伝播すること
はない。そこで本プロトタイプでは、「時間軸」を(a)環境ステップの入力系列
の多様性、(b)ネットワーク内の層(h1→h2)への擾乱の伝播、の2つを組み合わせ、
「特定の環境入力レパートリー全体に対して、局所的なユニット群への擾乱が
どれだけ複雑で予測しづらい応答パターンを引き起こすか」を測る指標として
再解釈している。これは元のTMS-PCI(1箇所を刺激し他の脳領域への伝播を
みる)の精神を踏襲しつつ、フィードフォワード構造向けに適応した設計判断
であり、時間的な自己組織化力学そのものを捉えているわけではない点に
注意が必要。

対照: 同一アーキテクチャ(9→32→32→5)のランダム初期化(未学習)
ネットワーク5シードで同じ手続きを行い、学習がPCIをどちらの方向に
動かすかの基準を作る。

判定基準: PartA/B/C間で明確で再現性のある差(n=3で一貫した傾向)が
見られれば、要件5のNN時代における最初の意味のある手がかりとして記録し、
さらなる展開を検討する。差が見られなければ、要件5のNN時代探索を
「重み構造・活動パターンいずれの指標でも、実在するネットワークから
統合度の意味のある違いを検出できなかった」という結論で総括し区切る。
"""

import pickle
import json
import numpy as np
from scipy import stats

import __main__


class _Blank:
    pass


for _name in ["DQNAgent", "MLPParams", "ReplayBuffer", "AdamState"]:
    setattr(__main__, _name, type(_name, (_Blank,), {}))

from homeostasis_prototype import HomeostasisEnv, ACTIONS
from homeostasis_nn_prototype import encode_state, forward as nn_forward, MLPParams


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# 既存PCI実装と完全同一のLZ圧縮・正規化
# ============================================================

def lz_complexity(s):
    n = len(s)
    i, k, l = 0, 1, 1
    c = 1
    k_max = 1
    while True:
        if s[i + k - 1] != s[l + k - 1]:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
        else:
            k += 1
            if l + k > n:
                c += 1
                break
    return c


def pci_from_bitmatrix(bitmatrix):
    """bitmatrix: (n_channels, T) の0/1配列。チャンネル優先で直列化しLZ圧縮。"""
    bits = []
    n, T = bitmatrix.shape
    for i in range(n):
        for t in range(T):
            bits.append(str(int(bitmatrix[i, t])))
    bitstring = "".join(bits)
    c = lz_complexity(bitstring)
    L = len(bitstring)
    return c * np.log2(L) / L if L > 1 else 0.0


# ============================================================
# 環境ロールアウトによる代表的入力系列の取得
# ============================================================

def rollout_states(params, T=60, eps=0.1, seed=0):
    rng = np.random.RandomState(seed)
    env = HomeostasisEnv(rng)
    state = env.reset()
    xs = []
    for _ in range(T):
        x = encode_state(state)
        xs.append(x)
        if rng.rand() < eps:
            action = ACTIONS[rng.randint(len(ACTIONS))]
        else:
            q, _ = nn_forward(params, x[None, :])
            action = ACTIONS[int(np.argmax(q[0]))]
        state, _, done, _ = env.step(action)
        if done:
            state = env.reset()
    return np.array(xs)  # (T, 9)


def activity_pci(params, T=60, n_perturb_trials=5, perturb_size=8, seed=0):
    X = rollout_states(params, T=T, eps=0.1, seed=seed)  # (T, 9)
    z1 = X @ params.W1 + params.b1
    h1_base = np.maximum(0.0, z1)
    z2 = h1_base @ params.W2 + params.b2
    h2_base = np.maximum(0.0, z2)

    rng = np.random.RandomState(seed + 5000)
    scores = []
    for _ in range(n_perturb_trials):
        idx = rng.choice(h1_base.shape[1], size=perturb_size, replace=False)
        h1_pert = h1_base.copy()
        h1_pert[:, idx] = 0.0
        z2_pert = h1_pert @ params.W2 + params.b2
        h2_pert = np.maximum(0.0, z2_pert)

        diff_h1 = np.abs(h1_pert - h1_base) > 0.0   # (T, 32)
        diff_h2 = np.abs(h2_pert - h2_base) > 0.0   # (T, 32)
        bitmatrix = np.concatenate([diff_h1.T, diff_h2.T], axis=0)  # (64, T)
        scores.append(pci_from_bitmatrix(bitmatrix))
    return float(np.mean(scores)), float(np.std(scores))


def random_init_pci(n_seeds=5, T=60):
    scores = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(2000 + seed)
        params = MLPParams(rng=rng)
        m, _ = activity_pci(params, T=T, seed=seed)
        scores.append(m)
    return scores


SEEDS = [0, 11, 22]


def main():
    per_part = {}
    for part in ["A", "B", "C"]:
        vals = []
        for s in SEEDS:
            st = load_pickle(f"nn_part{part}_state_seed{s}.pkl")
            agent = st["agent"]
            m, sd = activity_pci(agent.params, seed=s)
            vals.append(m)
            print(f"Part{part} seed{s}: PCI={m:.4f}(擾乱試行内SD={sd:.4f})")
        per_part[part] = vals
        print(f"Part{part}: 平均={np.mean(vals):.4f}±{np.std(vals):.4f}")

    baseline = random_init_pci(n_seeds=5)
    baseline_mean, baseline_std = float(np.mean(baseline)), float(np.std(baseline))
    print(f"ランダム初期化ベースライン(n=5): {baseline_mean:.4f}±{baseline_std:.4f}")

    results = {"seeds": SEEDS, "per_part": per_part,
               "baseline_mean": baseline_mean, "baseline_std": baseline_std,
               "baseline_samples": baseline, "ttests": {}}

    for part in ["A", "B", "C"]:
        t, p = stats.ttest_1samp(per_part[part], baseline_mean)
        results["ttests"][f"{part}_vs_baseline"] = {"t": float(t), "p": float(p)}
        print(f"Part{part} vs ベースライン: t={t:.4f}, p={p:.4f}")

    for pair in [("A", "B"), ("A", "C"), ("B", "C")]:
        t, p = stats.ttest_ind(per_part[pair[0]], per_part[pair[1]])
        results["ttests"][f"{pair[0]}_vs_{pair[1]}"] = {"t": float(t), "p": float(p)}
        print(f"Part{pair[0]} vs Part{pair[1]}: t={t:.4f}, p={p:.4f}")

    with open("nn_activity_pci_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n保存: nn_activity_pci_results.json")


if __name__ == "__main__":
    main()
