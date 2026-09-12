"""
感情AIプロジェクト フェーズ5 プロトタイプ: PCI本実装・GWT由来「到達範囲」指標の検証(要件5)
==========================================================

これまでの代替指標比較(iit_alt_metrics_prototype.py)では、代数的連結度・
クラスター単位シナジーが、厳密Φの山型(結合強度pの増加が、ある閾値を超えると
統合を破壊し始める非単調性、p=0.4でピーク)を再現できないことが分かった。
PCI風の指標は簡易版(単一の摂動条件)だと前向き型/再帰型の区別すらつかず、
32通り平均でようやく弱い方向の一致を示す程度だった。

本プロトタイプは、(1)PCIの本実装(複数ノード・複数初期状態への摂動、
LZ圧縮率、摂動箇所ごとの最大値)、(2)グローバルワークスペース理論(GWT)由来の
「到達範囲」指標(摂動が他のノードの将来状態にどれだけ広く・強く影響するか)
の2つを実装し、同じ2つのベンチマーク(FF/REC区別、p=0〜1.0の連続結合強度
スイープでの山型再現)で検証する。厳密Φの計算(pyphi)は必要ないため
(状態遷移関数さえあれば計算できる)、計算コストは軽い。

**(1) PCI本実装**: 状態sの各ノードi(1つずつ)を反転させ、そこからT=15
ステップ状態遷移をシミュレートし(確率的な遷移は都度サンプリング、n_trials
回試行して平均)、全ノード×全時刻の時系列(チャンネル順に平坦化した0/1系列)を
Lempel-Ziv複雑性で定量化する(c*log2(L)/Lで正規化)。ノードiへの摂動で
得られたスコアのうち、**全ノードにわたる最大値**をその状態sのPCIスコアと
する(どこか一箇所への摂動が引き起こす最大の複雑性、という考え方)。
これを複数の初期状態で計算し平均する。

**(2) GWT由来「到達範囲」指標**: 状態sの各ノードi(1つずつ)を反転させ、
同じ乱数系列(共通乱数、CRN)を使って摂動あり/なしの2つの軌跡をTステップ
並行してシミュレートする(共通乱数を使うことで、軌跡の違いが真に摂動の
効果によるものだと言える)。各時刻でハミング距離(何ノードの状態が
摂動あり/なしで異なるか)/ノード数、をTステップ分合計したものを
「到達範囲スコア」とする。ノードiへの摂動のスコアのうち最大値をその状態の
到達範囲スコアとし(PCIと同じ集約方法)、複数の初期状態で平均する。

使い方:
  python3 iit_pci_gwt_prototype.py run
"""

import sys, json, random, itertools
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# ネットワーク定義(既存実験と完全に同一のロジックを再利用)
# ============================================================

def ff_update_fn(state):
    a, b, c, d = state
    return [float(a), float(b), float(a and b), float(c)]


def rec_update_fn(state):
    a, b, c, d = state
    a_next = float((d and b) or ((not d) and a))
    b_next = float((c and a) or ((not c) and b))
    return [a_next, b_next, float(a and b), float(c)]


def make_p_update_fn(p):
    def fn(state):
        a, b, c, d, e = state
        pB = float(a and c)
        pC = float(a and b)
        pE = float(d)
        pA = (1 - p) * float(b and c) + p * float(d)
        pD = (1 - p) * float(e) + p * float(a)
        return [pA, pB, pC, pD, pE]
    return fn


# 既存実験(iit_phi_degree_correspondence_prototype.py)で計算済みの厳密Φ(再利用)
EXISTING_PHI = {
    0.0: 0.0000, 0.1: 0.2174, 0.2: 0.3714, 0.3: 0.4368, 0.4: 0.4379,
    0.5: 0.3860, 0.6: 0.3772, 0.7: 0.1992, 1.0: 0.0000,
}
P_VALUES = sorted(EXISTING_PHI.keys())

FF_REACHABLE_STATES = [s for s in itertools.product([0, 1], repeat=4) if s[2] == (s[0] and s[1])]
CANDIDATE_STATES_5 = [
    (0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (1, 0, 1, 0, 1), (0, 1, 0, 1, 0), (1, 1, 0, 0, 1),
]


# ============================================================
# 共通ユーティリティ: Lempel-Ziv複雑性・確率的サンプリング
# ============================================================

def lz_complexity(s):
    """Kaspar-Schuster版のLempel-Ziv複雑性(0/1文字列を対象)。"""
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


def sample_with_crn(probs, rand_draws):
    """共通乱数(rand_draws、ノードごとに1つずつの一様乱数)を使って次状態をサンプリングする。"""
    return tuple(1 if rand_draws[i] < probs[i] else 0 for i in range(len(probs)))


# ============================================================
# (1) PCI本実装: 複数ノード・複数初期状態への摂動、LZ圧縮率、摂動箇所ごとの最大値
# ============================================================

def pci_score_for_perturbation(update_fn, n, start_state, perturb_idx, T=15, n_trials=20, rng=None):
    rng = rng or random.Random(0)
    scores = []
    for _ in range(n_trials):
        state = list(start_state)
        state[perturb_idx] = 1 - state[perturb_idx]
        state = tuple(state)
        traj = [state]
        for _ in range(T):
            probs = update_fn(state)
            draws = [rng.random() for _ in range(n)]
            state = sample_with_crn(probs, draws)
            traj.append(state)
        bits = []
        for i in range(n):
            for t in range(len(traj)):
                bits.append(str(traj[t][i]))
        bitstring = "".join(bits)
        c = lz_complexity(bitstring)
        L = len(bitstring)
        norm = c * np.log2(L) / L if L > 1 else 0.0
        scores.append(norm)
    return float(np.mean(scores))


def pci_full(update_fn, n, candidate_states, T=15, n_trials=20, seed=0):
    rng = random.Random(seed)
    state_scores = []
    per_node_scores_all = []
    for start_state in candidate_states:
        node_scores = [pci_score_for_perturbation(update_fn, n, start_state, i, T, n_trials, rng) for i in range(n)]
        state_scores.append(max(node_scores))
        per_node_scores_all.append(node_scores)
    return float(np.mean(state_scores)), state_scores, per_node_scores_all


# ============================================================
# (2) GWT由来「到達範囲」指標: 共通乱数での摂動あり/なし比較
# ============================================================

def reach_score_for_perturbation(update_fn, n, start_state, perturb_idx, T=15, n_trials=20, rng=None):
    rng = rng or random.Random(0)
    scores = []
    for _ in range(n_trials):
        base_state = tuple(start_state)
        pert_state = list(start_state)
        pert_state[perturb_idx] = 1 - pert_state[perturb_idx]
        pert_state = tuple(pert_state)

        total_frac_diff = 0.0
        for _ in range(T):
            draws = [rng.random() for _ in range(n)]  # 共通乱数(baseline/perturbedで共有)
            base_probs = update_fn(base_state)
            pert_probs = update_fn(pert_state)
            base_state = sample_with_crn(base_probs, draws)
            pert_state = sample_with_crn(pert_probs, draws)
            hamming = sum(1 for i in range(n) if base_state[i] != pert_state[i])
            total_frac_diff += hamming / n
        scores.append(total_frac_diff)
    return float(np.mean(scores))


def reach_full(update_fn, n, candidate_states, T=15, n_trials=20, seed=0):
    rng = random.Random(seed)
    state_scores = []
    per_node_scores_all = []
    for start_state in candidate_states:
        node_scores = [reach_score_for_perturbation(update_fn, n, start_state, i, T, n_trials, rng) for i in range(n)]
        state_scores.append(max(node_scores))
        per_node_scores_all.append(node_scores)
    return float(np.mean(state_scores)), state_scores, per_node_scores_all


# ============================================================
# 実行
# ============================================================

def run():
    results = {}

    print("=== ベンチマークA: 前向き型(FF) vs 再帰型(REC)、4ノード ===")
    pci_ff, ff_state_scores, _ = pci_full(ff_update_fn, 4, FF_REACHABLE_STATES)
    pci_rec, rec_state_scores, _ = pci_full(rec_update_fn, 4, FF_REACHABLE_STATES)
    reach_ff, ff_reach_states, _ = reach_full(ff_update_fn, 4, FF_REACHABLE_STATES)
    reach_rec, rec_reach_states, _ = reach_full(rec_update_fn, 4, FF_REACHABLE_STATES)
    print(f"PCI本実装: FF={pci_ff:.4f}, REC={pci_rec:.4f} (Φ: FF=0.0000, REC=0.0625)")
    print(f"到達範囲: FF={reach_ff:.4f}, REC={reach_rec:.4f}")
    results["setA"] = {
        "pci_ff": pci_ff, "pci_rec": pci_rec, "pci_ff_states": ff_state_scores, "pci_rec_states": rec_state_scores,
        "reach_ff": reach_ff, "reach_rec": reach_rec, "reach_ff_states": ff_reach_states, "reach_rec_states": rec_reach_states,
        "phi_ff": 0.0, "phi_rec": 0.0625,
    }

    print("\n=== ベンチマークB: 連続結合強度スイープ(p=0.0〜1.0)、5ノード ===")
    pci_by_p, reach_by_p = {}, {}
    for p in P_VALUES:
        update_fn = make_p_update_fn(p)
        pci_mean, _, _ = pci_full(update_fn, 5, CANDIDATE_STATES_5, seed=int(p * 1000))
        reach_mean, _, _ = reach_full(update_fn, 5, CANDIDATE_STATES_5, seed=int(p * 1000) + 1)
        pci_by_p[p] = pci_mean
        reach_by_p[p] = reach_mean
        print(f"p={p}: Φ={EXISTING_PHI[p]:.4f}, PCI本実装={pci_mean:.4f}, 到達範囲={reach_mean:.4f}")

    results["setB"] = {
        "p_values": P_VALUES,
        "phi": [EXISTING_PHI[p] for p in P_VALUES],
        "pci": [pci_by_p[p] for p in P_VALUES],
        "reach": [reach_by_p[p] for p in P_VALUES],
    }

    # 山型の再現度を定量化(全域スピアマン相関、ピーク位置)
    phi_vals = [EXISTING_PHI[p] for p in P_VALUES]
    pci_vals = [pci_by_p[p] for p in P_VALUES]
    reach_vals = [reach_by_p[p] for p in P_VALUES]

    peak_p_phi = P_VALUES[int(np.argmax(phi_vals))]
    peak_p_pci = P_VALUES[int(np.argmax(pci_vals))]
    peak_p_reach = P_VALUES[int(np.argmax(reach_vals))]

    spear_pci_full = stats.spearmanr(phi_vals, pci_vals)
    spear_reach_full = stats.spearmanr(phi_vals, reach_vals)

    rise_idx = [i for i, p in enumerate(P_VALUES) if p <= peak_p_phi]
    fall_idx = [i for i, p in enumerate(P_VALUES) if p >= peak_p_phi]
    spear_pci_rise = stats.spearmanr([phi_vals[i] for i in rise_idx], [pci_vals[i] for i in rise_idx])
    spear_pci_fall = stats.spearmanr([phi_vals[i] for i in fall_idx], [pci_vals[i] for i in fall_idx])
    spear_reach_rise = stats.spearmanr([phi_vals[i] for i in rise_idx], [reach_vals[i] for i in rise_idx])
    spear_reach_fall = stats.spearmanr([phi_vals[i] for i in fall_idx], [reach_vals[i] for i in fall_idx])

    print(f"\nΦのピーク: p={peak_p_phi} | PCI本実装のピーク: p={peak_p_pci} | 到達範囲のピーク: p={peak_p_reach}")
    print(f"全域スピアマン相関: PCI本実装={spear_pci_full.statistic:.3f}, 到達範囲={spear_reach_full.statistic:.3f}")
    print(f"上昇局面(p<={peak_p_phi})スピアマン相関: PCI本実装={spear_pci_rise.statistic:.3f}, 到達範囲={spear_reach_rise.statistic:.3f}")
    print(f"下降局面(p>={peak_p_phi})スピアマン相関: PCI本実装={spear_pci_fall.statistic:.3f}, 到達範囲={spear_reach_fall.statistic:.3f}")

    results["setB_analysis"] = {
        "peak_p_phi": peak_p_phi, "peak_p_pci": peak_p_pci, "peak_p_reach": peak_p_reach,
        "spearman_full": {"pci": spear_pci_full.statistic, "reach": spear_reach_full.statistic},
        "spearman_rise": {"pci": spear_pci_rise.statistic, "reach": spear_reach_rise.statistic},
        "spearman_fall": {"pci": spear_pci_fall.statistic, "reach": spear_reach_fall.statistic},
    }

    with open("iit_pci_gwt_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nsaved iit_pci_gwt_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    labels = ["前向き型(FF)", "再帰型(REC)"]
    x = np.arange(2)
    width = 0.25
    axes[0].bar(x - width, [0.0, 0.0625], width, label="厳密Φ", color="#4472C4")
    axes[0].bar(x, [pci_ff, pci_rec], width, label="PCI本実装", color="#C0504D")
    axes[0].bar(x + width, [reach_ff, reach_rec], width, label="到達範囲", color="#9BBB59")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("ベンチマークA: FF vs REC")
    axes[0].legend(fontsize=8)

    ax2 = axes[1]
    ax2b = ax2.twinx()
    ax2.plot(P_VALUES, phi_vals, "o-", color="#4472C4", label="厳密Φ(左軸)", linewidth=2)
    ax2b.plot(P_VALUES, pci_vals, "s--", color="#C0504D", label="PCI本実装(右軸)")
    ax2b.plot(P_VALUES, reach_vals, "^--", color="#9BBB59", label="到達範囲(右軸)")
    ax2.axvline(peak_p_phi, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("結合強度 p")
    ax2.set_ylabel("Φ", color="#4472C4")
    ax2b.set_ylabel("PCI本実装 / 到達範囲")
    ax2.set_title(f"ベンチマークB: 連続結合強度スイープ(Φのピークp={peak_p_phi})")
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper right")

    fig.suptitle("要件5: PCI本実装・GWT由来「到達範囲」指標の検証")
    fig.tight_layout()
    fig.savefig("iit_pci_gwt_comparison.png", dpi=150)
    print("グラフを iit_pci_gwt_comparison.png に保存しました。")


if __name__ == "__main__":
    run()
