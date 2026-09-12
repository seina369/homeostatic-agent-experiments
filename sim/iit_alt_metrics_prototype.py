"""
感情AIプロジェクト フェーズ5 プロトタイプ: Φの代替指標の比較検証
==========================================================

これまでpyphiによる厳密なΦ計算(iit_phi_prototype.py: 前向き型/再帰型4ノード、
iit_phi_modularity_prototype.py: モジュール性パラメータpの5ノードネットワーク群)
を行ってきたが、Φ自体は計算量的に現実的な規模へ拡張できない。本プロトタイプは、
より軽量に計算できる3種類の代替指標を実装し、既存の2つのネットワーク群に適用して、
(1)前向き型vs再帰型の対比を厳密Φと同じように再現できるか、(2)モジュール性
p=0(完全モジュール)で厳密Φの階層近似がハマった「クラスター内部の冗長な構造を
誤って全系の統合とみなす」落とし穴を、これらの指標も同じように踏むか、を検証する。

  (1) **PCI風の擾乱複雑性**: 1ノードの状態を反転させ(摂動)、その後T=15ステップの
      全ノードの時空間的な活動パターンをLempel-Ziv複雑性(Kaspar-Schuster版)で
      定量化し、ランダム系列を基準に正規化する(実際のPCIと同じ考え方)。
      確率的な系(モジュール性実験、pによる混合)は30試行の平均を取る。
  (2) **相乗情報量(シナジー)の簡易代理指標**: 全体の予測的相互情報量
      I(X_t;X_{t+1})(現在の全ノード状態と次の全ノード状態の相互情報量、
      TPMから厳密に計算)から、部分(個々のノード、モジュール性実験ではさらに
      クラスター単位)ごとの予測的相互情報量の和を引いた差分。全体が部分の和を
      超えて情報を運んでいれば正の値になる。
  (3) **グラフ理論的な連結度**: 結合強度を重みとした無向化グラフ(自己ループは
      除く)のラプラシアンから代数的連結度(フィードラー値、2番目に小さい固有値)と
      最小カット値(全2^(n-1)通りの二分割を総当たりして最小の切断重みを求める)を
      計算する。前向き型/再帰型は結合の有無(0/1)、モジュール性実験は橋渡し辺の
      重みをpそのものとする。

使い方:
  python3 iit_alt_metrics_prototype.py
"""

import sys, json, random, itertools
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collections
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import pyphi
pyphi.config.WELCOME_OFF = True
pyphi.config.PROGRESS_BARS = False

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# ネットワーク定義(既存2実験と完全に同一のロジックを再利用)
# ============================================================

def ff_update_fn(state):
    a, b, c, d = state
    return [float(a), float(b), float(a and b), float(c)]


def rec_update_fn(state):
    a, b, c, d = state
    a_next = float((d and b) or ((not d) and a))
    b_next = float((c and a) or ((not c) and b))
    return [a_next, b_next, float(a and b), float(c)]


FF_CM = np.array([
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
], dtype=float)

REC_CM = np.array([
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 0],
], dtype=float)


def make_b_update_fn(p):
    def fn(state):
        a, b, c, d, e = state
        pB = float(a and c)
        pC = float(a and b)
        pE = float(d)
        pA = (1 - p) * float(b and c) + p * float(d)
        pD = (1 - p) * float(e) + p * float(a)
        return [pA, pB, pC, pD, pE]
    return fn


def make_b_cm(p):
    bridge = p  # 結合強度そのものを重みにする
    return np.array([
        [1, 1, 1, bridge, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [bridge, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=float)


B_CLUSTER1 = (0, 1, 2)
B_CLUSTER2 = (3, 4)
P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def build_tpm(update_fn, n):
    states = list(pyphi.utils.all_states(n))
    tpm = np.zeros((2 ** n, n))
    for i, s in enumerate(states):
        tpm[i] = update_fn(s)
    return tpm


# ============================================================
# (1) PCI風 擾乱複雑性指標
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


def sample_next_state(probs, rng):
    return tuple(1 if rng.random() < p else 0 for p in probs)


def run_trajectory(update_fn, start_state, perturb_idx, T, rng):
    state = list(start_state)
    state[perturb_idx] = 1 - state[perturb_idx]
    state = tuple(state)
    traj = [state]
    for _ in range(T):
        probs = update_fn(state)
        state = sample_next_state(probs, rng)
        traj.append(state)
    return traj


def pci_like_metric(update_fn, n, start_state, perturb_idx=0, T=15, n_trials=30, seed=0):
    rng = random.Random(seed)
    scores = []
    for _ in range(n_trials):
        traj = run_trajectory(update_fn, start_state, perturb_idx, T, rng)
        bits = []
        for i in range(n):
            for t in range(len(traj)):
                bits.append(str(traj[t][i]))
        bitstring = "".join(bits)
        c = lz_complexity(bitstring)
        L = len(bitstring)
        norm = c * np.log2(L) / L if L > 1 else 0.0
        scores.append(norm)
    return float(np.mean(scores)), float(np.std(scores))


def pci_like_metric_averaged(update_fn, n, candidate_states, T=15, n_trials=10, seed=0):
    """単一の(開始状態,摂動ノード)だけだと、たまたま両ネットワークの応答が
    一致してしまう組み合わせを引きやすい(実際にSet Aで確認済み)。複数の
    開始状態×全ノードへの摂動を平均して、頑健な代表値を得る。"""
    all_scores = []
    for start_state in candidate_states:
        for perturb_idx in range(n):
            mean, _ = pci_like_metric(update_fn, n, start_state, perturb_idx, T, n_trials, seed)
            all_scores.append(mean)
    return float(np.mean(all_scores)), float(np.std(all_scores))


# ============================================================
# (2) シナジー(相乗情報量)簡易代理指標
# ============================================================

def h_binary(p):
    if p <= 1e-12 or p >= 1 - 1e-12:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def node_probs_table(update_fn, n):
    states = list(itertools.product([0, 1], repeat=n))
    return {s: update_fn(s) for s in states}, states


def predictive_mi_whole(node_probs, states, n):
    prior = 1.0 / len(states)
    cond_h = np.mean([sum(h_binary(p) for p in node_probs[s]) for s in states])
    marg = {y: 0.0 for y in states}
    for s in states:
        probs = node_probs[s]
        for y in states:
            py = 1.0
            for i in range(n):
                py *= probs[i] if y[i] == 1 else (1 - probs[i])
            marg[y] += prior * py
    h_marg = -sum(v * np.log2(v) for v in marg.values() if v > 1e-12)
    return float(h_marg - cond_h)


def predictive_mi_node(node_probs, states, node_idx):
    p_given_0 = np.mean([node_probs[s][node_idx] for s in states if s[node_idx] == 0])
    p_given_1 = np.mean([node_probs[s][node_idx] for s in states if s[node_idx] == 1])
    p_marginal = 0.5 * p_given_0 + 0.5 * p_given_1
    h_y = h_binary(p_marginal)
    h_y_given_x = 0.5 * h_binary(p_given_0) + 0.5 * h_binary(p_given_1)
    return float(h_y - h_y_given_x)


def predictive_mi_cluster(node_probs, states, n, cluster_indices):
    k = len(cluster_indices)
    cluster_states = list(itertools.product([0, 1], repeat=k))
    prior_c = 1.0 / len(cluster_states)
    joint_by_xc = {}
    for xc in cluster_states:
        consistent = [s for s in states if tuple(s[i] for i in cluster_indices) == xc]
        joint_yc = {yc: 0.0 for yc in cluster_states}
        for s in consistent:
            probs_s = node_probs[s]
            for yc in cluster_states:
                p = 1.0
                for idx, ci in enumerate(cluster_indices):
                    p *= probs_s[ci] if yc[idx] == 1 else (1 - probs_s[ci])
                joint_yc[yc] += p / len(consistent)
        joint_by_xc[xc] = joint_yc
    h_y_given_x = 0.0
    marg_y = {yc: 0.0 for yc in cluster_states}
    for xc in cluster_states:
        dist = joint_by_xc[xc]
        h = -sum(v * np.log2(v) for v in dist.values() if v > 1e-12)
        h_y_given_x += prior_c * h
        for yc, v in dist.items():
            marg_y[yc] += prior_c * v
    h_y = -sum(v * np.log2(v) for v in marg_y.values() if v > 1e-12)
    return float(h_y - h_y_given_x)


def synergy_node_level(update_fn, n):
    node_probs, states = node_probs_table(update_fn, n)
    mi_whole = predictive_mi_whole(node_probs, states, n)
    mi_parts = sum(predictive_mi_node(node_probs, states, i) for i in range(n))
    return mi_whole, mi_parts, mi_whole - mi_parts


def synergy_cluster_level(update_fn, n, cluster1, cluster2):
    node_probs, states = node_probs_table(update_fn, n)
    mi_whole = predictive_mi_whole(node_probs, states, n)
    mi_c1 = predictive_mi_cluster(node_probs, states, n, cluster1)
    mi_c2 = predictive_mi_cluster(node_probs, states, n, cluster2)
    return mi_whole, mi_c1 + mi_c2, mi_whole - (mi_c1 + mi_c2)


# ============================================================
# (3) グラフ理論的な連結度指標
# ============================================================

def algebraic_connectivity(W):
    Wz = W.copy().astype(float)
    np.fill_diagonal(Wz, 0.0)
    Wsym = (Wz + Wz.T) / 2.0
    deg = np.diag(Wsym.sum(axis=1))
    L = deg - Wsym
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


def min_cut(W):
    Wz = W.copy().astype(float)
    np.fill_diagonal(Wz, 0.0)
    Wsym = (Wz + Wz.T) / 2.0
    n = W.shape[0]
    best = None
    for mask in range(1, 2 ** n - 1):
        S = [i for i in range(n) if (mask >> i) & 1]
        Tt = [i for i in range(n) if not ((mask >> i) & 1)]
        cut = sum(Wsym[i, j] for i in S for j in Tt)
        if best is None or cut < best:
            best = cut
    return float(best)


# ============================================================
# 厳密Φ(既存p=0〜0.5は再利用、p=0.7/1.0は新規計算)
# ============================================================

EXISTING_PHI = {
    0.0: 0.0000, 0.1: 0.2174, 0.2: 0.3714, 0.3: 0.4368, 0.5: 0.3860,
}
CANONICAL_STATE_B = (0, 0, 0, 0, 0)
CANONICAL_STATE_A = (0, 0, 0, 0)  # FF・RECの両方で到達可能な共通状態(FFはΦ=0、RECはΦ=0.0625)


def compute_exact_phi_b(p):
    tpm = build_tpm(make_b_update_fn(p), 5)
    cm = np.where(make_b_cm(p) > 0, 1, 0)  # pyphiのcmは0/1(依存の有無)
    network = pyphi.Network(tpm, cm=cm, node_labels=("A", "B", "C", "D", "E"))
    sub = pyphi.Subsystem(network, CANONICAL_STATE_B)
    return float(pyphi.compute.sia(sub).phi)


def compute_exact_phi_a(update_fn, cm, state):
    tpm = build_tpm(update_fn, 4)
    network = pyphi.Network(tpm, cm=cm.astype(int), node_labels=("A", "B", "C", "D"))
    try:
        sub = pyphi.Subsystem(network, state)
    except pyphi.exceptions.StateUnreachableError:
        return None
    return float(pyphi.compute.sia(sub).phi)


def main():
    rows = []

    # --- Set A: 前向き型 / 再帰型 ---
    # PCI風は単一の(開始状態,摂動ノード)だと両ネットワークの応答がたまたま
    # 一致する組み合わせを引きやすいことが予備検証で判明したため、両ネットワーク
    # 共通の到達可能状態8つ×全4ノードへの摂動、計32通りを平均した頑健な値を使う。
    ff_reachable_states = [s for s in itertools.product([0, 1], repeat=4) if s[2] == (s[0] and s[1])]
    for name, update_fn, cm in [("前向き型(FF)", ff_update_fn, FF_CM), ("再帰型(REC)", rec_update_fn, REC_CM)]:
        phi = compute_exact_phi_a(update_fn, cm, CANONICAL_STATE_A)
        pci_mean, pci_std = pci_like_metric_averaged(update_fn, 4, ff_reachable_states, n_trials=10)
        mi_whole, mi_parts, synergy = synergy_node_level(update_fn, 4)
        ac = algebraic_connectivity(cm)
        mc = min_cut(cm)
        rows.append({
            "group": "A", "label": name, "p": None, "phi_exact": phi,
            "pci_mean": pci_mean, "pci_std": pci_std,
            "mi_whole": mi_whole, "mi_parts": mi_parts, "synergy_node": synergy,
            "synergy_cluster": None, "alg_connectivity": ac, "min_cut": mc,
        })
        print(f"[Set A] {name}: Φ={phi}, PCI={pci_mean:.4f}±{pci_std:.4f}, "
              f"シナジー(node)={synergy:+.4f}(全体MI={mi_whole:.4f}, 部分和={mi_parts:.4f}), "
              f"代数的連結度={ac:.4f}, 最小カット={mc:.4f}")

    # --- Set B: モジュール性スイープ ---
    for p in P_VALUES:
        update_fn = make_b_update_fn(p)
        cm = make_b_cm(p)
        if p in EXISTING_PHI:
            phi = EXISTING_PHI[p]
        else:
            phi = compute_exact_phi_b(p)
        pci_mean, pci_std = pci_like_metric(update_fn, 5, (0, 0, 0, 0, 0), perturb_idx=0)
        mi_whole, mi_parts, synergy = synergy_node_level(update_fn, 5)
        _, _, synergy_c = synergy_cluster_level(update_fn, 5, B_CLUSTER1, B_CLUSTER2)
        ac = algebraic_connectivity(cm)
        mc = min_cut(cm)
        rows.append({
            "group": "B", "label": f"p={p}", "p": p, "phi_exact": phi,
            "pci_mean": pci_mean, "pci_std": pci_std,
            "mi_whole": mi_whole, "mi_parts": mi_parts, "synergy_node": synergy,
            "synergy_cluster": synergy_c, "alg_connectivity": ac, "min_cut": mc,
        })
        print(f"[Set B] p={p}: Φ={phi}, PCI={pci_mean:.4f}±{pci_std:.4f}, "
              f"シナジー(node)={synergy:+.4f}, シナジー(cluster)={synergy_c:+.4f}, "
              f"代数的連結度={ac:.4f}, 最小カット={mc:.4f}")

    with open("iit_alt_metrics_results.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved iit_alt_metrics_results.json")

    # --- グラフ化 ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    a_rows = [r for r in rows if r["group"] == "A"]
    labels_a = [r["label"] for r in a_rows]
    x = np.arange(len(labels_a))
    width = 0.35
    axes[0, 0].bar(x - width / 2, [r["phi_exact"] or 0 for r in a_rows], width, label="Φ(厳密)", color="#4472C4")
    axes[0, 0].bar(x + width / 2, [r["pci_mean"] for r in a_rows], width, label="PCI風", color="#C0504D")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels_a)
    axes[0, 0].set_title("Set A: Φ vs PCI風")
    axes[0, 0].legend(fontsize=8)

    b_rows = [r for r in rows if r["group"] == "B"]
    ps = [r["p"] for r in b_rows]
    axes[0, 1].plot(ps, [r["phi_exact"] for r in b_rows], "o-", label="Φ(厳密)", color="#4472C4")
    axes[0, 1].plot(ps, [r["pci_mean"] for r in b_rows], "s-", label="PCI風", color="#C0504D")
    axes[0, 1].set_xlabel("結合強度 p")
    axes[0, 1].set_title("Set B: Φ vs PCI風")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(ps, [r["synergy_node"] for r in b_rows], "o-", label="シナジー(ノード単位)", color="#9BBB59")
    axes[1, 0].plot(ps, [r["synergy_cluster"] for r in b_rows], "^-", label="シナジー(クラスター単位)", color="#8064A2")
    axes[1, 0].axhline(0, color="gray", linewidth=0.8)
    axes[1, 0].set_xlabel("結合強度 p")
    axes[1, 0].set_title("Set B: シナジー代理指標")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(ps, [r["alg_connectivity"] for r in b_rows], "o-", label="代数的連結度", color="#4472C4")
    axes[1, 1].plot(ps, [r["min_cut"] for r in b_rows], "s-", label="最小カット", color="#C0504D")
    axes[1, 1].set_xlabel("結合強度 p")
    axes[1, 1].set_title("Set B: グラフ理論的連結度")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("要件5: Φの代替指標の比較(擾乱複雑性・シナジー・グラフ連結度)")
    fig.tight_layout()
    fig.savefig("iit_alt_metrics_comparison.png", dpi=150)
    print("グラフを iit_alt_metrics_comparison.png に保存しました。")


if __name__ == "__main__":
    main()
