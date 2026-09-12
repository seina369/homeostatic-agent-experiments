"""
感情AIプロジェクト フェーズ5 プロトタイプ: クラスター単位シナジーの連続対応関係の検証(要件5)
==========================================================

前回、代数的連結度は「統合の程度」を表す連続指標としては限定的にしか使えない
ことが分かった(結合強度pが純粋な追加として働く範囲p<=0.4でのみ厳密Φと単調に
対応し、p>0.4でクラスター内部の統合を置き換え始めると対応が完全に逆転する)。

今回は、前回のΦ代替指標比較(iit_alt_metrics_prototype.py)で「p=0の階層近似の
落とし穴を回避した」もう一つの候補である**クラスター単位のシナジー**
(全体の予測的相互情報量 - 各クラスター単独の予測的相互情報量の和)が、代数的
連結度が再現できなかった厳密Φの山型(p=0.4を頂点に、それ以降は結合強化が
むしろ統合を破壊する)を再現できるか、スピアマン相関が全域で安定して高いか
を検証する。

ネットワーク・シナジー計算はiit_alt_metrics_prototype.pyの
synergy_cluster_level関数・predictive_mi_whole/predictive_mi_cluster関数を
完全に同一のロジックで再利用する(pyphiのΦ計算とは異なり、ブルートフォースの
確率計算のみなので計算コストは無視できるほど軽い)。厳密Φはiit_phi_degree_
correspondence_prototype.pyで計算済みの9点(p=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,1.0)
をそのまま再利用する。

使い方:
  python3 iit_synergy_degree_correspondence_prototype.py
"""

import sys, json, itertools
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

N = 5
CLUSTER1 = (0, 1, 2)
CLUSTER2 = (3, 4)

# 厳密Φ(iit_phi_degree_correspondence_prototype.pyで既に計算済みの値をそのまま再利用)
PHI = {
    0.0: 0.0000, 0.1: 0.2174, 0.2: 0.3714, 0.3: 0.4368, 0.4: 0.4379,
    0.5: 0.3860, 0.6: 0.3772, 0.7: 0.1992, 1.0: 0.0000,
}


def make_update_fn(p):
    def fn(state):
        a, b, c, d, e = state
        pB = float(a and c)
        pC = float(a and b)
        pE = float(d)
        pA = (1 - p) * float(b and c) + p * float(d)
        pD = (1 - p) * float(e) + p * float(a)
        return [pA, pB, pC, pD, pE]
    return fn


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


def synergy_cluster_level(update_fn, n, cluster1, cluster2):
    node_probs, states = node_probs_table(update_fn, n)
    mi_whole = predictive_mi_whole(node_probs, states, n)
    mi_c1 = predictive_mi_cluster(node_probs, states, n, cluster1)
    mi_c2 = predictive_mi_cluster(node_probs, states, n, cluster2)
    return mi_whole, mi_c1 + mi_c2, mi_whole - (mi_c1 + mi_c2)


def main():
    ps = sorted(PHI.keys())
    rows = []
    for p in ps:
        update_fn = make_update_fn(p)
        mi_whole, mi_parts, synergy_c = synergy_cluster_level(update_fn, N, CLUSTER1, CLUSTER2)
        rows.append({"p": p, "phi": PHI[p], "mi_whole": mi_whole, "mi_parts": mi_parts, "synergy_cluster": synergy_c})
        print(f"p={p}: Φ={PHI[p]:.4f}, シナジー(cluster)={synergy_c:+.4f} (全体MI={mi_whole:.4f}, 部分和MI={mi_parts:.4f})")

    phi_vals = [r["phi"] for r in rows]
    syn_vals = [r["synergy_cluster"] for r in rows]

    peak_idx_phi = int(np.argmax(phi_vals))
    peak_p_phi = ps[peak_idx_phi]
    peak_idx_syn = int(np.argmax(syn_vals))
    peak_p_syn = ps[peak_idx_syn]
    print(f"\nΦのピーク: p={peak_p_phi}({phi_vals[peak_idx_phi]:.4f})")
    print(f"シナジー(cluster)のピーク: p={peak_p_syn}({syn_vals[peak_idx_syn]:+.4f})")

    pear_full = stats.pearsonr(phi_vals, syn_vals)
    spear_full = stats.spearmanr(phi_vals, syn_vals)
    print(f"\n[全域 p=0.0-1.0] Pearson(Φ,シナジーcluster)={pear_full.statistic:.3f}(p={pear_full.pvalue:.3f}), "
          f"Spearman={spear_full.statistic:.3f}(p={spear_full.pvalue:.3f})")

    rise_idx = [i for i, p in enumerate(ps) if p <= peak_p_phi]
    fall_idx = [i for i, p in enumerate(ps) if p >= peak_p_phi]

    def seg(idxs):
        sp = [phi_vals[i] for i in idxs]
        ss = [syn_vals[i] for i in idxs]
        return stats.pearsonr(sp, ss), stats.spearmanr(sp, ss)

    pr, sr = seg(rise_idx)
    pf, sf = seg(fall_idx)
    print(f"[上昇局面 p<={peak_p_phi}] Pearson={pr.statistic:.3f}, Spearman={sr.statistic:.3f}")
    print(f"[下降局面 p>={peak_p_phi}] Pearson={pf.statistic:.3f}, Spearman={sf.statistic:.3f}")

    summary = {
        "ps": ps, "phi": phi_vals, "synergy_cluster": syn_vals,
        "peak_p_phi": peak_p_phi, "peak_p_synergy": peak_p_syn,
        "correlation_full": {"pearson": pear_full.statistic, "spearman": spear_full.statistic},
        "correlation_rising": {"pearson": pr.statistic, "spearman": sr.statistic},
        "correlation_falling": {"pearson": pf.statistic, "spearman": sf.statistic},
    }
    with open("iit_synergy_degree_correspondence_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].plot(ps, phi_vals, "o-", color="#4472C4", label="厳密Φ", linewidth=2)
    ax2 = axes[0].twinx()
    ax2.plot(ps, syn_vals, "^-", color="#8064A2", label="シナジー(クラスター単位)")
    axes[0].set_xlabel("結合強度 p")
    axes[0].set_ylabel("Φ", color="#4472C4")
    ax2.set_ylabel("シナジー(クラスター単位)", color="#8064A2")
    axes[0].axvline(peak_p_phi, color="gray", linestyle=":", alpha=0.6)
    axes[0].set_title("厳密Φ vs クラスター単位シナジー")
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[0].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    axes[1].scatter(phi_vals, syn_vals, color="#4472C4")
    for p, x, y in zip(ps, phi_vals, syn_vals):
        axes[1].annotate(f"p={p}", (x, y), fontsize=7)
    axes[1].set_xlabel("厳密Φ")
    axes[1].set_ylabel("シナジー(クラスター単位)")
    axes[1].set_title(f"散布図(全域Spearman={spear_full.statistic:.3f})")

    fig.suptitle("要件5: クラスター単位シナジーはΦの山型を再現できるか")
    fig.tight_layout()
    fig.savefig("iit_synergy_degree_correspondence.png", dpi=150)
    print("グラフを iit_synergy_degree_correspondence.png に保存しました。")


if __name__ == "__main__":
    main()
