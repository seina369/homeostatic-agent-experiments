"""
感情AIプロジェクト フェーズ5 プロトタイプ: 厳密Φと代数的連結度の連続対応関係の検証(要件5)
==========================================================

前回、代数的連結度は「統合か分断か」の二値判定では厳密Φと頑健に一致することを
確認した。今回は、値そのものが「統合の程度」を表す連続指標として使えるか
(値が大きいほど両方とも大きくなる単調な対応関係にあるか)を検証する。

ネットワークは既存のiit_phi_modularity_prototype.py / iit_alt_metrics_prototype.py
と同じ5ノード設計(クラスター1={A,B,C}のAND再帰構造、クラスター2={D,E}の
フリップフロップ、A<->D間の結合強度pで連続的に制御)を再利用し、pを
0.0〜1.0まで細かく振って、厳密Φと代数的連結度の値そのものの対応を調べる。

p=0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0の7点は既存の実験(iit_modularity_p*.json,
iit_alt_metrics_results.json)からそのまま再利用し、ピーク位置を精緻化する
ためにp=0.4, 0.6を新たに計算して追加した(いずれもN=5、1状態あたり1秒未満、
45秒の壁に対して十分な余裕がある)。

さらに、「なぜ対応が崩れるのか」を構造的に特定するため、代数的連結度の
計算に使う重み付きグラフを2通り用意して比較した。
  (1) 素朴な重み(これまでの実験で使用): クラスター内部の結合(A-B,A-C,B-C,D-E)
      は常に重み1固定、橋渡し(A-D)だけをpで動かす。
  (2) 修正版の重み: 実際のTPMの係数を反映し、A<-B,A<-C,D<-Eの依存が
      (1-p)で希釈されていく効果も重みに含める(A-B, A-C, D-E の重みを
      1 - p/2 とする。A'=(1-p)(B∧C)+p*D, D'=(1-p)E+p*A という式で、
      A→B, A→C, B→C, D→E方向の依存は元々pに依存しないため、双方向の
      平均を取ると各辺の重みは (1 + (1-p))/2 = 1-p/2 になる)。
素朴な重みは「橋渡しが強くなるほど連結度が増える」という一方向の効果しか
捉えられないのに対し、修正版は「橋渡しが強くなるほどクラスター内部の
実質的な結合が薄まっていく」という、TPMの式に実際に存在するもう一方の
効果も反映している。

使い方:
  python3 iit_phi_degree_correspondence_prototype.py compute_new <p>
  python3 iit_phi_degree_correspondence_prototype.py aggregate
"""

import sys, json, time
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import collections
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

import numpy as np
from scipy import stats
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

N = 5
NODE_LABELS = ("A", "B", "C", "D", "E")

# 既存実験(iit_alt_metrics_results.json, group B)からそのまま再利用する値
EXISTING = {
    0.0: 0.0, 0.1: 0.2174, 0.2: 0.3714, 0.3: 0.4368,
    0.5: 0.3860, 0.7: 0.1992, 1.0: 0.0,
}


def make_tpm(p):
    states = list(pyphi.utils.all_states(N))
    tpm = np.zeros((2 ** N, N))
    for i, s in enumerate(states):
        a, b, c, d, e = s
        pB = float(a and c)
        pC = float(a and b)
        pE = float(d)
        pA = (1 - p) * float(b and c) + p * float(d)
        pD = (1 - p) * float(e) + p * float(a)
        tpm[i] = [pA, pB, pC, pD, pE]
    return tpm


def make_cm(p):
    bridge = 1 if p > 0 else 0
    return np.array([
        [0, 1, 1, bridge, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [bridge, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
    ])


def naive_W(p):
    W = np.zeros((5, 5))
    W[0, 1] = W[1, 0] = 1.0
    W[0, 2] = W[2, 0] = 1.0
    W[1, 2] = W[2, 1] = 1.0
    W[3, 4] = W[4, 3] = 1.0
    W[0, 3] = W[3, 0] = p
    return W


def corrected_W(p):
    W = np.zeros((5, 5))
    W[0, 1] = W[1, 0] = 1 - p / 2
    W[0, 2] = W[2, 0] = 1 - p / 2
    W[1, 2] = W[2, 1] = 1.0
    W[3, 4] = W[4, 3] = 1 - p / 2
    W[0, 3] = W[3, 0] = p
    return W


def algebraic_connectivity(W):
    Wz = W.copy()
    np.fill_diagonal(Wz, 0.0)
    Wsym = (Wz + Wz.T) / 2.0
    deg = np.diag(Wsym.sum(axis=1))
    L = deg - Wsym
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


def compute_new(p, n_states=3):
    tpm = make_tpm(p)
    cm = make_cm(p)
    network = pyphi.Network(tpm, cm=cm, node_labels=NODE_LABELS)
    phis = []
    count = 0
    t0 = time.time()
    for state in pyphi.utils.all_states(N):
        if count >= n_states:
            break
        try:
            sub = pyphi.Subsystem(network, state)
        except pyphi.exceptions.StateUnreachableError:
            continue
        sia = pyphi.compute.sia(sub)
        phis.append(float(sia.phi))
        count += 1
    phi_mean = float(np.mean(phis))
    dt = time.time() - t0
    print(f"p={p}: phis={phis}, mean={phi_mean:.4f}, elapsed={dt:.2f}s")
    fname = f"iit_degree_p{str(p).replace('.', '_')}.json"
    with open(fname, "w") as f:
        json.dump({"p": p, "phis": phis, "phi_mean": phi_mean, "seconds": dt}, f, ensure_ascii=False, indent=2)
    print(f"saved {fname}")


def aggregate():
    ps_all = sorted(set(list(EXISTING.keys()) + [0.4, 0.6]))
    phi_vals = []
    for p in ps_all:
        if p in EXISTING:
            phi_vals.append(EXISTING[p])
        else:
            fname = f"iit_degree_p{str(p).replace('.', '_')}.json"
            with open(fname) as f:
                d = json.load(f)
            phi_vals.append(d["phi_mean"])

    alg_naive_vals = [algebraic_connectivity(naive_W(p)) for p in ps_all]
    alg_corr_vals = [algebraic_connectivity(corrected_W(p)) for p in ps_all]

    print("=== 統合パラメータpに対する厳密Φ・代数的連結度(素朴/修正)の一覧 ===")
    for p, phi, an, ac in zip(ps_all, phi_vals, alg_naive_vals, alg_corr_vals):
        print(f"p={p}: Φ={phi:.4f}, 連結度(素朴)={an:.4f}, 連結度(修正)={ac:.4f}")

    peak_idx = int(np.argmax(phi_vals))
    peak_p = ps_all[peak_idx]
    print(f"\nΦのピークはp={peak_p}(Φ={phi_vals[peak_idx]:.4f})付近")

    # 全域相関
    pear_naive = stats.pearsonr(phi_vals, alg_naive_vals)
    spear_naive = stats.spearmanr(phi_vals, alg_naive_vals)
    pear_corr = stats.pearsonr(phi_vals, alg_corr_vals)
    spear_corr = stats.spearmanr(phi_vals, alg_corr_vals)
    print(f"\n[全域 p=0.0-1.0] Pearson(Φ,素朴連結度)={pear_naive.statistic:.3f}(p={pear_naive.pvalue:.3f}), "
          f"Spearman={spear_naive.statistic:.3f}")
    print(f"[全域 p=0.0-1.0] Pearson(Φ,修正連結度)={pear_corr.statistic:.3f}(p={pear_corr.pvalue:.3f}), "
          f"Spearman={spear_corr.statistic:.3f}")

    # 上昇局面 / 下降局面に分割
    rise_idx = [i for i, p in enumerate(ps_all) if p <= peak_p]
    fall_idx = [i for i, p in enumerate(ps_all) if p >= peak_p]

    def seg_stats(idxs, arr):
        sub_phi = [phi_vals[i] for i in idxs]
        sub_arr = [arr[i] for i in idxs]
        return stats.pearsonr(sub_phi, sub_arr), stats.spearmanr(sub_phi, sub_arr)

    pr_naive, sr_naive = seg_stats(rise_idx, alg_naive_vals)
    pf_naive, sf_naive = seg_stats(fall_idx, alg_naive_vals)
    pr_corr, sr_corr = seg_stats(rise_idx, alg_corr_vals)
    pf_corr, sf_corr = seg_stats(fall_idx, alg_corr_vals)

    print(f"\n[上昇局面 p<= {peak_p}] Pearson(素朴)={pr_naive.statistic:.3f}, Spearman(素朴)={sr_naive.statistic:.3f} | "
          f"Pearson(修正)={pr_corr.statistic:.3f}, Spearman(修正)={sr_corr.statistic:.3f}")
    print(f"[下降局面 p>= {peak_p}] Pearson(素朴)={pf_naive.statistic:.3f}, Spearman(素朴)={sf_naive.statistic:.3f} | "
          f"Pearson(修正)={pf_corr.statistic:.3f}, Spearman(修正)={sf_corr.statistic:.3f}")

    summary = {
        "ps": ps_all, "phi": phi_vals, "alg_naive": alg_naive_vals, "alg_corrected": alg_corr_vals,
        "peak_p": peak_p,
        "correlation_full": {
            "pearson_naive": pear_naive.statistic, "spearman_naive": spear_naive.statistic,
            "pearson_corrected": pear_corr.statistic, "spearman_corrected": spear_corr.statistic,
        },
        "correlation_rising": {
            "pearson_naive": pr_naive.statistic, "spearman_naive": sr_naive.statistic,
            "pearson_corrected": pr_corr.statistic, "spearman_corrected": sr_corr.statistic,
        },
        "correlation_falling": {
            "pearson_naive": pf_naive.statistic, "spearman_naive": sf_naive.statistic,
            "pearson_corrected": pf_corr.statistic, "spearman_corrected": sf_corr.statistic,
        },
    }
    with open("iit_degree_correspondence_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ps_all, phi_vals, "o-", color="#4472C4", label="厳密Φ", linewidth=2)
    ax.plot(ps_all, alg_naive_vals, "s--", color="#C0504D", label="代数的連結度(素朴な重み)")
    ax.plot(ps_all, alg_corr_vals, "^--", color="#9BBB59", label="代数的連結度(修正した重み)")
    ax.axvline(peak_p, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("結合強度 p")
    ax.set_ylabel("値")
    ax.set_title(f"要件5: 厳密Φと代数的連結度の連続対応(ピークはp={peak_p}付近)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("iit_phi_degree_correspondence.png", dpi=150)
    print("グラフを iit_phi_degree_correspondence.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "compute_new":
        compute_new(float(sys.argv[2]))
    elif cmd == "aggregate":
        aggregate()
