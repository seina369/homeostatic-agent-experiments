"""
感情AIプロジェクト フェーズ5 プロトタイプ: 要件5 Φの階層的近似計算の検証
==========================================================

前回(iit_phi_prototype.py)は4ノードの最小トイモデルで、前向き型はΦ=0、
再帰型はΦ>0という、IITの基本予測を確認した。今回は要件5の核心的な壁である
「Φの計算量」への対処として、ノードをクラスターにまとめて階層的に近似計算する
手法が有効かどうかを検証する。

理論的な限界(検証前の予想): この近似は、真の最小情報分割(MIP)がクラスターの
境界を横切る可能性を排除できず、システムがモジュール的な構造を持つという
仮定に精度が依存するはずである。

**近似手法の定義**: 系をクラスター1・クラスター2に分け、各クラスターを
「他方のクラスターは現在の状態で固定された背景条件」として単独のサブシステム
とみなし、pyphiのSubsystem(network, state, nodes=cluster)でクラスターごとの
厳密Φを計算する。これを Φ_cluster1, Φ_cluster2 とし、
  近似Φ(和) = Φ_cluster1 + Φ_cluster2
  近似Φ(min) = min(Φ_cluster1, Φ_cluster2)
の2通りで全系の近似Φを構成する(前者は「弱結合な部分系の統合はおおむね加算的」
という物理的直感、後者は「全体のΦは最も統合の弱い部分によって支配される」
という直感に対応する、どちらも一つの妥当な選択肢であり、他の集約則もありうる
点に注意)。

**ネットワーク設計**: 5ノード(A,B,C,D,E)、クラスター1={A,B,C}(3ノード、
A'=B AND C(結合項付き)、B'=A AND C、C'=A AND Bの再帰的AND構造)、
クラスター2={D,E}(2ノード、D'=E、E'=D の相互コピーによるフリップフロップ、
Dのみ結合項を持つ)。クラスター間の結合はA<->Dの1本の橋渡しのみとし、
その強さを確率パラメータp(モジュール性の強さ、p=0で完全に独立な2系、
pが大きいほど強結合)で連続的に制御する:

  P(A'=1) = (1-p)*[B AND C] + p*[D]
  P(D'=1) = (1-p)*[E]       + p*[A]

p=0/0.1/0.2/0.3/0.5 で、全系の厳密Φ・各クラスター単独のΦ・近似Φ・誤差を計算し、
モジュール性(pの小ささ)と近似精度の関係を定量化する。

pyphiは6ノード以上になると1状態あたりの計算が45秒の壁を超えるため(実測確認済み)、
5ノードを本検証の上限とした。

使い方:
  python3 iit_phi_modularity_prototype.py run <p>
  python3 iit_phi_modularity_prototype.py aggregate
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
CLUSTER1 = (0, 1, 2)
CLUSTER2 = (3, 4)
P_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]
N_STATES_PER_P = 3


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


def run_p(p):
    tpm = make_tpm(p)
    cm = make_cm(p)
    network = pyphi.Network(tpm, cm=cm, node_labels=NODE_LABELS)

    results = []
    count = 0
    for state in pyphi.utils.all_states(N):
        if count >= N_STATES_PER_P:
            break
        try:
            sub_whole = pyphi.Subsystem(network, state)
        except pyphi.exceptions.StateUnreachableError:
            continue

        t0 = time.time()
        sia_whole = pyphi.compute.sia(sub_whole)
        phi_whole = float(sia_whole.phi)

        sub_c1 = pyphi.Subsystem(network, state, nodes=CLUSTER1)
        sia_c1 = pyphi.compute.sia(sub_c1)
        phi_c1 = float(sia_c1.phi)

        sub_c2 = pyphi.Subsystem(network, state, nodes=CLUSTER2)
        sia_c2 = pyphi.compute.sia(sub_c2)
        phi_c2 = float(sia_c2.phi)
        dt = time.time() - t0

        phi_approx_sum = phi_c1 + phi_c2
        phi_approx_min = min(phi_c1, phi_c2)
        err_sum = phi_whole - phi_approx_sum
        err_min = phi_whole - phi_approx_min

        results.append({
            "p": p, "state": state, "phi_whole": phi_whole,
            "phi_cluster1": phi_c1, "phi_cluster2": phi_c2,
            "phi_approx_sum": phi_approx_sum, "phi_approx_min": phi_approx_min,
            "error_sum": err_sum, "error_min": err_min, "seconds": dt,
        })
        print(f"p={p} state={state}: Φ_whole={phi_whole:.4f}, Φ_c1={phi_c1:.4f}, Φ_c2={phi_c2:.4f}, "
              f"近似(和)={phi_approx_sum:.4f}(誤差{err_sum:+.4f}), 近似(min)={phi_approx_min:.4f}(誤差{err_min:+.4f}), {dt:.2f}s")
        count += 1

    fname = f"iit_modularity_p{str(p).replace('.', '_')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved {fname}")


def aggregate():
    all_results = []
    for p in P_VALUES:
        fname = f"iit_modularity_p{str(p).replace('.', '_')}.json"
        with open(fname) as f:
            all_results.extend(json.load(f))

    print("=== モジュール性(p)と近似誤差の関係(状態ごとの平均) ===")
    summary = {}
    for p in P_VALUES:
        rows = [r for r in all_results if r["p"] == p]
        phi_whole = [r["phi_whole"] for r in rows]
        err_sum = [r["error_sum"] for r in rows]
        err_min = [r["error_min"] for r in rows]
        abs_err_sum = [abs(e) for e in err_sum]
        abs_err_min = [abs(e) for e in err_min]
        summary[p] = {
            "n_states": len(rows),
            "phi_whole_mean": float(np.mean(phi_whole)),
            "err_sum_mean": float(np.mean(err_sum)), "abs_err_sum_mean": float(np.mean(abs_err_sum)),
            "err_min_mean": float(np.mean(err_min)), "abs_err_min_mean": float(np.mean(abs_err_min)),
        }
        print(f"p={p}(n={len(rows)}): Φ_whole平均={np.mean(phi_whole):.4f}, "
              f"誤差(和)平均={np.mean(err_sum):+.4f}(|誤差|平均{np.mean(abs_err_sum):.4f}), "
              f"誤差(min)平均={np.mean(err_min):+.4f}(|誤差|平均{np.mean(abs_err_min):.4f})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ps = P_VALUES
    phi_whole_means = [summary[p]["phi_whole_mean"] for p in ps]
    axes[0].plot(ps, phi_whole_means, "o-", color="#4472C4", label="Φ_whole(厳密)")
    approx_sum_means = [summary[p]["phi_whole_mean"] - summary[p]["err_sum_mean"] for p in ps]
    approx_min_means = [summary[p]["phi_whole_mean"] - summary[p]["err_min_mean"] for p in ps]
    axes[0].plot(ps, approx_sum_means, "s--", color="#C0504D", label="近似Φ(和)")
    axes[0].plot(ps, approx_min_means, "^--", color="#9BBB59", label="近似Φ(min)")
    axes[0].set_xlabel("結合強度 p (0=完全モジュール)")
    axes[0].set_ylabel("Φ")
    axes[0].set_title("厳密Φ vs 近似Φ")
    axes[0].legend(fontsize=8)

    abs_err_sum_means = [summary[p]["abs_err_sum_mean"] for p in ps]
    abs_err_min_means = [summary[p]["abs_err_min_mean"] for p in ps]
    axes[1].plot(ps, abs_err_sum_means, "s-", color="#C0504D", label="|誤差|(和近似)")
    axes[1].plot(ps, abs_err_min_means, "^-", color="#9BBB59", label="|誤差|(min近似)")
    axes[1].set_xlabel("結合強度 p (0=完全モジュール)")
    axes[1].set_ylabel("|Φ_whole - Φ_approx| (平均絶対誤差)")
    axes[1].set_title("結合強度と近似誤差の関係")
    axes[1].legend(fontsize=8)

    fig.suptitle("要件5: モジュール構造を仮定したΦの階層的近似")
    fig.tight_layout()
    fig.savefig("iit_phi_modularity_comparison.png", dpi=150)
    print("グラフを iit_phi_modularity_comparison.png に保存しました。")


if __name__ == "__main__":
    if sys.argv[1] == "aggregate":
        aggregate()
    else:
        run_p(float(sys.argv[2]))
