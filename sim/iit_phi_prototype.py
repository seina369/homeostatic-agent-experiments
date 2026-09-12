"""
感情AIプロジェクト フェーズ5 プロトタイプ: 高統合アーキテクチャ(要件5、IITのΦ)
==========================================================

要件5は「現実的な規模での実装は本質的に困難」とされている要件であり、計画書は
IIT(統合情報理論)の予測として「前向き型・フォン・ノイマン型のアーキテクチャは、
どれほど複雑な振る舞いをしてもΦ(統合情報量)が低くなる」を引用している。本
プロトタイプは、この予測がごく小さなトイモデルで実際に成り立つかどうかを、
pyphi(IITのΦを厳密に計算するPythonパッケージ)で検証する、原理の最小限の
デモンストレーションである。

4ノード・2値のネットワークを2種類用意する。CとDの更新関数(C=A AND B、D=C)は
共通にしたまま、AとBの更新関数だけを変えることで「同程度の入出力ふるまい」
(C・Dが計算する関数は文字通り同一)を保ちつつ、ネットワーク全体の因果構造を
前向き型/再帰型で作り分けた。

  (1) 前向き型(FF): A' = A, B' = B (どちらも自己保持のみ), C' = A AND B,
      D' = C。サイクルを持たない純粋なDAG。
  (2) 再帰型(REC): A' = (D AND B) OR (NOT D AND A), B' = (C AND A) OR
      (NOT C AND B), C' = A AND B(FFと同一), D' = C(FFと同一)。
      A→C→D→A、B→C→B という2つのフィードバックループを持つ。

両ネットワークについて、全16状態(2^4)でΦを厳密計算し、平均・分布を比較する。

使い方:
  python3 iit_phi_prototype.py
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

NODE_LABELS = ("A", "B", "C", "D")
N = 4


def build_tpm(update_fn):
    """pyphiの状態順序(pyphi.utils.all_states)に沿って、状態ごとの次状態を
    update_fnで計算し、state-by-nodeのTPM(2^N x N)を作る。"""
    states = list(pyphi.utils.all_states(N))
    tpm = np.zeros((2 ** N, N))
    for idx, state in enumerate(states):
        next_state = update_fn(state)
        tpm[idx] = next_state
    return tpm


def ff_update(state):
    a, b, c, d = state
    a_next = a
    b_next = b
    c_next = int(a and b)
    d_next = c
    return (a_next, b_next, c_next, d_next)


def rec_update(state):
    a, b, c, d = state
    a_next = int((d and b) or ((not d) and a))
    b_next = int((c and a) or ((not c) and b))
    c_next = int(a and b)  # FFと同一の関数
    d_next = c             # FFと同一の関数
    return (a_next, b_next, c_next, d_next)


# 前向き型: A,B(自己保持のみ、外部からの入力を受けない)-> C(AND) -> D(コピー)
FF_CM = np.array([
    [1, 0, 1, 0],  # A -> A(自己), A -> C
    [0, 1, 1, 0],  # B -> B(自己), B -> C
    [0, 0, 0, 1],  # C -> D
    [0, 0, 0, 0],  # D -> (なし)
])

# 再帰型: A<-D, B<-C という2本のフィードバックを追加(A->C->D->A, B->C->B)
REC_CM = np.array([
    [1, 0, 1, 0],  # A -> A(自己の一部), A -> C
    [0, 1, 1, 0],  # B -> B(自己の一部), B -> C
    [0, 1, 0, 1],  # C -> B, C -> D
    [1, 0, 0, 0],  # D -> A
])


def compute_phi_for_all_states(network, label):
    """TPM上到達不能な状態(決定論的ネットワークでは、直前状態が存在しない状態)は
    pyphiがStateUnreachableErrorを送出するため、到達可能な状態のみを対象にΦを
    計算する(un reachable=N/Aとして記録し、平均等の集計からは除外する)。"""
    results = []
    states = list(pyphi.utils.all_states(N))
    for state in states:
        t0 = time.time()
        try:
            subsystem = pyphi.Subsystem(network, state)
        except pyphi.exceptions.StateUnreachableError:
            print(f"[{label}] state={state}: 到達不能(N/A)")
            results.append({"state": state, "phi": None, "reachable": False, "seconds": time.time() - t0})
            continue
        try:
            sia = pyphi.compute.sia(subsystem)
            phi = float(sia.phi)
        except Exception as e:
            phi = None
            print(f"[{label}] state={state}: エラー({e})")
        dt = time.time() - t0
        results.append({"state": state, "phi": phi, "reachable": True, "seconds": dt})
        print(f"[{label}] state={state}: phi={phi}, {dt:.2f}s")
    return results


def main():
    ff_tpm = build_tpm(ff_update)
    rec_tpm = build_tpm(rec_update)

    ff_network = pyphi.Network(ff_tpm, cm=FF_CM, node_labels=NODE_LABELS)
    rec_network = pyphi.Network(rec_tpm, cm=REC_CM, node_labels=NODE_LABELS)

    print("=== 前向き型(FF)ネットワーク: 全16状態のΦ ===")
    ff_results = compute_phi_for_all_states(ff_network, "FF")
    print("\n=== 再帰型(REC)ネットワーク: 全16状態のΦ ===")
    rec_results = compute_phi_for_all_states(rec_network, "REC")

    with open("iit_phi_results.json", "w") as f:
        json.dump({"ff": ff_results, "rec": rec_results}, f, ensure_ascii=False, indent=2, default=str)

    ff_phis = [r["phi"] for r in ff_results if r["phi"] is not None]
    rec_phis = [r["phi"] for r in rec_results if r["phi"] is not None]
    print(f"\n=== 集計 ===")
    print(f"FF:  平均Φ={np.mean(ff_phis):.4f}, 最大Φ={np.max(ff_phis):.4f}, 0の状態数={sum(1 for p in ff_phis if p == 0)}/{len(ff_phis)}")
    print(f"REC: 平均Φ={np.mean(rec_phis):.4f}, 最大Φ={np.max(rec_phis):.4f}, 0の状態数={sum(1 for p in rec_phis if p == 0)}/{len(rec_phis)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [f"{r['state']}" for r in ff_results]
    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, [r["phi"] or 0 for r in ff_results], width, label="前向き型(FF)", color="#BFBFBF")
    axes[0].bar(x + width / 2, [r["phi"] or 0 for r in rec_results], width, label="再帰型(REC)", color="#4472C4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=90, fontsize=7)
    axes[0].set_ylabel("Φ")
    axes[0].set_title("状態ごとのΦ(全16状態)")
    axes[0].legend()

    axes[1].bar(["前向き型(FF)", "再帰型(REC)"], [np.mean(ff_phis), np.mean(rec_phis)],
                color=["#BFBFBF", "#4472C4"])
    axes[1].set_ylabel("平均Φ(全16状態)")
    axes[1].set_title("平均Φの比較")

    fig.suptitle("要件5: IIT Φの前向き型 vs 再帰型ネットワーク比較")
    fig.tight_layout()
    fig.savefig("iit_phi_comparison.png", dpi=150)
    print("グラフを iit_phi_comparison.png に保存しました。")


if __name__ == "__main__":
    main()
