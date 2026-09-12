"""
感情AIプロジェクト フェーズ5 プロトタイプ: 代数的連結度の汎化検証(要件5)
==========================================================

前回(iit_alt_metrics_prototype.py)は、前向き型vs再帰型・2クラスターの
モジュール性スイープという単純なケースで、代数的連結度が厳密Φの定性的な
区別(統合 vs 分断)を再現し、階層近似が陥った「p=0での罠」も回避することを
確認した。本プロトタイプはこれが「2クラスターの単純なケースだけの偶然」
ではないかを確かめるため、2部構成で検証する。

【事前の予備調査で判明した重要な事実】
pyphiの厳密Φ計算は、ユーザーが想定していた10〜12ノード程度までは実測上
到達できなかった。予備テストの結果:
  - 疎結合な3クラスター(コピーペアの鎖、AND橋渡し)6ノード: 全30到達可能状態が
    0.04秒で計算完了。ただし全状態でΦ=0となり、統合ネットワークの検証には使えない
    (決定論的なAND橋渡しが常に無料のカット(自由分割)を許してしまうため)。
  - 密結合な2つのANDクラスター(3+3ノード)を橋渡しした6ノード: 1状態の計算だけで
    40秒のタイムアウトを超過(計算不能)。
  - ANDクラスター(3ノード)+フリップフロップ三つ組(3ノード)の6ノード: 1状態
    あたり約6秒で計算可能(45秒の壁内で3〜4状態が限界)。
  - ANDクラスター(3ノード)+フリップフロップペア(2ノード)+橋渡し先の
    さらなるANDクラスター、という7ノード構成: 1状態も45秒以内に計算不能。
本検証はこの実測結果を踏まえ、Part1は「厳密計算できる現実的な上限」である
5〜6ノードの範囲で、これまでより複雑な構造(3クラスター・非対称結合・
階層構造)を作って検証する。10〜12ノードは断念し、この経緯自体を結果として
報告する。

【Part1: 5〜6ノードでの複雑な構造における厳密Φ vs 代数的連結度の一致検証】
ネットワーク: A,B,C(ANDクラスター、三角結合)+ D,E(フリップフロップペア)+
F(単独ノード、A方向へ弱く追従)の6ノード、3クラスター構成。
  P(A'=1) = (1-p1-p2)*(B AND C) + p1*D + p2*F
  P(B'=1) = A AND C
  P(C'=1) = A AND B
  P(D'=1) = (1-p1)*E + p1*A
  P(E'=1) = D
  P(F'=1) = (1-p2)*F + p2*A
p1: クラスター1<->クラスター2の橋渡し強度、p2: クラスター1<->クラスター3(単独ノード)
の橋渡し強度。p1, p2を独立に振ることで、非対称な結合強度・部分的な分断
(例: p1>0だがp2=0で、F だけが完全に孤立)を作れる。

各設定について、重み付きグラフ(結合行列に強度p1, p2, 三角結合=1, D-E結合=1を
重みとして与えたもの)から代数的連結度(Fiedler値)を計算し、厳密Φ(いくつかの
到達可能な状態)と比べて、「統合されている(Φ>0 かつ Fiedler>0)」
「分断されている(Φ=0 かつ Fiedler=0)」の定性的判定が一致するかを確認する。

【Part2: 厳密Φが不可能な数十〜数百ノード規模での代数的連結度の挙動】
構造があらかじめ分かっている2種類のネットワーク:
  (a) 「明確に統合」: ランダム重み付き連結グラフ(Erdos-Renyiでp十分大きく、
      連結性を保証)、または全ノードを鎖状+ランダム追加辺で繋いだもの。
  (b) 「明確に分断」: k個の独立したクリック(完全グラフ)の非交和(クラスター間の
      辺は一切なし)。
ノード数Nを10〜500まで振り、代数的連結度がPure Python/numpyの
np.linalg.eigvalsh でどの程度の時間で計算できるか、(a)(b)の判定
(Fiedler>0 vs Fiedler=0)が常に正しいかを確認する。厳密Φは計算しない
(この規模では原理的に不可能なため、比較対象ではなく「代数的連結度が
実用的な代理指標として機能するスケール」を単独で確認する)。

使い方:
  python3 iit_connectivity_scale_prototype.py part1_run <p1> <p2> <n_states>
  python3 iit_connectivity_scale_prototype.py part1_aggregate
  python3 iit_connectivity_scale_prototype.py part2_run
"""

import sys, json, time, itertools
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
# Part 1: 5〜6ノード、3クラスター構造での厳密Φ vs 代数的連結度
# ============================================================

N1 = 6
NODE_LABELS_1 = ("A", "B", "C", "D", "E", "F")


def make_tpm_part1(p1, p2):
    states = list(pyphi.utils.all_states(N1))
    tpm = np.zeros((2 ** N1, N1))
    for i, s in enumerate(states):
        a, b, c, d, e, f = s
        pB = float(a and c)
        pC = float(a and b)
        pA = (1 - p1 - p2) * float(b and c) + p1 * float(d) + p2 * float(f)
        pE = float(d)
        pD = (1 - p1) * float(e) + p1 * float(a)
        pF = (1 - p2) * float(f) + p2 * float(a)
        tpm[i] = [pA, pB, pC, pD, pE, pF]
    return tpm


def make_cm_part1(p1, p2):
    bridge1 = 1 if p1 > 0 else 0
    bridge2 = 1 if p2 > 0 else 0
    # 行=影響を与える側, 列=影響を受ける側
    return np.array([
        [0, 1, 1, bridge1, 0, bridge2],  # A -> B, C, D, F
        [0, 0, 1, 0, 0, 0],              # B -> C
        [1, 1, 0, 0, 0, 0],              # C -> B (A,Bと合わせcの依存にも使う。cmはゆるい上界でよい)
        [bridge1, 0, 0, 0, 1, 0],        # D -> A, E
        [0, 0, 0, 1, 0, 0],              # E -> D
        [bridge2, 0, 0, 0, 0, 1],        # F -> A, F(自己)
    ])


def weighted_adj_part1(p1, p2):
    """代数的連結度計算用の重み付き隣接行列(対称化前)。
    三角結合(A-B,B-C,C-A)は重み1、D-E結合は重み1、橋渡しはp1(A-D), p2(A-F)。"""
    W = np.zeros((N1, N1))
    W[0, 1] = W[1, 0] = 1.0  # A-B
    W[1, 2] = W[2, 1] = 1.0  # B-C
    W[0, 2] = W[2, 0] = 1.0  # A-C
    W[3, 4] = W[4, 3] = 1.0  # D-E
    W[0, 3] = W[3, 0] = p1   # A-D bridge
    W[0, 5] = W[5, 0] = p2   # A-F bridge
    return W


def algebraic_connectivity(W):
    Wz = W.copy()
    np.fill_diagonal(Wz, 0.0)
    Wsym = (Wz + Wz.T) / 2.0
    deg = np.diag(Wsym.sum(axis=1))
    L = deg - Wsym
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


def part1_run(p1, p2, n_states):
    tpm = make_tpm_part1(p1, p2)
    cm = make_cm_part1(p1, p2)
    network = pyphi.Network(tpm, cm=cm, node_labels=NODE_LABELS_1)

    results = []
    count = 0
    t_start = time.time()
    for state in pyphi.utils.all_states(N1):
        if count >= n_states:
            break
        if time.time() - t_start > 38:
            print(f"打ち切り(時間切れ): {count}/{n_states} 状態で終了")
            break
        try:
            sub = pyphi.Subsystem(network, state)
        except pyphi.exceptions.StateUnreachableError:
            continue
        t0 = time.time()
        sia = pyphi.compute.sia(sub)
        phi = float(sia.phi)
        dt = time.time() - t0
        results.append({"p1": p1, "p2": p2, "state": state, "phi": phi, "seconds": dt})
        print(f"p1={p1} p2={p2} state={state}: phi={phi:.4f} ({dt:.2f}s)")
        count += 1

    W = weighted_adj_part1(p1, p2)
    fiedler = algebraic_connectivity(W)
    print(f"代数的連結度(Fiedler値) = {fiedler:.4f}")

    fname = f"iit_conn_part1_p1{str(p1).replace('.', '_')}_p2{str(p2).replace('.', '_')}.json"
    try:
        with open(fname) as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {"p1": p1, "p2": p2, "fiedler": fiedler, "phi_results": []}
    existing["fiedler"] = fiedler
    existing_states = {tuple(r["state"]) for r in existing["phi_results"]}
    for r in results:
        if tuple(r["state"]) not in existing_states:
            existing["phi_results"].append(r)
    with open(fname, "w") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"saved {fname} (累計 {len(existing['phi_results'])} 状態)")


PART1_CONFIGS = [
    {"name": "完全分断(3クラスター孤立)", "p1": 0.0, "p2": 0.0},
    # p1=0.3, p2=0.05(片方だけ弱く接続)は1状態の計算が45秒を超え断念。
    # 実測上、橋渡しがどちらか一方でもゼロでない限り事実上「完全連結」に近い
    # 計算コストがかかることが判明したため、このケースは実装記録.mdに
    # 「試みたが計算不能だった」旨を明記する形で扱う。
    {"name": "非対称(片方は孤立のまま)", "p1": 0.3, "p2": 0.0},
    {"name": "対称・中程度結合(全クラスター連結)", "p1": 0.3, "p2": 0.3},
]


def part1_aggregate():
    print("=== Part1: 3クラスター構造での厳密Φ vs 代数的連結度の判定一致 ===")
    rows = []
    for cfg in PART1_CONFIGS:
        p1, p2 = cfg["p1"], cfg["p2"]
        fname = f"iit_conn_part1_p1{str(p1).replace('.', '_')}_p2{str(p2).replace('.', '_')}.json"
        with open(fname) as f:
            data = json.load(f)
        phis = [r["phi"] for r in data["phi_results"]]
        fiedler = data["fiedler"]
        phi_mean = float(np.mean(phis)) if phis else None
        phi_judgment = "統合" if (phi_mean is not None and phi_mean > 1e-9) else "分断(自由カット)"
        fiedler_judgment = "統合" if fiedler > 1e-9 else "分断"
        agree = phi_judgment.startswith(fiedler_judgment[:2]) or (phi_judgment == "分断(自由カット)" and fiedler_judgment == "分断")
        rows.append({
            "name": cfg["name"], "p1": p1, "p2": p2, "n_states": len(phis),
            "phi_mean": phi_mean, "fiedler": fiedler,
            "phi_judgment": phi_judgment, "fiedler_judgment": fiedler_judgment,
            "agree": bool(agree),
        })
        print(f"[{cfg['name']}] p1={p1}, p2={p2}: Φ平均={phi_mean}({len(phis)}状態), "
              f"判定={phi_judgment} | Fiedler={fiedler:.4f}, 判定={fiedler_judgment} | 一致={agree}")

    with open("iit_conn_part1_summary.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    phi_vals = [r["phi_mean"] or 0 for r in rows]
    fiedler_vals = [r["fiedler"] for r in rows]
    width = 0.35
    ax.bar(x - width / 2, phi_vals, width, label="厳密Φ(平均)", color="#4472C4")
    ax.bar(x + width / 2, fiedler_vals, width, label="代数的連結度", color="#C0504D")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_title("Part1: 3クラスター構造(6ノード)でのΦと代数的連結度")
    ax.legend()
    fig.tight_layout()
    fig.savefig("iit_conn_part1_comparison.png", dpi=150)
    print("グラフを iit_conn_part1_comparison.png に保存しました。")


# ============================================================
# Part 2: 数十〜数百ノードでの代数的連結度のスケーラビリティ
# ============================================================

def make_disconnected_graph(n_total, n_clusters, rng):
    """n_clusters個の独立したクリック(完全グラフ)の非交和。クラスター間の辺は無し。"""
    W = np.zeros((n_total, n_total))
    base = n_total // n_clusters
    sizes = [base] * n_clusters
    sizes[-1] += n_total - base * n_clusters
    idx = 0
    for size in sizes:
        members = list(range(idx, idx + size))
        for i in members:
            for j in members:
                if i != j:
                    W[i, j] = 1.0
        idx += size
    return W


def make_connected_graph(n_total, rng, extra_edge_prob=0.05):
    """鎖状(0-1-2-...-N-1)で連結性を保証した上で、ランダムな追加辺を加えた連結グラフ。"""
    W = np.zeros((n_total, n_total))
    for i in range(n_total - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    for i in range(n_total):
        for j in range(i + 2, n_total):
            if rng.random() < extra_edge_prob:
                W[i, j] = W[j, i] = 1.0
    return W


def part2_run():
    rng = np.random.default_rng(0)
    Ns = [10, 20, 50, 100, 200, 500]
    results = []
    for n in Ns:
        # (a) 明確に統合されたネットワーク
        t0 = time.time()
        W_conn = make_connected_graph(n, rng)
        fiedler_conn = algebraic_connectivity(W_conn)
        t_conn = time.time() - t0

        # (b) 明確に分断されたネットワーク(3クラスターのクリック非交和)
        t0 = time.time()
        W_disc = make_disconnected_graph(n, 3, rng)
        fiedler_disc = algebraic_connectivity(W_disc)
        t_disc = time.time() - t0

        judgment_conn = "統合" if fiedler_conn > 1e-9 else "分断"
        judgment_disc = "統合" if fiedler_disc > 1e-9 else "分断"
        correct_conn = judgment_conn == "統合"
        correct_disc = judgment_disc == "分断"

        results.append({
            "n": n, "fiedler_connected": fiedler_conn, "seconds_connected": t_conn,
            "fiedler_disconnected": fiedler_disc, "seconds_disconnected": t_disc,
            "judgment_connected": judgment_conn, "judgment_disconnected": judgment_disc,
            "correct_connected": correct_conn, "correct_disconnected": correct_disc,
        })
        print(f"N={n}: 統合グラフ Fiedler={fiedler_conn:.5f}({t_conn*1000:.2f}ms, 判定={judgment_conn}, 正解={correct_conn}) | "
              f"分断グラフ Fiedler={fiedler_disc:.5f}({t_disc*1000:.2f}ms, 判定={judgment_disc}, 正解={correct_disc})")

    with open("iit_conn_part2_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ns = [r["n"] for r in results]
    axes[0].plot(ns, [r["fiedler_connected"] for r in results], "o-", color="#4472C4", label="明確に統合(鎖状+ランダム辺)")
    axes[0].plot(ns, [r["fiedler_disconnected"] for r in results], "s-", color="#C0504D", label="明確に分断(3クリック非交和)")
    axes[0].set_xlabel("ノード数 N")
    axes[0].set_ylabel("代数的連結度(Fiedler値)")
    axes[0].set_title("規模と代数的連結度の判定")
    axes[0].legend(fontsize=8)

    axes[1].plot(ns, [r["seconds_connected"] * 1000 for r in results], "o-", color="#4472C4", label="統合グラフ")
    axes[1].plot(ns, [r["seconds_disconnected"] * 1000 for r in results], "s-", color="#C0504D", label="分断グラフ")
    axes[1].set_xlabel("ノード数 N")
    axes[1].set_ylabel("計算時間(ミリ秒)")
    axes[1].set_title("計算コストのスケーリング")
    axes[1].legend(fontsize=8)

    fig.suptitle("Part2: 大規模(N=10〜500)での代数的連結度の実用性")
    fig.tight_layout()
    fig.savefig("iit_conn_part2_scaling.png", dpi=150)
    print("グラフを iit_conn_part2_scaling.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "part1_run":
        part1_run(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "part1_aggregate":
        part1_aggregate()
    elif cmd == "part2_run":
        part2_run()
