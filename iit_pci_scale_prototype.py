"""
感情AIプロジェクト フェーズ5 プロトタイプ: PCI(擾乱複雑性指標)の規模検証(要件5)
==========================================================

代数的連結度について行った規模検証(iit_connectivity_scale_prototype.py、
Part1: 5〜6ノードでの複雑な構造での厳密Φとの一致確認、Part2: N=10〜500での
大規模スケーリング)と同じ2部構成を、PCI本実装(iit_pci_gwt_prototype.py)
についても行う。PCIは摂動ごとに時系列応答をLempel-Ziv圧縮率で評価する必要が
あるため、単純な固有値計算である代数的連結度より計算コストが高い。この
コスト自体も定量的に確認する。

【Part1: 6ノード・3クラスター構造での厳密Φ vs PCIの判定一致検証】
iit_connectivity_scale_prototype.pyのPart1と全く同じ6ノード3クラスター
ネットワーク(A,B,C=ANDクラスター、D,E=フリップフロップペア、F=単独ノード)
・全く同じ3設定(完全分断p1=0,p2=0/非対称p1=0.3,p2=0/対称中程度結合
p1=0.3,p2=0.3)を使う。既存の厳密Φ・代数的連結度の結果(iit_conn_part1_
summary.json、Task#36-38で確定済み)と、新たに計算するPCIを並べて比較する。

【Part2: N=10〜500での大規模スケーリング】
iit_connectivity_scale_prototype.pyのPart2と同じ2種類の構造(鎖状+ランダム辺
=明確に統合、k個の独立クリック=明確に分断)を再利用する。ただしPCIには
「状態遷移関数」が必要なため、既存の重み付きグラフWを使った単純な拡散
(voter model)動態を新たに定義する: P(node_i'=1) = (Wの行iに沿った隣接
ノードの現在状態の重み付き平均)。これにより、分断グラフでは摂動の影響が
そのクラスター内に留まり、統合グラフでは全体に伝播しうるという、代数的
連結度の検証と同じ論理構造をPCIでも再現できる。

**計算コストの制約への対応**: PCIは摂動ごとにLempel-Ziv圧縮を計算する必要が
あり、N=500全ノードへの摂動を網羅すると(N×試行回数×状態数)のLZ計算が
必要になり非現実的。そこで摂動対象ノードは全数ではなく固定数(min(N,10))を
サンプリングし、この設計変更を明記した上で計算時間を測定する。

【Part3(可能であれば): より大きな規模での山型再現】
既存の5ノード(3+2クラスター、結合強度pスイープ)モチーフを、独立した
k個のコピーとして複製したN=5k ノードのネットワークを作り(各コピーは互いに
無関係、同じpを共有)、pスイープに対してPCIが同じ山型を示すか確認する。
これは「他の多数の無関係なコピーが背景ノイズとして存在する中でも、PCIが
局所的な山型シグナルを検出できるか」という追加の頑健性検証になる(厳密Φは
この規模では計算不能なため比較対象はPCI単独の内的整合性)。

使い方:
  python3 iit_pci_scale_prototype.py part1
  python3 iit_pci_scale_prototype.py part2
  python3 iit_pci_scale_prototype.py part3
"""

import sys, json, time, random
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
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


# ============================================================
# 共通ユーティリティ(iit_pci_gwt_prototype.pyと同一実装、依存を避けるため複製)
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


def sample_with_crn(probs, rand_draws):
    return tuple(1 if rand_draws[i] < probs[i] else 0 for i in range(len(probs)))


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


def pci_full(update_fn, n, candidate_states, perturb_indices=None, T=15, n_trials=20, seed=0):
    rng = random.Random(seed)
    perturb_indices = list(range(n)) if perturb_indices is None else list(perturb_indices)
    state_scores = []
    for start_state in candidate_states:
        node_scores = [pci_score_for_perturbation(update_fn, n, start_state, i, T, n_trials, rng) for i in perturb_indices]
        state_scores.append(max(node_scores))
    return float(np.mean(state_scores)), state_scores


def algebraic_connectivity(W):
    Wz = W.copy()
    np.fill_diagonal(Wz, 0.0)
    Wsym = (Wz + Wz.T) / 2.0
    deg = np.diag(Wsym.sum(axis=1))
    L = deg - Wsym
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


# ============================================================
# Part1: 6ノード・3クラスター構造(iit_connectivity_scale_prototype.pyと同一)
# ============================================================

N1 = 6
NODE_LABELS_1 = ("A", "B", "C", "D", "E", "F")

PART1_CONFIGS = [
    {"name": "完全分断(3クラスター孤立)", "p1": 0.0, "p2": 0.0},
    {"name": "非対称(片方は孤立のまま)", "p1": 0.3, "p2": 0.0},
    {"name": "対称・中程度結合(全クラスター連結)", "p1": 0.3, "p2": 0.3},
]

CANDIDATE_STATES_6 = [
    (0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1), (1, 0, 1, 0, 1, 0),
    (0, 1, 0, 1, 0, 1), (1, 1, 0, 0, 1, 0), (1, 0, 0, 0, 0, 0),
]


def make_update_fn_part1(p1, p2):
    def fn(state):
        a, b, c, d, e, f = state
        pB = float(a and c)
        pC = float(a and b)
        pA = (1 - p1 - p2) * float(b and c) + p1 * float(d) + p2 * float(f)
        pE = float(d)
        pD = (1 - p1) * float(e) + p1 * float(a)
        pF = (1 - p2) * float(f) + p2 * float(a)
        return [pA, pB, pC, pD, pE, pF]
    return fn


def part1():
    # 既存の厳密Φ・代数的連結度の結果(Task#36-38で確定済み)を再利用
    try:
        with open("iit_conn_part1_summary.json") as f:
            existing = {row["name"]: row for row in json.load(f)}
    except FileNotFoundError:
        existing = {}

    rows = []
    print("=== Part1: 3クラスター構造(6ノード)での厳密Φ vs 代数的連結度 vs PCI ===")
    for cfg in PART1_CONFIGS:
        p1, p2 = cfg["p1"], cfg["p2"]
        update_fn = make_update_fn_part1(p1, p2)
        pci_mean, state_scores = pci_full(update_fn, N1, CANDIDATE_STATES_6, seed=int(p1 * 1000 + p2 * 100))
        ex = existing.get(cfg["name"], {})
        row = {
            "name": cfg["name"], "p1": p1, "p2": p2,
            "phi_mean": ex.get("phi_mean"), "phi_judgment": ex.get("phi_judgment"),
            "fiedler": ex.get("fiedler"), "fiedler_judgment": ex.get("fiedler_judgment"),
            "pci_mean": pci_mean, "pci_state_scores": state_scores,
        }
        rows.append(row)
        print(f"[{cfg['name']}] p1={p1}, p2={p2}: Φ平均={ex.get('phi_mean')}(判定={ex.get('phi_judgment')}) | "
              f"Fiedler={ex.get('fiedler')}(判定={ex.get('fiedler_judgment')}) | PCI平均={pci_mean:.4f}")

    with open("iit_pci_scale_part1_results.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("saved iit_pci_scale_part1_results.json")

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["name"] for r in rows]
    x = np.arange(len(names))
    width = 0.25
    phi_vals = [r["phi_mean"] or 0 for r in rows]
    fiedler_vals = [r["fiedler"] or 0 for r in rows]
    pci_vals = [r["pci_mean"] for r in rows]
    ax.bar(x - width, phi_vals, width, label="厳密Φ(平均)", color="#4472C4")
    ax.bar(x, fiedler_vals, width, label="代数的連結度", color="#C0504D")
    ax.bar(x + width, pci_vals, width, label="PCI本実装(平均)", color="#2E7D32")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    ax.set_title("Part1: 3クラスター構造(6ノード)でのΦ・代数的連結度・PCI")
    ax.legend()
    fig.tight_layout()
    fig.savefig("iit_pci_scale_part1_comparison.png", dpi=150)
    print("グラフを iit_pci_scale_part1_comparison.png に保存しました。")


# ============================================================
# Part2: N=10〜500での大規模スケーリング
# ============================================================

def make_disconnected_graph(n_total, n_clusters, rng):
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
    W = np.zeros((n_total, n_total))
    for i in range(n_total - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0
    for i in range(n_total):
        for j in range(i + 2, n_total):
            if rng.random() < extra_edge_prob:
                W[i, j] = W[j, i] = 1.0
    return W


def make_diffusion_update_fn(W):
    """重み付きグラフWに基づく単純な拡散(voter model)動態:
    P(node_i'=1) = 隣接ノードの現在状態の重み付き平均(孤立ノードは自身の状態を維持)。"""
    deg = W.sum(axis=1)

    def fn(state):
        s = np.array(state, dtype=float)
        num = W @ s
        probs = np.where(deg > 0, num / np.where(deg > 0, deg, 1.0), s)
        return probs.tolist()
    return fn


def part2(Ns=(10, 20, 50, 100, 200, 500), n_perturb_sample=5, n_trials=3, T=10, n_states=1):
    results = []
    print("=== Part2: 大規模(N=10〜500)でのPCIのスケーリング検証 ===")
    print(f"設計上の制約: 摂動対象ノードは全数ではなく最大{n_perturb_sample}個をサンプリング、"
          f"n_trials={n_trials}, T={T}ステップ、初期状態数={n_states}")
    for n in Ns:
        rng_np = np.random.default_rng(n)
        rng_py = random.Random(n)

        candidate_states = [tuple(rng_py.randint(0, 1) for _ in range(n)) for _ in range(n_states)]
        perturb_sample = sorted(rng_py.sample(range(n), min(n_perturb_sample, n)))

        # (a) 明確に統合されたネットワーク
        W_conn = make_connected_graph(n, rng_np)
        fiedler_conn = algebraic_connectivity(W_conn)
        update_conn = make_diffusion_update_fn(W_conn)
        t0 = time.time()
        pci_conn, _ = pci_full(update_conn, n, candidate_states, perturb_indices=perturb_sample,
                                T=T, n_trials=n_trials, seed=n)
        t_conn = time.time() - t0

        # (b) 明確に分断されたネットワーク(3クラスターのクリック非交和)
        W_disc = make_disconnected_graph(n, 3, rng_np)
        fiedler_disc = algebraic_connectivity(W_disc)
        update_disc = make_diffusion_update_fn(W_disc)
        t0 = time.time()
        pci_disc, _ = pci_full(update_disc, n, candidate_states, perturb_indices=perturb_sample,
                                T=T, n_trials=n_trials, seed=n + 1)
        t_disc = time.time() - t0

        results.append({
            "n": n,
            "fiedler_connected": fiedler_conn, "fiedler_disconnected": fiedler_disc,
            "pci_connected": pci_conn, "pci_disconnected": pci_disc,
            "seconds_connected": t_conn, "seconds_disconnected": t_disc,
        })
        print(f"N={n}: 統合グラフ Fiedler={fiedler_conn:.5f}, PCI={pci_conn:.4f}({t_conn*1000:.1f}ms) | "
              f"分断グラフ Fiedler={fiedler_disc:.5f}, PCI={pci_disc:.4f}({t_disc*1000:.1f}ms)")

    with open("iit_pci_scale_part2_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved iit_pci_scale_part2_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ns = [r["n"] for r in results]
    axes[0].plot(ns, [r["pci_connected"] for r in results], "o-", color="#4472C4", label="明確に統合(鎖状+ランダム辺)")
    axes[0].plot(ns, [r["pci_disconnected"] for r in results], "s-", color="#C0504D", label="明確に分断(3クリック非交和)")
    axes[0].set_xlabel("ノード数 N")
    axes[0].set_ylabel("PCI本実装(平均)")
    axes[0].set_title("規模とPCIの判定")
    axes[0].legend(fontsize=8)

    axes[1].plot(ns, [r["seconds_connected"] * 1000 for r in results], "o-", color="#4472C4", label="統合グラフ")
    axes[1].plot(ns, [r["seconds_disconnected"] * 1000 for r in results], "s-", color="#C0504D", label="分断グラフ")
    axes[1].set_xlabel("ノード数 N")
    axes[1].set_ylabel("計算時間(ミリ秒)")
    axes[1].set_title("PCIの計算コストのスケーリング(摂動対象を最大10個に制限)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Part2: 大規模(N=10〜500)でのPCIの実用性")
    fig.tight_layout()
    fig.savefig("iit_pci_scale_part2_scaling.png", dpi=150)
    print("グラフを iit_pci_scale_part2_scaling.png に保存しました。")


# ============================================================
# Part3(可能であれば): 独立コピー複製によるより大きな規模での山型再現
# ============================================================

EXISTING_PHI = {
    0.0: 0.0000, 0.1: 0.2174, 0.2: 0.3714, 0.3: 0.4368, 0.4: 0.4379,
    0.5: 0.3860, 0.6: 0.3772, 0.7: 0.1992, 1.0: 0.0000,
}
P_VALUES = sorted(EXISTING_PHI.keys())


def make_multi_motif_update_fn(k_copies, p):
    """既存の5ノードモチーフ(3+2クラスター、結合強度p)をk_copies個、互いに無関係に
    複製したN=5*k_copiesノードのネットワーク。各コピーは他のコピーの状態に一切
    依存しない(独立)。摂動を受けたコピー以外は「背景ノイズ」として振る舞う。"""
    n = 5 * k_copies

    def fn(state):
        probs = [0.0] * n
        for c in range(k_copies):
            off = 5 * c
            a, b, cc, d, e = state[off:off + 5]
            pB = float(a and cc)
            pC = float(a and b)
            pA = (1 - p) * float(b and cc) + p * float(d)
            pE = float(d)
            pD = (1 - p) * float(e) + p * float(a)
            probs[off:off + 5] = [pA, pB, pC, pD, pE]
        return probs
    return fn


def part3(k_copies_list=(5, 10)):
    print("=== Part3(可能であれば): 独立コピー複製によるより大きな規模での山型再現 ===")
    print("(厳密Φはこの規模では計算不能なため、PCI自身が既存5ノードでの山型形状を"
          "背景ノイズ(他コピー)の中でも検出できるかという内的整合性の検証)")
    results = {}
    for k in k_copies_list:
        n = 5 * k
        rng_py = random.Random(k)
        # 摂動はコピー0の5ノードのみに与える(他コピーは常に背景ノイズとして存在)
        perturb_sample = list(range(5))
        # 初期状態: 各コピーは独立にランダムな0/1状態(計算コストの制約により1状態のみ)
        candidate_states = [tuple(rng_py.randint(0, 1) for _ in range(n)) for _ in range(1)]

        pci_by_p = {}
        t0 = time.time()
        for p in P_VALUES:
            update_fn = make_multi_motif_update_fn(k, p)
            pci_mean, _ = pci_full(update_fn, n, candidate_states, perturb_indices=perturb_sample,
                                    T=10, n_trials=5, seed=int(p * 1000) + k)
            pci_by_p[p] = pci_mean
        dt = time.time() - t0

        peak_p = P_VALUES[int(np.argmax([pci_by_p[p] for p in P_VALUES]))]
        print(f"k_copies={k}(N={n}): 計算時間={dt:.1f}s, PCIのピーク位置=p{peak_p}(単体5ノードでのΦのピークはp0.4)")
        for p in P_VALUES:
            print(f"  p={p}: PCI={pci_by_p[p]:.4f}")
        results[k] = {"n": n, "pci_by_p": pci_by_p, "peak_p": peak_p, "seconds": dt}

    with open("iit_pci_scale_part3_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved iit_pci_scale_part3_results.json")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#4472C4", "#C0504D", "#2E7D32"]
    # 単体5ノード(既存結果、iit_pci_gwt_results.jsonから)も比較のため重ねる
    try:
        base = json.load(open("iit_pci_gwt_results.json"))
        base_p = base["setB"]["p_values"]
        base_pci = base["setB"]["pci"]
        ax.plot(base_p, base_pci, "o-", color="#808080", label="N=5(単体、既存結果)")
    except FileNotFoundError:
        pass
    for idx, k in enumerate(k_copies_list):
        r = results[k]
        ax.plot(P_VALUES, [r["pci_by_p"][p] for p in P_VALUES], "s--", color=colors[idx % len(colors)],
                label=f"N={r['n']}(k={k}コピー、背景ノイズあり)")
    ax.axvline(0.4, color="gray", linestyle=":", alpha=0.6, label="単体5ノードでのΦのピーク(p=0.4)")
    ax.set_xlabel("結合強度 p")
    ax.set_ylabel("PCI本実装(平均)")
    ax.set_title("Part3: 背景ノイズを伴うより大きな規模でのPCIの山型再現")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("iit_pci_scale_part3_comparison.png", dpi=150)
    print("グラフを iit_pci_scale_part3_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "part1":
        part1()
    elif cmd == "part2":
        part2()
    elif cmd == "part3":
        part3()
