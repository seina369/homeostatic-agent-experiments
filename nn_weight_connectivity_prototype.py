"""
感情AIプロジェクト フェーズ13 プロトタイプ(要件5・NN時代):
実在するNNの重み構造から統合度を測る
================================================================

目的: これまでのΦ代理指標(代数的連結度)はトイモデルの人工的なグラフに対して
計算していたが、これを実際に学習済みのNNエージェント(要件4・6・7で学習した
送り手・受け手・モニタ関連方策・エルダー・サクセサー)の重み構造に適用し、
現実に学習された系が「統合されているか、分断されているか」を測る。

【第0段階: 重み保存状況の確認結果】
要件4・6・7のNN実験では、専用の「最終重みファイル」は明示的には保存して
いなかった。しかし各実験は45秒のサンドボックス時間制限に対応するため
時間区切り式のpickle再開状態(state pickle)を使っており、これらの中に
学習済みエージェントオブジェクト(重み行列を含む)がそのまま保存されて
いることが確認できた。ただし重要な制約が2点ある。
  (1) これらのstate pickleは全て一時的な作業ディレクトリ(outputs)にのみ
      存在し、永続フォルダ(Emotional AI)にはコピーされていない。一時
      ディレクトリはセッション間でクリアされる可能性があるため、この
      分析は「今読み出せるうちに」実施する必要がある。
  (2) 要件4のオリジナル(共有ヘッド版)の逐次state pickleは正常に削除されて
      おり現存しない。現存するのはベース(教示前)エルダーの重みと、
      教示ヘッド分離実験(フェーズ8)で偶然生き残ったstate pickle
      (エルダー・サクセサーとも移動サブネットワークは共有ヘッド版と同一
      アーキテクチャ 9→32→32→5)のみ。よって要件4のエルダー・サクセサー
      比較は分離実験(bonus=0, bonus=3, seed=100)の重みを代用する。
      これは方法論的に完全な代替ではないが、「教示ヘッド分離後もアーキ
      テクチャ自体は移動サブネットワークに関して共有ヘッド版と同一」
      (フェーズ8参照)であるため、移動判断に関する重み構造としては
      妥当な代理と考える。

【第1段階: 代表的な学習済みネットワークの代数的連結度】
対象(全てseed=0または100、各実験の代表1系統):
  - 要件6(共同体形成、NN移行版): 送り手(NNMoveAgent, 12→32→32→6)、
    受け手(NNGuessAgent, 1→32→32→3)
  - 要件7(モニタ、NN移行版): PartA(U字型・単一マップ)、
    PartB(複数マップ+履歴8手・汎化改善)、PartC(grokking探索・8000ep)の
    各方策ネットワーク(9→32→32→N_ACTIONS)
  - 要件4(レガシー本能、NN移行版・教示ヘッド分離実験より):
    エルダー(教示前ベース)、エルダー(bonus=0教示後)、
    エルダー(bonus=3教示後)、サクセサー(bonus=0)、サクセサー(bonus=3)
    (いずれも移動サブネットワーク 9→32→32→5 のみを対象、教示ヘッドは除く)

グラフ構築法: ネットワークの各層のユニットをノードとし、隣接層間の重み
|W_ij| をエッジとする層状(bipartite層の連結)グラフを作る(バイアスは
ノード・エッジに含めない)。これまでのトイモデル実験
(iit_connectivity_scale_prototype.py)と全く同じ
algebraic_connectivity(W)関数(対称化した重み付きラプラシアンの
第2固有値=Fiedler値)をそのまま再利用する。

重みの絶対スケールは初期化・学習率・学習量によって層ごとに大きく異なり
うるため、生のFiedler値だけでなく、全結合の平均絶対値で正規化した
「構造だけを見るFiedler値」も併せて計算し、両方を報告する。

【注意】本段階はいずれもn=1(各条件代表1系統)の探索的な最初の一瞥であり、
統計的な比較ではない。既知の学習ダイナミクス(要件6の系統間ばらつきの
大きさ、要件7のA/B/Cの質的な違い、要件4フェーズ11で確認済みのbonus依存
エルダー劣化)との対応関係を探るための予備観察として位置づける。
"""

import pickle
import json
import numpy as np


# ============================================================
# __main__ クラス解決(pickle復元用)
# 各stateファイルは元スクリプトを直接 __main__ として実行して生成された
# ため、属性トランスプラント方式のpickle復元では「同名の空クラス」を
# バインドしておけば元の実装と一致していなくても状態は正しく復元される。
# ============================================================
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


# ============================================================
# 代数的連結度(過去のトイモデル実験と完全同一の実装を再利用)
# ============================================================

def algebraic_connectivity(W):
    Wz = W.copy()
    np.fill_diagonal(Wz, 0.0)
    Wsym = (Wz + Wz.T) / 2.0
    deg = np.diag(Wsym.sum(axis=1))
    L = deg - Wsym
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


def mlp_layer_graph(weight_matrices, normalize=False):
    """複数の重み行列(層ごとの2D配列のリスト)から、ユニットをノードとする
    層状グラフの重み付き隣接行列を構築する。normalize=Trueなら全エッジの
    平均絶対値で正規化してから隣接行列を作る(スケール非依存の構造比較用)。"""
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


def analyze(name, weight_matrices):
    W_raw, n = mlp_layer_graph(weight_matrices, normalize=False)
    W_norm, _ = mlp_layer_graph(weight_matrices, normalize=True)
    fc_raw = algebraic_connectivity(W_raw)
    fc_norm = algebraic_connectivity(W_norm)
    mean_abs_w = float(np.mean([np.abs(w).mean() for w in weight_matrices]))
    result = {
        "name": name,
        "n_nodes": int(n),
        "layer_sizes": [int(weight_matrices[0].shape[0])] + [int(w.shape[1]) for w in weight_matrices],
        "fiedler_raw": fc_raw,
        "fiedler_normalized": fc_norm,
        "mean_abs_weight": mean_abs_w,
    }
    print(f"{name}: nodes={n}, layers={result['layer_sizes']}, "
          f"Fiedler(raw)={fc_raw:.4f}, Fiedler(正規化)={fc_norm:.4f}, "
          f"平均|w|={mean_abs_w:.4f}")
    return result


def random_init_baseline(layer_sizes, n_seeds=10):
    """同じ層構成のランダム初期化(He初期化)ネットワークでのFiedler値分布。
    学習によって連結度がどちらの方向にずれたかを判定するための対照。"""
    raws, norms = [], []
    for seed in range(n_seeds):
        rng = np.random.RandomState(1000 + seed)
        mats = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            mats.append(rng.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / fan_in))
        W_raw, _ = mlp_layer_graph(mats, normalize=False)
        W_norm, _ = mlp_layer_graph(mats, normalize=True)
        raws.append(algebraic_connectivity(W_raw))
        norms.append(algebraic_connectivity(W_norm))
    return {
        "fiedler_raw_mean": float(np.mean(raws)), "fiedler_raw_std": float(np.std(raws)),
        "fiedler_normalized_mean": float(np.mean(norms)), "fiedler_normalized_std": float(np.std(norms)),
    }


def main():
    results = []

    # ---- 要件6: 送り手・受け手(NN移行版, seed=0) ----
    st = load_pickle("nn_comm_state_seed0.pkl")
    agent0 = st["agent0"]  # NNMoveAgent, 12->32->32->6
    guess0 = st["guess0"]  # NNGuessAgent, 1->32->32->3
    results.append(analyze("要件6_送り手(seed0)",
                            [agent0.params.W1, agent0.params.W2, agent0.params.W3]))
    results.append(analyze("要件6_受け手(seed0)",
                            [guess0.params.W1, guess0.params.W2, guess0.params.W3]))

    # ---- 要件7: モニタ関連方策 PartA/B/C(NN移行版, seed=0) ----
    for part, label in [("A", "U字型_単一マップ"), ("B", "複数マップ_汎化改善"), ("C", "grokking探索")]:
        st = load_pickle(f"nn_part{part}_state_seed0.pkl")
        agent = st["agent"]  # DQNAgent, 9->32->32->N_ACTIONS
        results.append(analyze(f"要件7_Part{part}({label}, seed0)",
                                [agent.params.W1, agent.params.W2, agent.params.W3]))

    # ---- 要件4: エルダー・サクセサー(教示ヘッド分離実験より, seed=100) ----
    st_base = load_pickle("nn_legacy_base_params.pkl")
    elder_base_params = st_base["params"]  # 教示前ベースエルダー
    results.append(analyze("要件4_エルダー(教示前ベース)",
                            [elder_base_params.W1, elder_base_params.W2, elder_base_params.W3]))

    for bonus_label, fname in [("bonus0", "nn_legacy_split_state_b0_s100.pkl"),
                                ("bonus3", "nn_legacy_split_state_b3_s100.pkl")]:
        st = load_pickle(fname)
        elder = st["elder"]
        successor = st["successor"]
        results.append(analyze(f"要件4_エルダー({bonus_label}教示後, s100)",
                                [elder.move_params.W1, elder.move_params.W2, elder.move_params.W3]))
        results.append(analyze(f"要件4_サクセサー({bonus_label}, s100)",
                                [successor.move_params.W1, successor.move_params.W2, successor.move_params.W3]))

    # ---- ランダム初期化ベースライン(層構成ごと) ----
    baselines = {}
    for shape in [[12, 32, 32, 6], [1, 32, 32, 3], [9, 32, 32, 5]]:
        key = "-".join(map(str, shape))
        baselines[key] = random_init_baseline(shape)
        print(f"ランダム初期化ベースライン{shape}: "
              f"Fiedler(raw)={baselines[key]['fiedler_raw_mean']:.4f}±{baselines[key]['fiedler_raw_std']:.4f}, "
              f"Fiedler(正規化)={baselines[key]['fiedler_normalized_mean']:.4f}±{baselines[key]['fiedler_normalized_std']:.4f}")

    with open("nn_weight_connectivity_results.json", "w", encoding="utf-8") as f:
        json.dump({"trained": results, "random_init_baselines": baselines}, f, ensure_ascii=False, indent=2)
    print("\n保存: nn_weight_connectivity_results.json")


if __name__ == "__main__":
    main()
