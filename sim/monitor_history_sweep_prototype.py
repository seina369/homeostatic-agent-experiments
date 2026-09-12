"""
感情AIプロジェクト フェーズ2〜3 プロトタイプ: 要件7 行動履歴長の深掘り
==========================================================

monitor_feature_richness_prototype.pyで、行動履歴を2手から8手に延ばすだけで
未経験マップ相関が0.063→0.189(約3倍)に改善することを確認した。本プロトタイプは
これを2方向に深掘りする。

  (1) 履歴長スイープ: 2/4/8/16/32手で未経験マップ相関がどう変わるか。8手が
      たまたま良かっただけなのか、もっと長くすればさらに改善するのか、
      あるいは8手あたりが最適点(それ以上は過学習などで悪化)なのかを見る。
  (2) 単一マップ学習 vs 複数マップ学習(いずれも履歴8手): これまでの「履歴8手で
      0.189」は複数マップ学習(seed=0,1,2,3)と組み合わせた結果だった。履歴を
      延ばすこと単体の効果と、複数マップ学習を重ねることの効果を分離するため、
      履歴8手のまま単一マップ(seed=0)だけで学習した場合の未経験マップ相関を
      測定し、複数マップ版(0.189)と比較する。

効率化のため、ロールアウトはマップごとに1回だけ行い(monitor_feature_richness_
prototype.collect_rollout_rawを再利用、MAX_HISTORYを32に拡張)、そこから複数の
履歴長・単一/複数マップの組み合わせの特徴量行列を構築して比較する。

学習系列の乱数(traj_seed=0,11,22)を変えた3系統で再現性を確認する。

使い方:
  python3 monitor_history_sweep_prototype.py <traj_seed> run
  python3 monitor_history_sweep_prototype.py aggregate
"""

import sys, json
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from homeostasis_prototype import HomeostasisEnv, QLearningAgent
import instinct_bias_prototype as ib
from monitor_maturity_prototype import fit_linear_regression, predict_linear, mean_correlation
import monitor_feature_richness_prototype as mfr

mfr.MAX_HISTORY = 32  # 履歴長スイープ(最大32手)に対応させる

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAIN_SEED = 0
MULTI_MAP_SEEDS = [0, 1, 2, 3]
UNSEEN_SEEDS = [5, 6, 7]
N_EPISODES_PER_MAP = 100
N_EPISODES_UNSEEN = 60
ROLLOUT_EPS = 0.1
HISTORY_LENGTHS = [2, 4, 8, 16, 32]
TRAJ_SEEDS = [0, 11, 22]


def run_one_seed(traj_seed):
    random.seed(traj_seed)
    np.random.seed(traj_seed)
    env = HomeostasisEnv(random.Random(TRAIN_SEED))
    agent = QLearningAgent()
    ib.train(env, agent, ib.PARENT_EPISODES, ib.PARENT_EPS_DECAY_EPISODES)
    policy_q = agent.q

    records_by_map = {}
    for seed in MULTI_MAP_SEEDS:
        random.seed(traj_seed * 1000 + seed)
        np.random.seed(traj_seed * 1000 + seed)
        map_env = HomeostasisEnv(random.Random(seed))
        records_by_map[seed] = mfr.collect_rollout_raw(map_env, dict(policy_q), N_EPISODES_PER_MAP, ROLLOUT_EPS)

    unseen_records = {}
    for seed in UNSEEN_SEEDS:
        random.seed(traj_seed * 5000 + seed)
        np.random.seed(traj_seed * 5000 + seed)
        map_env = HomeostasisEnv(random.Random(seed))
        unseen_records[seed] = mfr.collect_rollout_raw(map_env, dict(policy_q), N_EPISODES_UNSEEN, ROLLOUT_EPS)

    multi_map_records = []
    for seed in MULTI_MAP_SEEDS:
        multi_map_records.extend(records_by_map[seed])
    single_map_records = records_by_map[TRAIN_SEED]

    result = {"traj_seed": traj_seed, "sweep": {}, "single_map_hist8": {}}

    # --- (1) 履歴長スイープ(複数マップ学習、既存の標準手法) ---
    rng = np.random.RandomState(traj_seed * 999)
    perm = rng.permutation(len(multi_map_records))
    split = int(len(multi_map_records) * 0.7)
    train_idx, holdout_idx = perm[:split], perm[split:]

    for hlen in HISTORY_LENGTHS:
        X_all, Y_all = mfr.build_features(multi_map_records, hlen, False)
        W = fit_linear_regression(X_all[train_idx], Y_all[train_idx])
        corr_holdout = mean_correlation(Y_all[holdout_idx], predict_linear(X_all[holdout_idx], W))
        unseen_corrs = []
        for seed in UNSEEN_SEEDS:
            X_u, Y_u = mfr.build_features(unseen_records[seed], hlen, False)
            unseen_corrs.append(mean_correlation(Y_u, predict_linear(X_u, W)))
        result["sweep"][str(hlen)] = {
            "n_features": int(X_all.shape[1]),
            "corr_holdout": corr_holdout,
            "corr_unseen_mean": float(np.mean(unseen_corrs)),
            "corr_unseen_std": float(np.std(unseen_corrs)),
        }
        print(f"[seed={traj_seed}] 履歴長={hlen}(次元数={X_all.shape[1]}): "
              f"held-out相関={corr_holdout:.4f}, 未経験マップ相関={np.mean(unseen_corrs):.4f}±{np.std(unseen_corrs):.4f}")

    # --- (2) 履歴8手を固定し、単一マップ学習 vs 複数マップ学習を比較 ---
    rng2 = np.random.RandomState(traj_seed * 777)
    perm_s = rng2.permutation(len(single_map_records))
    split_s = int(len(single_map_records) * 0.7)
    X_all_s, Y_all_s = mfr.build_features(single_map_records, 8, False)
    W_s = fit_linear_regression(X_all_s[perm_s[:split_s]], Y_all_s[perm_s[:split_s]])
    corr_holdout_s = mean_correlation(Y_all_s[perm_s[split_s:]], predict_linear(X_all_s[perm_s[split_s:]], W_s))
    unseen_corrs_s = []
    for seed in UNSEEN_SEEDS:
        X_u, Y_u = mfr.build_features(unseen_records[seed], 8, False)
        unseen_corrs_s.append(mean_correlation(Y_u, predict_linear(X_u, W_s)))
    result["single_map_hist8"] = {
        "corr_holdout": corr_holdout_s,
        "corr_unseen_mean": float(np.mean(unseen_corrs_s)),
        "corr_unseen_std": float(np.std(unseen_corrs_s)),
    }
    print(f"[seed={traj_seed}] 履歴8・単一マップ学習: held-out相関={corr_holdout_s:.4f}, "
          f"未経験マップ相関={np.mean(unseen_corrs_s):.4f}±{np.std(unseen_corrs_s):.4f}")

    fname = f"history_sweep_seed{traj_seed}.json"
    with open(fname, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved {fname}")


def aggregate():
    data = []
    for seed in TRAJ_SEEDS:
        with open(f"history_sweep_seed{seed}.json") as f:
            data.append(json.load(f))

    print("=== (1) 履歴長スイープ(複数マップ学習、n=3の平均±標準偏差) ===")
    sweep_summary = {}
    for hlen in HISTORY_LENGTHS:
        key = str(hlen)
        holdout_vals = [d["sweep"][key]["corr_holdout"] for d in data]
        unseen_vals = [d["sweep"][key]["corr_unseen_mean"] for d in data]
        n_feat = data[0]["sweep"][key]["n_features"]
        sweep_summary[hlen] = {
            "n_features": n_feat,
            "holdout_mean": float(np.mean(holdout_vals)), "holdout_std": float(np.std(holdout_vals)),
            "unseen_mean": float(np.mean(unseen_vals)), "unseen_std": float(np.std(unseen_vals)),
        }
        print(f"履歴長={hlen}(次元数={n_feat}): held-out相関={np.mean(holdout_vals):.4f}±{np.std(holdout_vals):.4f}, "
              f"未経験マップ相関={np.mean(unseen_vals):.4f}±{np.std(unseen_vals):.4f}")

    best_hlen = max(sweep_summary, key=lambda h: sweep_summary[h]["unseen_mean"])
    print(f"未経験マップ相関が最大になる履歴長: {best_hlen} ({sweep_summary[best_hlen]['unseen_mean']:.4f})")

    print("\n=== (2) 履歴8手: 単一マップ学習 vs 複数マップ学習 ===")
    single_holdout = [d["single_map_hist8"]["corr_holdout"] for d in data]
    single_unseen = [d["single_map_hist8"]["corr_unseen_mean"] for d in data]
    multi_holdout = [d["sweep"]["8"]["corr_holdout"] for d in data]
    multi_unseen = [d["sweep"]["8"]["corr_unseen_mean"] for d in data]
    print(f"単一マップ学習: held-out相関={np.mean(single_holdout):.4f}±{np.std(single_holdout):.4f}, "
          f"未経験マップ相関={np.mean(single_unseen):.4f}±{np.std(single_unseen):.4f}")
    print(f"複数マップ学習: held-out相関={np.mean(multi_holdout):.4f}±{np.std(multi_holdout):.4f}, "
          f"未経験マップ相関={np.mean(multi_unseen):.4f}±{np.std(multi_unseen):.4f}")
    print(f"複数マップ学習による追加の改善幅(履歴8手時): {np.mean(multi_unseen) - np.mean(single_unseen):+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    hlens = HISTORY_LENGTHS
    holdout_means = [sweep_summary[h]["holdout_mean"] for h in hlens]
    holdout_stds = [sweep_summary[h]["holdout_std"] for h in hlens]
    unseen_means = [sweep_summary[h]["unseen_mean"] for h in hlens]
    unseen_stds = [sweep_summary[h]["unseen_std"] for h in hlens]
    axes[0].errorbar(hlens, holdout_means, yerr=holdout_stds, marker="o", label="held-out(学習分布内)", color="#BFBFBF")
    axes[0].errorbar(hlens, unseen_means, yerr=unseen_stds, marker="o", label="未経験マップ(真の汎化)", color="#4472C4")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(hlens)
    axes[0].set_xticklabels(hlens)
    axes[0].set_xlabel("行動履歴長")
    axes[0].set_ylabel("相関係数")
    axes[0].set_title("(1) 履歴長スイープ")
    axes[0].legend()

    labels = ["単一マップ学習", "複数マップ学習"]
    x = np.arange(2)
    width = 0.35
    axes[1].bar(x - width / 2, [np.mean(single_holdout), np.mean(multi_holdout)], width,
                yerr=[np.std(single_holdout), np.std(multi_holdout)], label="held-out", color="#BFBFBF")
    axes[1].bar(x + width / 2, [np.mean(single_unseen), np.mean(multi_unseen)], width,
                yerr=[np.std(single_unseen), np.std(multi_unseen)], label="未経験マップ", color="#4472C4")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("相関係数")
    axes[1].set_title("(2) 履歴8手: 単一 vs 複数マップ学習")
    axes[1].legend()

    fig.suptitle("要件7: 行動履歴長の深掘り")
    fig.tight_layout()
    fig.savefig("monitor_history_sweep_comparison.png", dpi=150)
    print("グラフを monitor_history_sweep_comparison.png に保存しました。")


if __name__ == "__main__":
    if sys.argv[1] == "aggregate":
        aggregate()
    else:
        run_one_seed(int(sys.argv[1]))
