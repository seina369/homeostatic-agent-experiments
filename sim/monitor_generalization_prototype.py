"""
感情AIプロジェクト フェーズ2〜3 プロトタイプ: 高階自己モニタリング層(要件7)の汎化性能改善

これまでの検証で判明した要件7の最大の弱点は、モニタを1つのマップのロールアウト
データだけで学習させると、未経験マップへの連続値の追従(相関)がほぼゼロに近い
水準まで失われる、という点だった(held-out相関0.35〜0.63に対し未経験マップ相関
0.02〜0.10)。本プロトタイプは、この弱点が「モニタの学習データの多様性不足」
という改善可能な問題なのか、それとも設計そのものの限界なのかを切り分けるため、
モニタを複数マップのロールアウトデータで学習させ、真に未経験のマップへの汎化が
改善するかを検証する。

設計:
  - 方策(エージェント)はTRAIN_SEED=0で1回だけ学習する(従来通り、相対方向の
    状態表現によりマップが変わっても方策はそのまま使える)。
  - 単一マップ版モニタ(従来の比較対象): 方策の学習に使ったマップ(seed=0)だけの
    ロールアウトデータでモニタを学習する。
  - 複数マップ版モニタ(本提案): モニタの学習データを、方策の学習には使っていない
    複数の別マップ(seed=1,2,3)のロールアウトデータも合わせて学習する。
  - どちらのモニタも、真に未経験の3マップ(seed=5,6,7、モニタの学習には一切
    使わない)で最終評価し、相関係数を比較する。
  - 学習系列の乱数(traj_seed=0,11,22)を変えた3系統で再現性を確認する
    (既存の一連の実験と同じ設定)。

使い方:
  python3 monitor_generalization_prototype.py <traj_seed> run       # 1系統だけ実行しJSON保存
  python3 monitor_generalization_prototype.py aggregate             # 保存済みJSONを集計・グラフ化
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
from monitor_maturity_prototype import collect_rollout, fit_linear_regression, predict_linear, mean_correlation

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAIN_SEED = 0
MULTI_MAP_SEEDS = [0, 1, 2, 3]     # モニタの複数マップ学習に使うマップ(方策学習マップ0を含む)
UNSEEN_SEEDS = [5, 6, 7]           # 真に未経験(モニタ学習には一切使わない)の評価専用マップ
N_EPISODES_PER_MAP = 100
N_EPISODES_UNSEEN = 60
ROLLOUT_EPS = 0.1
TRAJ_SEEDS = [0, 11, 22]


def train_policy(traj_seed):
    random.seed(traj_seed)
    np.random.seed(traj_seed)
    env = HomeostasisEnv(random.Random(TRAIN_SEED))
    agent = QLearningAgent()
    ib.train(env, agent, ib.PARENT_EPISODES, ib.PARENT_EPS_DECAY_EPISODES)
    return agent.q


def evaluate_on_unseen_with_q(policy_q, W, traj_seed):
    corrs = []
    for seed in UNSEEN_SEEDS:
        random.seed(traj_seed * 5000 + seed)
        np.random.seed(traj_seed * 5000 + seed)
        env = HomeostasisEnv(random.Random(seed))
        X, yc, ycont = collect_rollout(env, dict(policy_q), N_EPISODES_UNSEEN, ROLLOUT_EPS)
        pred = predict_linear(X, W)
        corrs.append(mean_correlation(ycont, pred))
    return float(np.mean(corrs)), float(np.std(corrs)), corrs


def run_one_seed(traj_seed):
    policy_q = train_policy(traj_seed)

    # --- 単一マップ版モニタ(従来法): 方策学習マップ(seed=0)だけで学習 ---
    random.seed(traj_seed * 1000)
    np.random.seed(traj_seed * 1000)
    env0 = HomeostasisEnv(random.Random(TRAIN_SEED))
    X0, yc0, ycont0 = collect_rollout(env0, dict(policy_q), N_EPISODES_PER_MAP, ROLLOUT_EPS)
    n0 = len(X0)
    idx0 = np.random.permutation(n0)
    split0 = int(n0 * 0.7)
    W_single = fit_linear_regression(X0[idx0[:split0]], ycont0[idx0[:split0]])
    pred_single_holdout = predict_linear(X0[idx0[split0:]], W_single)
    corr_single_holdout = mean_correlation(ycont0[idx0[split0:]], pred_single_holdout)
    unseen_mean_single, unseen_std_single, unseen_list_single = evaluate_on_unseen_with_q(policy_q, W_single, traj_seed)

    # --- 複数マップ版モニタ(提案手法): seed=0,1,2,3のロールアウトを合算して学習 ---
    X_list, ycont_list = [X0], [ycont0]
    for seed in MULTI_MAP_SEEDS:
        if seed == TRAIN_SEED:
            continue  # seed=0はすでに収集済み(X0)なので重複収集しない
        random.seed(traj_seed * 1000 + seed)
        np.random.seed(traj_seed * 1000 + seed)
        env_s = HomeostasisEnv(random.Random(seed))
        X_s, yc_s, ycont_s = collect_rollout(env_s, dict(policy_q), N_EPISODES_PER_MAP, ROLLOUT_EPS)
        X_list.append(X_s)
        ycont_list.append(ycont_s)
    X_multi = np.concatenate(X_list, axis=0)
    ycont_multi = np.concatenate(ycont_list, axis=0)
    n_multi = len(X_multi)
    idx_multi = np.random.permutation(n_multi)
    split_multi = int(n_multi * 0.7)
    W_multi = fit_linear_regression(X_multi[idx_multi[:split_multi]], ycont_multi[idx_multi[:split_multi]])
    pred_multi_holdout = predict_linear(X_multi[idx_multi[split_multi:]], W_multi)
    corr_multi_holdout = mean_correlation(ycont_multi[idx_multi[split_multi:]], pred_multi_holdout)
    unseen_mean_multi, unseen_std_multi, unseen_list_multi = evaluate_on_unseen_with_q(policy_q, W_multi, traj_seed)

    result = {
        "traj_seed": traj_seed,
        "n_single_map_samples": n0,
        "n_multi_map_samples": n_multi,
        "corr_single_holdout": corr_single_holdout,
        "corr_single_unseen_mean": unseen_mean_single,
        "corr_single_unseen_std": unseen_std_single,
        "corr_single_unseen_list": unseen_list_single,
        "corr_multi_holdout": corr_multi_holdout,
        "corr_multi_unseen_mean": unseen_mean_multi,
        "corr_multi_unseen_std": unseen_std_multi,
        "corr_multi_unseen_list": unseen_list_multi,
    }
    fname = f"generalization_seed{traj_seed}.json"
    with open(fname, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved {fname}")
    print(f"  単一マップ版: held-out相関={corr_single_holdout:.4f}, 未経験マップ相関={unseen_mean_single:.4f}±{unseen_std_single:.4f} (内訳={[f'{c:.3f}' for c in unseen_list_single]})")
    print(f"  複数マップ版: held-out相関={corr_multi_holdout:.4f}, 未経験マップ相関={unseen_mean_multi:.4f}±{unseen_std_multi:.4f} (内訳={[f'{c:.3f}' for c in unseen_list_multi]})")


def aggregate():
    data = []
    for seed in TRAJ_SEEDS:
        with open(f"generalization_seed{seed}.json") as f:
            data.append(json.load(f))

    def stat(key):
        vals = [d[key] for d in data]
        return float(np.mean(vals)), float(np.std(vals))

    single_holdout = stat("corr_single_holdout")
    single_unseen = stat("corr_single_unseen_mean")
    multi_holdout = stat("corr_multi_holdout")
    multi_unseen = stat("corr_multi_unseen_mean")

    print("=== 集計(n=3の平均±標準偏差) ===")
    print(f"単一マップ版モニタ: held-out相関={single_holdout[0]:.4f}±{single_holdout[1]:.4f}, "
          f"未経験マップ相関={single_unseen[0]:.4f}±{single_unseen[1]:.4f}")
    print(f"複数マップ版モニタ: held-out相関={multi_holdout[0]:.4f}±{multi_holdout[1]:.4f}, "
          f"未経験マップ相関={multi_unseen[0]:.4f}±{multi_unseen[1]:.4f}")
    improvement = multi_unseen[0] - single_unseen[0]
    print(f"未経験マップ相関の改善幅: {improvement:+.4f}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels = ["held-out\n(学習分布内)", "未経験マップ\n(真の汎化)"]
    single_means = [single_holdout[0], single_unseen[0]]
    single_stds = [single_holdout[1], single_unseen[1]]
    multi_means = [multi_holdout[0], multi_unseen[0]]
    multi_stds = [multi_holdout[1], multi_unseen[1]]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, single_means, width, yerr=single_stds, label="単一マップ版モニタ(従来)", color="#BFBFBF")
    ax.bar(x + width / 2, multi_means, width, yerr=multi_stds, label="複数マップ版モニタ(提案)", color="#4472C4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("相関係数(真の逸脱量とモニタ予測の平均相関)")
    ax.set_title("要件7: モニタの学習データを複数マップにすると汎化は改善するか")
    ax.legend()
    fig.tight_layout()
    fig.savefig("monitor_generalization_comparison.png", dpi=150)
    print("グラフを monitor_generalization_comparison.png に保存しました。")


if __name__ == "__main__":
    if sys.argv[1] == "aggregate":
        aggregate()
    else:
        run_one_seed(int(sys.argv[1]))
