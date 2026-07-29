"""
感情AIプロジェクト フェーズ2〜3 プロトタイプ: 高階自己モニタリング層(要件7)のU字型の要因切り分け
==========================================================

これまでの実装記録で、モニタのheld-out相関は学習量(150/500/1500/3000ep)に対して
単調ではなくU字型に推移し(0.47→0.38→0.35→0.59、3系統の学習系列で再現性確認済み)、
「方策が素朴な初期段階・複雑化する過渡期・洗練された最終段階を経る中で、モニタが
捉えやすい一貫した行動と捉えにくい行動が交互に現れているのではないか」という仮説が
立てられていた。本プロトタイプはこの仮説を、モニタの精度とは独立に測定できる
「方策自体の複雑さ・一貫性」の指標を使って直接検証する。

3つの複雑さ指標:
  1. 行動エントロピー: ロールアウト中に選ばれた行動の周辺分布のシャノンエントロピー。
     低いほど行動が特定パターンに偏っている(単純・一貫)ことを意味する。
  2. 平均|Q値ギャップ|: 選んだ行動と次点行動のQ値差の平均。大きいほど、その状態での
     行動選択に「迷いがない」(確信度が高い)ことを意味する。
  3. 方策の変化率(churn): 直前のチェックポイントと比べて、同じ状態での貪欲行動が
     どれだけ変わったか。低いほど方策が安定していることを意味する。

これら3指標がすべて「150epと3000epで低く(単純・安定)、500〜1500epで高い
(複雑・不安定)」という谷型(モニタのU字型とは逆位相)を示せば、方策の複雑さの
推移がモニタ精度のU字型の原因である、という仮説を支持する証拠になる。
マップ(TRAIN_SEED=0)は固定したまま、学習系列の乱数のみを変えた3系統
(traj_seed=0, 11, 22、既存の再現性確認実験と同じ設定)で検証する。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from homeostasis_prototype import HomeostasisEnv, QLearningAgent, ACTIONS
import instinct_bias_prototype as ib
from monitor_maturity_prototype import (
    train_with_checkpoints, collect_rollout, fit_linear_regression, predict_linear,
    mean_correlation, accuracy, majority_baseline, action_one_hot,
)

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAIN_SEED = 0
TRAJ_SEEDS = [0, 11, 22]                       # 既存の再現性確認実験と同じ学習系列
CHECKPOINT_EPISODES = [150, 500, 1500, 3000]
ROLLOUT_EPS = 0.1
N_EPISODES_TRAIN_MAP = 100
Q_GAP_COL = len(ACTIONS) + 1                   # collect_rolloutの特徴量中でq_gapが入る列


def action_entropy(actions_list):
    """行動の周辺分布のシャノンエントロピー(底2)。行動が5種なら理論上最大はlog2(5)=2.32。"""
    counts = np.array([actions_list.count(a) for a in ACTIONS], dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def collect_actions_and_states(env, q_table, n_episodes, eps):
    """action_entropy計算用に生の行動列を、churn計算用に訪問した状態集合を集める。"""
    agent = QLearningAgent()
    agent.q = q_table
    actions_list = []
    visited_states = set()
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = agent.best_action(state)
            actions_list.append(action)
            visited_states.add(state)
            next_state, reward, done, deviation = env.step(action)
            state = next_state
    return actions_list, visited_states


def policy_churn(prev_q, curr_q, states):
    """直前チェックポイントと比べ、同じ状態集合で貪欲行動が変わった割合。"""
    if not states:
        return float("nan")
    prev_agent = QLearningAgent(); prev_agent.q = prev_q
    curr_agent = QLearningAgent(); curr_agent.q = curr_q
    diffs = sum(1 for s in states if prev_agent.best_action(s) != curr_agent.best_action(s))
    return diffs / len(states)


if __name__ == "__main__":
    train_env = HomeostasisEnv(random.Random(TRAIN_SEED))  # マップは固定して構築

    per_seed_records = []  # 各traj_seedごとに、チェックポイント順のdictリスト

    for traj_seed in TRAJ_SEEDS:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        agent = QLearningAgent()
        checkpoints = train_with_checkpoints(
            train_env, agent, max(CHECKPOINT_EPISODES), ib.PARENT_EPS_DECAY_EPISODES, CHECKPOINT_EPISODES
        )

        seed_records = []
        prev_q = None
        for n_ep in CHECKPOINT_EPISODES:
            q_table = checkpoints[n_ep]

            # --- モニタ精度(monitor_maturity_prototypeと同一手順で再計算) ---
            random.seed(traj_seed * 1000 + n_ep)
            np.random.seed(traj_seed * 1000 + n_ep)
            map_env = HomeostasisEnv(random.Random(TRAIN_SEED))
            X_all, yc_all, ycont_all = collect_rollout(map_env, dict(q_table), N_EPISODES_TRAIN_MAP, ROLLOUT_EPS)
            n = len(X_all)
            idx = np.random.permutation(n)
            split = int(n * 0.7)
            X_tr, ycont_tr = X_all[idx[:split]], ycont_all[idx[:split]]
            X_te, ycont_te = X_all[idx[split:]], ycont_all[idx[split:]]
            W = fit_linear_regression(X_tr, ycont_tr)
            pred_te = predict_linear(X_te, W)
            corr_te = mean_correlation(ycont_te, pred_te)
            mean_q_gap = float(np.mean(X_all[:, Q_GAP_COL]))

            # --- 方策の複雑さ指標(モニタとは独立に測定) ---
            random.seed(traj_seed * 2000 + n_ep)
            np.random.seed(traj_seed * 2000 + n_ep)
            complexity_env = HomeostasisEnv(random.Random(TRAIN_SEED))
            actions_list, visited_states = collect_actions_and_states(
                complexity_env, dict(q_table), N_EPISODES_TRAIN_MAP, ROLLOUT_EPS
            )
            ent = action_entropy(actions_list)
            churn = policy_churn(prev_q, q_table, visited_states) if prev_q is not None else float("nan")

            seed_records.append({
                "n_episodes": n_ep, "corr_holdout": corr_te, "mean_q_gap": mean_q_gap,
                "action_entropy": ent, "policy_churn": churn,
            })
            prev_q = q_table

        per_seed_records.append(seed_records)
        print(f"--- traj_seed={traj_seed} ---")
        for r in seed_records:
            print(
                f"  {r['n_episodes']}ep: held-out相関={r['corr_holdout']:.4f}, "
                f"平均|Qギャップ|={r['mean_q_gap']:.4f}, 行動エントロピー={r['action_entropy']:.4f}, "
                f"方策変化率={r['policy_churn']:.4f}"
            )

    # --- 集計(seed間で平均±標準偏差) ---
    summary = []
    for i, n_ep in enumerate(CHECKPOINT_EPISODES):
        vals = {k: [rec[i][k] for rec in per_seed_records] for k in
                ["corr_holdout", "mean_q_gap", "action_entropy", "policy_churn"]}
        summary.append({
            "n_episodes": n_ep,
            **{f"{k}_mean": float(np.nanmean(v)) for k, v in vals.items()},
            **{f"{k}_std": float(np.nanstd(v)) for k, v in vals.items()},
        })

    print("\n=== 集計(n=3の平均±標準偏差) ===")
    for s in summary:
        print(
            f"{s['n_episodes']}ep: held-out相関={s['corr_holdout_mean']:.4f}±{s['corr_holdout_std']:.4f}, "
            f"平均|Qギャップ|={s['mean_q_gap_mean']:.4f}±{s['mean_q_gap_std']:.4f}, "
            f"行動エントロピー={s['action_entropy_mean']:.4f}±{s['action_entropy_std']:.4f}, "
            f"方策変化率={s['policy_churn_mean']:.4f}±{s['policy_churn_std']:.4f}"
        )

    # --- 複雑さ指標とモニタ精度の相関(全seed×全チェックポイント、n=12点) ---
    flat = [rec for seed_records in per_seed_records for rec in seed_records]
    corr_vals = np.array([r["corr_holdout"] for r in flat])
    ent_vals = np.array([r["action_entropy"] for r in flat])
    gap_vals = np.array([r["mean_q_gap"] for r in flat])
    churn_vals = np.array([r["policy_churn"] for r in flat if not np.isnan(r["policy_churn"])])
    corr_vals_for_churn = np.array([r["corr_holdout"] for r in flat if not np.isnan(r["policy_churn"])])

    r_ent = np.corrcoef(ent_vals, corr_vals)[0, 1]
    r_gap = np.corrcoef(gap_vals, corr_vals)[0, 1]
    r_churn = np.corrcoef(churn_vals, corr_vals_for_churn)[0, 1] if len(churn_vals) > 1 else float("nan")
    print(f"\n複雑さ指標とモニタheld-out相関との相関係数(n=12点): "
          f"行動エントロピー r={r_ent:.3f}, 平均|Qギャップ| r={r_gap:.3f}, 方策変化率 r={r_churn:.3f}")

    # --- 可視化 ---
    ns = [s["n_episodes"] for s in summary]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].errorbar(ns, [s["corr_holdout_mean"] for s in summary],
                         yerr=[s["corr_holdout_std"] for s in summary], marker="o", color="#4472C4")
    axes[0, 0].set_title("モニタのheld-out相関(既知のU字型)")
    axes[0, 0].set_xlabel("学習量(episode数)")
    axes[0, 0].set_ylabel("相関係数")

    axes[0, 1].errorbar(ns, [s["action_entropy_mean"] for s in summary],
                         yerr=[s["action_entropy_std"] for s in summary], marker="o", color="#C0504D")
    axes[0, 1].set_title("行動エントロピー(高いほど行動が多様・複雑)")
    axes[0, 1].set_xlabel("学習量(episode数)")
    axes[0, 1].set_ylabel("エントロピー(bit)")

    axes[1, 0].errorbar(ns, [s["mean_q_gap_mean"] for s in summary],
                         yerr=[s["mean_q_gap_std"] for s in summary], marker="o", color="#9BBB59")
    axes[1, 0].set_title("平均|Qギャップ|(低いほど行動選択に迷いがある)")
    axes[1, 0].set_xlabel("学習量(episode数)")
    axes[1, 0].set_ylabel("平均Qギャップ")

    churn_means = [s["policy_churn_mean"] for s in summary]
    churn_stds = [s["policy_churn_std"] for s in summary]
    axes[1, 1].errorbar(ns[1:], churn_means[1:], yerr=churn_stds[1:], marker="o", color="#4BACC6")
    axes[1, 1].set_title("直前チェックポイントからの方策変化率(高いほど不安定)")
    axes[1, 1].set_xlabel("学習量(episode数)")
    axes[1, 1].set_ylabel("変化率")

    fig.suptitle("要件7追加検証: 方策の複雑さ指標とモニタ精度のU字型の関係")
    fig.tight_layout()
    fig.savefig("monitor_policy_complexity.png", dpi=150)
    print("グラフを monitor_policy_complexity.png に保存しました。")
