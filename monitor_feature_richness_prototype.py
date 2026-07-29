"""
感情AIプロジェクト フェーズ2〜3 プロトタイプ: 高階自己モニタリング層(要件7)の
特徴量設計を見直して汎化性能を改善できるか検証する。

monitor_generalization_prototype.pyで、モニタの学習データを複数マップにすると
未経験マップ相関が0.026→0.064へ改善することを確認したが、学習分布内(0.39〜0.59)
にはまだ遠く及ばなかった。今回はモニタの特徴量自体を見直す:

  - 現状: 選んだ行動(one-hot)・そのQ値・次点とのQ値差・直近2手の行動(one-hot)
  - 変更案(A) 行動履歴を2手から8手へ延長: 「同じ行動が異なる欲求の場面で使い回され
    ている」問題に対し、より長い文脈を与えれば区別しやすくなるか
  - 変更案(B) 資源(食料・シェルター・危険地帯)への相対方向を特徴量に追加:
    行動そのものだけでなく「その行動が取られた文脈」を与える。これはセンサーの
    生の値(エネルギー・体温・損傷)そのものではないが、エージェントが参照している
    状態の一部(環境構造)であり、「行動だけから見えるか」という純粋な検証からは
    一歩踏み込んだ設計になる点に注意。
  - 変更案(A)+(B)の組み合わせ

効率化のため、ロールアウト自体は1回だけ行い、各ステップの生ログ(選んだ行動・Q値・
Qギャップ・直近10手の行動履歴・資源への相対方向・真の逸脱量)を保存しておき、
そこから4通りの特徴量行列を構築して比較する(方策の学習・ロールアウトそのものは
特徴量設計に依らず共通のため、使い回すことで計算量を抑える)。

学習データは複数マップ(seed=0,1,2,3)、評価は真に未経験の3マップ(seed=5,6,7)、
学習系列の乱数(traj_seed=0,11,22)を変えた3系統で再現性を確認する。

使い方:
  python3 monitor_feature_richness_prototype.py <traj_seed> run
  python3 monitor_feature_richness_prototype.py aggregate
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

from homeostasis_prototype import HomeostasisEnv, QLearningAgent, ACTIONS
import instinct_bias_prototype as ib
from monitor_maturity_prototype import fit_linear_regression, predict_linear, mean_correlation

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
MAX_HISTORY = 10
TRAJ_SEEDS = [0, 11, 22]

OPTIMAL_ENERGY = 100.0
OPTIMAL_TEMP = 20.0
OPTIMAL_DAMAGE = 0.0

VARIANTS = {
    "baseline(履歴2,方向なし)": {"history_len": 2, "use_direction": False},
    "履歴8(方向なし)": {"history_len": 8, "use_direction": False},
    "履歴2+資源方向": {"history_len": 2, "use_direction": True},
    "履歴8+資源方向": {"history_len": 8, "use_direction": True},
}


def deviations(energy, temperature, damage):
    dev_energy = abs(energy - OPTIMAL_ENERGY) / 100.0
    dev_temp = abs(temperature - OPTIMAL_TEMP) / 30.0
    dev_damage = abs(damage - OPTIMAL_DAMAGE) / 100.0
    return np.array([dev_energy, dev_temp, dev_damage])


def action_one_hot(action):
    v = np.zeros(len(ACTIONS))
    v[ACTIONS.index(action)] = 1.0
    return v


def collect_rollout_raw(env, q_table, n_episodes, eps):
    """1ステップごとの生ログを集める。特徴量の組み立ては後段のbuild_featuresで行う。"""
    agent = QLearningAgent()
    agent.q = q_table
    records = []
    for ep in range(n_episodes):
        state = env.reset()
        history = ["stay"] * MAX_HISTORY
        done = False
        while not done:
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = agent.best_action(state)
            q_values = [agent.q_value(state, a) for a in ACTIONS]
            sorted_q = sorted(q_values, reverse=True)
            chosen_q = agent.q_value(state, action)
            q_gap = sorted_q[0] - (sorted_q[1] if len(sorted_q) > 1 else sorted_q[0])
            food_dir, shelter_dir, hazard_dir = state[0], state[1], state[2]

            next_state, reward, done, deviation = env.step(action)
            dev_vec = deviations(env.energy, env.temperature, env.damage)

            records.append({
                "action": action, "chosen_q": chosen_q, "q_gap": q_gap,
                "history": list(history),
                "food_dir": food_dir, "shelter_dir": shelter_dir, "hazard_dir": hazard_dir,
                "y_cont": dev_vec,
            })
            history = history[1:] + [action]
            state = next_state
    return records


def build_features(records, history_len, use_direction):
    X, Y = [], []
    for r in records:
        parts = [action_one_hot(r["action"]), np.array([r["chosen_q"], r["q_gap"]])]
        hist = r["history"][-history_len:]
        for a in hist:
            parts.append(action_one_hot(a))
        if use_direction:
            parts.append(np.array(r["food_dir"], dtype=float))
            parts.append(np.array(r["shelter_dir"], dtype=float))
            parts.append(np.array(r["hazard_dir"], dtype=float))
        X.append(np.concatenate(parts))
        Y.append(r["y_cont"])
    return np.array(X), np.array(Y)


def run_one_seed(traj_seed):
    random.seed(traj_seed)
    np.random.seed(traj_seed)
    env = HomeostasisEnv(random.Random(TRAIN_SEED))
    agent = QLearningAgent()
    ib.train(env, agent, ib.PARENT_EPISODES, ib.PARENT_EPS_DECAY_EPISODES)
    policy_q = agent.q

    train_records = []
    for seed in MULTI_MAP_SEEDS:
        random.seed(traj_seed * 1000 + seed)
        np.random.seed(traj_seed * 1000 + seed)
        map_env = HomeostasisEnv(random.Random(seed))
        train_records.extend(collect_rollout_raw(map_env, dict(policy_q), N_EPISODES_PER_MAP, ROLLOUT_EPS))

    unseen_records = {}
    for seed in UNSEEN_SEEDS:
        random.seed(traj_seed * 5000 + seed)
        np.random.seed(traj_seed * 5000 + seed)
        map_env = HomeostasisEnv(random.Random(seed))
        unseen_records[seed] = collect_rollout_raw(map_env, dict(policy_q), N_EPISODES_UNSEEN, ROLLOUT_EPS)

    result = {"traj_seed": traj_seed, "variants": {}}
    rng = np.random.RandomState(traj_seed * 999)
    perm = rng.permutation(len(train_records))
    split = int(len(train_records) * 0.7)
    train_idx, holdout_idx = perm[:split], perm[split:]

    for name, cfg in VARIANTS.items():
        X_all, Y_all = build_features(train_records, cfg["history_len"], cfg["use_direction"])
        X_tr, Y_tr = X_all[train_idx], Y_all[train_idx]
        X_ho, Y_ho = X_all[holdout_idx], Y_all[holdout_idx]
        W = fit_linear_regression(X_tr, Y_tr)
        corr_holdout = mean_correlation(Y_ho, predict_linear(X_ho, W))

        unseen_corrs = []
        for seed in UNSEEN_SEEDS:
            X_u, Y_u = build_features(unseen_records[seed], cfg["history_len"], cfg["use_direction"])
            unseen_corrs.append(mean_correlation(Y_u, predict_linear(X_u, W)))

        result["variants"][name] = {
            "n_features": X_all.shape[1],
            "corr_holdout": corr_holdout,
            "corr_unseen_mean": float(np.mean(unseen_corrs)),
            "corr_unseen_std": float(np.std(unseen_corrs)),
        }
        print(f"[seed={traj_seed}] {name} (次元数={X_all.shape[1]}): "
              f"held-out相関={corr_holdout:.4f}, 未経験マップ相関={np.mean(unseen_corrs):.4f}±{np.std(unseen_corrs):.4f}")

    fname = f"feature_richness_seed{traj_seed}.json"
    with open(fname, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved {fname}")


def aggregate():
    data = []
    for seed in TRAJ_SEEDS:
        with open(f"feature_richness_seed{seed}.json") as f:
            data.append(json.load(f))

    print("=== 集計(n=3の平均±標準偏差) ===")
    summary = {}
    for name in VARIANTS:
        holdout_vals = [d["variants"][name]["corr_holdout"] for d in data]
        unseen_vals = [d["variants"][name]["corr_unseen_mean"] for d in data]
        n_feat = data[0]["variants"][name]["n_features"]
        summary[name] = {
            "n_features": n_feat,
            "holdout_mean": float(np.mean(holdout_vals)), "holdout_std": float(np.std(holdout_vals)),
            "unseen_mean": float(np.mean(unseen_vals)), "unseen_std": float(np.std(unseen_vals)),
        }
        print(f"{name} (次元数={n_feat}): held-out相関={np.mean(holdout_vals):.4f}±{np.std(holdout_vals):.4f}, "
              f"未経験マップ相関={np.mean(unseen_vals):.4f}±{np.std(unseen_vals):.4f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = list(VARIANTS.keys())
    x = np.arange(len(names))
    width = 0.35
    holdout_means = [summary[n]["holdout_mean"] for n in names]
    holdout_stds = [summary[n]["holdout_std"] for n in names]
    unseen_means = [summary[n]["unseen_mean"] for n in names]
    unseen_stds = [summary[n]["unseen_std"] for n in names]
    ax.bar(x - width / 2, holdout_means, width, yerr=holdout_stds, label="held-out(学習分布内)", color="#BFBFBF")
    ax.bar(x + width / 2, unseen_means, width, yerr=unseen_stds, label="未経験マップ(真の汎化)", color="#4472C4")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("相関係数")
    ax.set_title("要件7: 特徴量設計の見直しによる汎化性能の変化")
    ax.legend()
    fig.tight_layout()
    fig.savefig("monitor_feature_richness_comparison.png", dpi=150)
    print("グラフを monitor_feature_richness_comparison.png に保存しました。")


if __name__ == "__main__":
    if sys.argv[1] == "aggregate":
        aggregate()
    else:
        run_one_seed(int(sys.argv[1]))
