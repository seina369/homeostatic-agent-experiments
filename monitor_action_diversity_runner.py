"""
monitor_action_diversity_prototype.py の実行を1系統(traj_seed)ずつに分割して
実行するランナー(1回あたりの実行時間が長すぎるため)。

使い方:
  python3 monitor_action_diversity_runner.py <condition> <traj_seed> run       # 1系統だけ実行しJSON保存
  python3 monitor_action_diversity_runner.py <condition> aggregate            # 保存済みJSONを集計・グラフ化
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

import homeostasis_prototype as hp
from homeostasis_prototype import HomeostasisEnv, QLearningAgent
import instinct_bias_prototype as ib
from monitor_maturity_prototype import (
    train_with_checkpoints, collect_rollout, fit_linear_regression, predict_linear, mean_correlation,
)
from monitor_action_diversity_prototype import (
    ExtendedHomeostasisEnv, Homeostasis3DEnv, BASE_ACTIONS, EXT_ACTIONS, ACTIONS_3D,
    shannon_entropy, mutual_information,
    TRAIN_SEED, CHECKPOINT_EPISODES, ROLLOUT_EPS, N_EPISODES_TRAIN_MAP,
)

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAJ_SEEDS = [0, 11, 22]


def run_one_seed(condition, traj_seed):
    if condition == "extended":
        hp.ACTIONS[:] = EXT_ACTIONS
        env_cls = ExtendedHomeostasisEnv
    elif condition == "3d":
        hp.ACTIONS[:] = ACTIONS_3D
        env_cls = Homeostasis3DEnv
    else:
        hp.ACTIONS[:] = BASE_ACTIONS
        env_cls = HomeostasisEnv
    n_actions = len(hp.ACTIONS)
    max_entropy = float(np.log2(n_actions))

    train_env = env_cls(random.Random(TRAIN_SEED))
    random.seed(traj_seed)
    np.random.seed(traj_seed)
    agent = QLearningAgent()
    checkpoints = train_with_checkpoints(
        train_env, agent, max(CHECKPOINT_EPISODES), ib.PARENT_EPS_DECAY_EPISODES, CHECKPOINT_EPISODES
    )

    records = []
    for n_ep in CHECKPOINT_EPISODES:
        q_table = checkpoints[n_ep]
        random.seed(traj_seed * 1000 + n_ep)
        np.random.seed(traj_seed * 1000 + n_ep)
        map_env = env_cls(random.Random(TRAIN_SEED))
        X_all, yc_all, ycont_all = collect_rollout(map_env, dict(q_table), N_EPISODES_TRAIN_MAP, ROLLOUT_EPS)

        action_idx = np.argmax(X_all[:, :n_actions], axis=1)
        counts = np.bincount(action_idx, minlength=n_actions).astype(float)
        ent = shannon_entropy(counts)
        mi, h_y, h_y_given_a = mutual_information(action_idx, yc_all, n_actions)

        n = len(X_all)
        idx = np.random.permutation(n)
        split = int(n * 0.7)
        X_tr, ycont_tr = X_all[idx[:split]], ycont_all[idx[:split]]
        X_te, ycont_te = X_all[idx[split:]], ycont_all[idx[split:]]
        W = fit_linear_regression(X_tr, ycont_tr)
        pred_te = predict_linear(X_te, W)
        corr_te = mean_correlation(ycont_te, pred_te)

        records.append({
            "n_episodes": n_ep, "corr_holdout": corr_te,
            "entropy": ent, "entropy_norm": ent / max_entropy,
            "mi": mi, "h_y": h_y, "h_y_given_a": h_y_given_a,
        })

    out = {"condition": condition, "traj_seed": traj_seed, "n_actions": n_actions,
           "max_entropy": max_entropy, "records": records}
    fname = f"diversity_{condition}_seed{traj_seed}.json"
    with open(fname, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"saved {fname}")
    for r in records:
        print(f"  {r['n_episodes']}ep: 相関={r['corr_holdout']:.4f}, MI={r['mi']:.4f}, "
              f"H(Y|A)={r['h_y_given_a']:.4f}, エントロピー={r['entropy']:.4f}({r['entropy_norm']:.2%})")


def aggregate(condition):
    all_data = []
    for seed in TRAJ_SEEDS:
        with open(f"diversity_{condition}_seed{seed}.json") as f:
            all_data.append(json.load(f))
    n_actions = all_data[0]["n_actions"]
    max_entropy = all_data[0]["max_entropy"]

    summary = []
    for i, n_ep in enumerate(CHECKPOINT_EPISODES):
        keys = ["corr_holdout", "entropy", "entropy_norm", "mi", "h_y_given_a"]
        vals = {k: [d["records"][i][k] for d in all_data] for k in keys}
        summary.append({"n_episodes": n_ep,
                         **{f"{k}_mean": float(np.mean(v)) for k, v in vals.items()},
                         **{f"{k}_std": float(np.std(v)) for k, v in vals.items()}})

    print(f"=== 条件: {condition}(行動数={n_actions}, 理論上限エントロピー={max_entropy:.3f}bit) ===")
    for s in summary:
        print(
            f"{s['n_episodes']}ep: held-out相関={s['corr_holdout_mean']:.4f}±{s['corr_holdout_std']:.4f}, "
            f"MI={s['mi_mean']:.4f}±{s['mi_std']:.4f}bit, "
            f"H(Y|A)={s['h_y_given_a_mean']:.4f}±{s['h_y_given_a_std']:.4f}bit, "
            f"エントロピー={s['entropy_mean']:.4f}±{s['entropy_std']:.4f}bit(上限比{s['entropy_norm_mean']:.1%}±{s['entropy_norm_std']:.1%})"
        )

    corr_vals = np.array([s["corr_holdout_mean"] for s in summary])
    mi_vals = np.array([s["mi_mean"] for s in summary])
    ent_vals = np.array([s["entropy_mean"] for s in summary])
    r_mi = np.corrcoef(mi_vals, corr_vals)[0, 1]
    r_ent = np.corrcoef(ent_vals, corr_vals)[0, 1]
    print(f"\nチェックポイント平均どうしの相関(n=4点): MIとheld-out相関 r={r_mi:.3f}, エントロピーとheld-out相関 r={r_ent:.3f}")

    ns = [s["n_episodes"] for s in summary]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].errorbar(ns, [s["corr_holdout_mean"] for s in summary], yerr=[s["corr_holdout_std"] for s in summary],
                      marker="o", color="#4472C4")
    axes[0].set_title("モニタのheld-out相関")
    axes[0].set_xlabel("学習量(episode数)")

    axes[1].errorbar(ns, [s["mi_mean"] for s in summary], yerr=[s["mi_std"] for s in summary],
                      marker="o", color="#C0504D")
    axes[1].set_title("相互情報量 I(A;Y)")
    axes[1].set_xlabel("学習量(episode数)")
    axes[1].set_ylabel("bit")

    axes[2].errorbar(ns, [s["entropy_norm_mean"] * 100 for s in summary], yerr=[s["entropy_norm_std"] * 100 for s in summary],
                      marker="o", color="#9BBB59")
    axes[2].set_title(f"行動エントロピー(理論上限比、行動数={n_actions})")
    axes[2].set_xlabel("学習量(episode数)")
    axes[2].set_ylabel("%")

    fig.suptitle(f"要件7追加切り分け: 条件={condition}(行動数={n_actions})")
    fig.tight_layout()
    fig.savefig(f"monitor_action_diversity_{condition}.png", dpi=150)
    print(f"グラフを monitor_action_diversity_{condition}.png に保存しました。")
    return summary, r_mi, r_ent


if __name__ == "__main__":
    condition = sys.argv[1]
    if sys.argv[2] == "aggregate":
        aggregate(condition)
    else:
        traj_seed = int(sys.argv[2])
        run_one_seed(condition, traj_seed)
