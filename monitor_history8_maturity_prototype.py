"""
感情AIプロジェクト フェーズ2〜3 プロトタイプ: 要件7 U字型の再検証(履歴8手モニタ)
==========================================================

monitor_action_diversity_prototype.py(baseline条件)で確認したU字型は、モニタの
特徴量が「直近2手の行動履歴」(history_len=2)だった時点の結果:
  150ep=0.4849 → 500ep=0.4039 → 1500ep=0.3631 → 3000ep=0.6011 (n=3の平均)

monitor_history_sweep_prototype.pyで、履歴長を8手に延ばすだけで(複数マップ学習と
組み合わせた)未経験マップ相関が大きく改善することを確認した。これが「学習量に
対する精度の推移(U字型)」自体をどう変えるかを、同じ条件(単一マップ学習・
TRAIN_SEED=0・チェックポイント150/500/1500/3000ep・traj_seed=0,11,22)で再検証する。

monitor_maturity_prototype.collect_rolloutはHISTORY_LEN(モジュール定数)を参照する
ので、呼び出し前にモンキーパッチして8に差し替える(単一マップ学習・単一エピソード
集合での評価であり、monitor_history_sweep_prototype.pyの複数マップ学習とは別の
実験である点に注意)。

使い方:
  python3 monitor_history8_maturity_prototype.py <traj_seed> run
  python3 monitor_history8_maturity_prototype.py aggregate
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
import monitor_maturity_prototype as mm

mm.HISTORY_LEN = 8  # U字型再検証のため、モニタの行動履歴長を2→8に拡張

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAIN_SEED = 0
CHECKPOINT_EPISODES = [150, 500, 1500, 3000]
ROLLOUT_EPS = 0.1
N_EPISODES_TRAIN_MAP = 100
TRAJ_SEEDS = [0, 11, 22]

# 履歴長2手の既存結果(monitor_action_diversity_prototype.py baseline条件、n=3平均)
BASELINE_HIST2 = {
    150: 0.4849, 500: 0.4039, 1500: 0.3631, 3000: 0.6011,
}


def run_one_seed(traj_seed):
    random.seed(traj_seed)
    np.random.seed(traj_seed)
    train_env = HomeostasisEnv(random.Random(TRAIN_SEED))
    agent = QLearningAgent()
    checkpoints = mm.train_with_checkpoints(
        train_env, agent, max(CHECKPOINT_EPISODES), ib.PARENT_EPS_DECAY_EPISODES, CHECKPOINT_EPISODES
    )

    result = {"traj_seed": traj_seed, "checkpoints": {}}
    for n_ep in CHECKPOINT_EPISODES:
        q_table = checkpoints[n_ep]
        random.seed(traj_seed * 1000 + n_ep)
        np.random.seed(traj_seed * 1000 + n_ep)
        map_env = HomeostasisEnv(random.Random(TRAIN_SEED))
        X_all, yc_all, ycont_all = mm.collect_rollout(map_env, dict(q_table), N_EPISODES_TRAIN_MAP, ROLLOUT_EPS)

        n = len(X_all)
        idx = np.random.permutation(n)
        split = int(n * 0.7)
        X_tr, ycont_tr = X_all[idx[:split]], ycont_all[idx[:split]]
        X_te, yc_te, ycont_te = X_all[idx[split:]], yc_all[idx[split:]], ycont_all[idx[split:]]

        W = mm.fit_linear_regression(X_tr, ycont_tr)
        pred_te = mm.predict_linear(X_te, W)
        pred_class_te = np.argmax(pred_te, axis=1)
        acc_te = mm.accuracy(yc_te, pred_class_te)
        corr_te = mm.mean_correlation(ycont_te, pred_te)

        result["checkpoints"][str(n_ep)] = {"acc_holdout": acc_te, "corr_holdout": corr_te}
        print(f"[seed={traj_seed}] {n_ep}ep(履歴8手): held-out精度={acc_te:.4f}, held-out相関={corr_te:.4f}")

    fname = f"history8_maturity_seed{traj_seed}.json"
    with open(fname, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved {fname}")


def aggregate():
    data = []
    for seed in TRAJ_SEEDS:
        with open(f"history8_maturity_seed{seed}.json") as f:
            data.append(json.load(f))

    print("=== U字型再検証(履歴8手モニタ、n=3の平均±標準偏差) ===")
    summary = {}
    for n_ep in CHECKPOINT_EPISODES:
        key = str(n_ep)
        corrs = [d["checkpoints"][key]["corr_holdout"] for d in data]
        accs = [d["checkpoints"][key]["acc_holdout"] for d in data]
        summary[n_ep] = {
            "corr_mean": float(np.mean(corrs)), "corr_std": float(np.std(corrs)),
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        }
        print(f"{n_ep}ep: held-out相関={np.mean(corrs):.4f}±{np.std(corrs):.4f} "
              f"(履歴2手時: {BASELINE_HIST2[n_ep]:.4f}), held-out精度={np.mean(accs):.4f}±{np.std(accs):.4f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ns = CHECKPOINT_EPISODES
    corr_means = [summary[n]["corr_mean"] for n in ns]
    corr_stds = [summary[n]["corr_std"] for n in ns]
    baseline_vals = [BASELINE_HIST2[n] for n in ns]
    ax.errorbar(ns, corr_means, yerr=corr_stds, marker="o", label="履歴8手", color="#4472C4", linewidth=2)
    ax.plot(ns, baseline_vals, "o--", label="履歴2手(既存結果)", color="#C0504D", linewidth=2)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("エージェントの学習量(episode数)")
    ax.set_ylabel("held-out相関")
    ax.set_title("要件7: U字型は履歴長を8手にすると変わるか")
    ax.legend()
    fig.tight_layout()
    fig.savefig("monitor_history8_maturity_comparison.png", dpi=150)
    print("グラフを monitor_history8_maturity_comparison.png に保存しました。")


if __name__ == "__main__":
    if sys.argv[1] == "aggregate":
        aggregate()
    else:
        run_one_seed(int(sys.argv[1]))
