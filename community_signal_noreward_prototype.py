"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 直接報酬なしでの信号創発の再検証
==========================================================

二度目の成功実験(community_signal_v2_prototype.py)は「受け手が送り手の内部状態を
正しく当てたら双方にボーナス」という推測ゲーム(GuessAgent、GUESS_BONUS)を
導入していた。これは測定したい量(信号と内部状態の対応関係)を目的関数に
直接組み込んでいるに等しく、成功が設計によってある程度保証されている
可能性が指摘されている。一度目の失敗実験(community_signal_prototype.py)は
直接報酬なしだったが、協調圧力自体も弱い設定(5x5グリッド、衝突ペナルティ
2.0)だった。

本プロトタイプは、二度目の実験と同じ強い協調圧力(4x4グリッド、衝突ペナルティ
8.0)は維持したまま、推測ゲームの直接報酬(GuessAgent、GUESS_BONUS)だけを
完全に外し、衝突回避+恒常性維持の報酬のみで、学習エピソード数を大幅に
伸ばして(8000ep、二度目の実験の3500epの2倍以上)再検証する。信号と内部状態の
相互情報量(MI)・信号送信率を学習チェックポイントごとに測定し、直接報酬
なしでも(時間はかかっても)legibleな信号が創発するか、それとも一度目と
同様に創発しないままかを確認する。

環境・エージェントクラスはcommunity_signal_v2_prototype.pyのものを完全に
再利用する(4x4グリッド・衝突ペナルティ8.0への変更は同モジュールのimport時に
既に適用済み)。信号行動・観測仕様(相手の直前信号・相対方向を含む)も同一。
違いは学習ループから推測ゲーム機構(GuessAgent、GUESS_BONUS)を完全に除去した
点のみ。

学習系列の乱数(traj_seed=0,11,22)を変えた3系統で確認する。45秒のbash呼び出し
制限に対応するため、学習を1600epずつ5チャンクに分割する(合計8000ep)。

使い方:
  python3 community_signal_noreward_prototype.py train_chunk <traj_seed> <end_ep>
  python3 community_signal_noreward_prototype.py train_finalize <traj_seed>
  python3 community_signal_noreward_prototype.py aggregate
"""

import sys, json, pickle
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import homeostasis_prototype as hp
from homeostasis_prototype import QLearningAgent
import instinct_bias_prototype as ib
import community_signal_v2_prototype as m  # GRID_SIZE=4, ACTIONS_COMM, COLLISION_PENALTY=8.0を継承

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAJ_SEEDS = [0, 11, 22]
N_EPISODES = 8000
DECAY_EPISODES = 6000
CHECKPOINT_EPISODES = [500, 1500, 3000, 5000, 8000]
N_ROLLOUT_EPISODES = 100
ROLLOUT_EPS = 0.1


def run_episode_noreward(env, agent0, agent1, eps0, eps1, learn0=True, learn1=True):
    """推測ゲーム機構を一切使わない、衝突回避+恒常性維持の報酬のみの学習エピソード。"""
    obs = env.reset()
    done = False
    devs, collisions, steps = [], 0, 0
    while not done:
        a0 = m.act(agent0, obs[0], eps0)
        a1 = m.act(agent1, obs[1], eps1)
        next_obs, rewards, done, deviations, collided = env.step([a0, a1])
        if learn0:
            agent0.update(obs[0], a0, rewards[0], next_obs[0], done)
        if learn1:
            agent1.update(obs[1], a1, rewards[1], next_obs[1], done)
        obs = next_obs
        devs.append((deviations[0] + deviations[1]) / 2.0)
        collisions += int(collided)
        steps += 1
    return float(np.mean(devs)), collisions / steps


def train_range(env, agent0, agent1, start_ep, end_ep, decay_episodes, checkpoint_eps=None):
    checkpoint_eps = checkpoint_eps or []
    checkpoint_set = set(checkpoint_eps)
    checkpoints = {}
    avg_dev_hist, coll_hist = [], []
    for ep in range(start_ep, end_ep):
        eps = ib.epsilon_for_episode(ep, decay_episodes)
        avg_dev, coll_rate = run_episode_noreward(env, agent0, agent1, eps, eps)
        avg_dev_hist.append(avg_dev)
        coll_hist.append(coll_rate)
        if (ep + 1) in checkpoint_set:
            checkpoints[ep + 1] = (dict(agent0.q), dict(agent1.q))
    return avg_dev_hist, coll_hist, checkpoints


def train_chunk(traj_seed, end_ep):
    state_file = f"noreward_state_seed{traj_seed}.pkl"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env = state["env"]
        agent0, agent1 = state["agent0"], state["agent1"]
        avg_dev_hist, coll_hist = state["avg_dev_hist"], state["coll_hist"]
        all_checkpoints = state["checkpoints"]
        start_ep = state["last_ep"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
        print(f"[無報酬seed={traj_seed}] 再開: {start_ep}ep目から{end_ep}epまで学習")
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        agent0, agent1 = QLearningAgent(), QLearningAgent()
        avg_dev_hist, coll_hist = [], []
        all_checkpoints = {}
        start_ep = 0
        print(f"[無報酬seed={traj_seed}] 新規開始: 0ep目から{end_ep}epまで学習(推測ゲームなし)")

    dev_h, coll_h, checkpoints = train_range(
        env, agent0, agent1, start_ep, end_ep, DECAY_EPISODES, checkpoint_eps=CHECKPOINT_EPISODES
    )
    avg_dev_hist.extend(dev_h); coll_hist.extend(coll_h)
    all_checkpoints.update(checkpoints)

    state = {
        "env": env, "agent0": agent0, "agent1": agent1,
        "avg_dev_hist": avg_dev_hist, "coll_hist": coll_hist,
        "checkpoints": all_checkpoints, "last_ep": end_ep,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)
    print(f"[無報酬seed={traj_seed}] {end_ep}epまで完了・保存 "
          f"(直近200ep衝突率={np.mean(coll_hist[-200:]):.4f}, 平均逸脱={np.mean(avg_dev_hist[-200:]):.4f})")


def rollout_for_mi(env, q0, q1, n_episodes, eps):
    agent0 = QLearningAgent(); agent0.q = q0
    agent1 = QLearningAgent(); agent1.q = q1
    records = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            dom0 = env.dominant_deviation(0)
            dom1 = env.dominant_deviation(1)
            a0 = m.act(agent0, obs[0], eps)
            a1 = m.act(agent1, obs[1], eps)
            records.append((dom0, 1 if a0 == "signal" else 0))
            records.append((dom1, 1 if a1 == "signal" else 0))
            next_obs, rewards, done, deviations, collided = env.step([a0, a1])
            obs = next_obs
    return records


def train_finalize(traj_seed):
    state_file = f"noreward_state_seed{traj_seed}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    avg_dev_hist, coll_hist = state["avg_dev_hist"], state["coll_hist"]
    checkpoints = state["checkpoints"]

    print(f"[無報酬seed={traj_seed}] === 土台: 衝突回避タスク自体の改善 ===")
    print(f"[無報酬seed={traj_seed}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[:500]):.4f}")
    print(f"[無報酬seed={traj_seed}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[-500:]):.4f}")

    mi_by_checkpoint = {}
    for n_ep in CHECKPOINT_EPISODES:
        q0, q1 = checkpoints[n_ep]
        random.seed(traj_seed * 5000 + n_ep)
        np.random.seed(traj_seed * 5000 + n_ep)
        rollout_env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        records = rollout_for_mi(rollout_env, dict(q0), dict(q1), N_ROLLOUT_EPISODES, ROLLOUT_EPS)
        mi, signal_rate, cond_dist, marg_dist = m.mutual_info_signal_vs_class(records)
        mi_by_checkpoint[str(n_ep)] = {
            "mi": mi, "signal_rate": signal_rate,
            "cond_dist_given_signal": cond_dist, "marginal_dist": marg_dist,
        }
        print(f"[無報酬seed={traj_seed}] {n_ep}ep: I(signal;dominant_dev)={mi:.4f}bit, "
              f"信号送信率={signal_rate:.4f}, signal時分布={cond_dist}, 全体分布={marg_dist}")

    result = {
        "traj_seed": traj_seed,
        "avg_dev_history": avg_dev_hist,
        "collision_rate_history": coll_hist,
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(f"noreward_train_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved noreward_train_seed{traj_seed}.json")


def aggregate():
    data = [json.load(open(f"noreward_train_seed{s}.json")) for s in TRAJ_SEEDS]

    print("=== (0) 土台: 衝突回避タスク自体の改善(n=3の平均±標準偏差) ===")
    coll_early = [np.mean(d["collision_rate_history"][:500]) for d in data]
    coll_late = [np.mean(d["collision_rate_history"][-500:]) for d in data]
    print(f"衝突率: 序盤(最初500ep)={np.mean(coll_early):.4f}±{np.std(coll_early):.4f}, "
          f"終盤(最後500ep)={np.mean(coll_late):.4f}±{np.std(coll_late):.4f}")

    print("\n=== (1) 信号と内部状態のMI・信号送信率(チェックポイント別、n=3の平均±標準偏差) ===")
    mi_summary = {}
    for n_ep in CHECKPOINT_EPISODES:
        key = str(n_ep)
        mis = [d["mi_by_checkpoint"][key]["mi"] for d in data]
        rates = [d["mi_by_checkpoint"][key]["signal_rate"] for d in data]
        mi_summary[n_ep] = {
            "mi_mean": float(np.mean(mis)), "mi_std": float(np.std(mis)),
            "rate_mean": float(np.mean(rates)), "rate_std": float(np.std(rates)),
        }
        print(f"{n_ep}ep: MI={np.mean(mis):.4f}±{np.std(mis):.4f}bit, "
              f"信号送信率={np.mean(rates):.4f}±{np.std(rates):.4f}")

    summary = {
        "collision_early_mean": float(np.mean(coll_early)), "collision_early_std": float(np.std(coll_early)),
        "collision_late_mean": float(np.mean(coll_late)), "collision_late_std": float(np.std(coll_late)),
        "mi_by_checkpoint": {str(k): v for k, v in mi_summary.items()},
    }
    with open("noreward_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved noreward_summary.json")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    window = 200
    for d in data:
        arr = np.array(d["collision_rate_history"])
        smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(smoothed, alpha=0.6, color="#4472C4")
    axes[0, 0].set_xlabel("エピソード")
    axes[0, 0].set_ylabel("衝突率(移動平均200ep)")
    axes[0, 0].set_title("(0) 衝突率の学習推移(3系統、直接報酬なし)")

    ns = CHECKPOINT_EPISODES
    mi_means = [mi_summary[n]["mi_mean"] for n in ns]
    mi_stds = [mi_summary[n]["mi_std"] for n in ns]
    rate_means = [mi_summary[n]["rate_mean"] for n in ns]
    rate_stds = [mi_summary[n]["rate_std"] for n in ns]
    ax2 = axes[0, 1].twinx()
    axes[0, 1].errorbar(ns, mi_means, yerr=mi_stds, marker="o", color="#4472C4", label="MI(左軸)")
    ax2.errorbar(ns, rate_means, yerr=rate_stds, marker="s", color="#C0504D", label="信号送信率(右軸)")
    axes[0, 1].set_xlabel("学習量(episode数)")
    axes[0, 1].set_ylabel("I(signal;dominant_dev)[bit]", color="#4472C4")
    ax2.set_ylabel("信号送信率", color="#C0504D")
    axes[0, 1].set_title("(1) 信号のMI・送信率の推移(直接報酬なし)")
    lines1, labs1 = axes[0, 1].get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    axes[0, 1].legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    x = np.arange(2)
    axes[1, 0].bar(x, [np.mean(coll_early), np.mean(coll_late)],
                    yerr=[np.std(coll_early), np.std(coll_late)], color=["#BFBFBF", "#4472C4"])
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(["序盤(最初500ep)", "終盤(最後500ep)"])
    axes[1, 0].set_ylabel("衝突率")
    axes[1, 0].set_title("(0) 衝突率: 序盤 vs 終盤")

    # 比較: community_signal_v2(推測ゲームあり)のMI推移を参考線として重ねる
    try:
        v2_data = [json.load(open(f"community_v2_train_seed{s}.json")) for s in TRAJ_SEEDS]
        v2_cps = m.CHECKPOINT_EPISODES
        v2_mi_means = [np.mean([d["mi_by_checkpoint"][str(cp)]["mi"] for d in v2_data]) for cp in v2_cps]
        axes[1, 1].plot(v2_cps, v2_mi_means, "s--", color="#9BBB59", label="推測ゲームあり(v2、3500epまで)")
    except FileNotFoundError:
        pass
    axes[1, 1].plot(ns, mi_means, "o-", color="#4472C4", label="推測ゲームなし(本実験、8000epまで)")
    axes[1, 1].set_xlabel("学習量(episode数)")
    axes[1, 1].set_ylabel("I(signal;dominant_dev)[bit]")
    axes[1, 1].set_title("(2) 推測ゲームあり/なしでのMI推移の比較")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("要件6: 直接報酬なしでの信号創発の再検証(8000ep、n=3)")
    fig.tight_layout()
    fig.savefig("community_signal_noreward_comparison.png", dpi=150)
    print("グラフを community_signal_noreward_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train_chunk":
        train_chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "train_finalize":
        train_finalize(int(sys.argv[2]))
    elif cmd == "aggregate":
        aggregate()
