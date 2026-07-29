"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 ハイブリッド報酬(噛み合い+推測ゲーム)での信号創発の検証
==========================================================

これまでの2つの報酬設計を比較すると:
  - 推測ゲーム(GUESS_BONUS、正誤判定): 学習後期(3500ep)に安定して高いMI
    (0.0224bit)へ単調に育つが、立ち上がりが遅い(300epでは0.0053bit)。
  - 噛み合い報酬(RECIPROCITY_BONUS、正誤に無関係): 学習初期(300ep)に
    これまでで最高のMI(0.1028bit)へ非常に速く立ち上がるが、学習を続けると
    シグナリングゲーム理論の「プーリング均衡」(内部状態によらず頻繁に信号を
    送り合う)へ収束し、MIはむしろ低下する(3500epで0.0222bit)。

本プロトタイプは、この2つを同時に報酬へ組み込んだハイブリッド設計を検証する。
  (a) 双方向の信号交換自体への小さなボーナス(前回の噛み合い報酬、ただし
      係数を0.5→0.15に下げて「種火」程度に弱め、単独でプーリング均衡を
      支配的にしないようにする)。
  (b) 推測ゲームによる正誤判定ボーナス(前回のGUESS_BONUS=1.0、そのまま)。
両方を同時に報酬に加えることで、学習初期は噛み合い報酬が信号使用そのものを
後押しして立ち上がりを速め、学習後期は推測ゲームの正誤判定が「内容の正しさ」
への継続的な圧力として働き、プーリング均衡への崩壊を防ぐことを狙う。

二度目の実験・噛み合い実験と同じ強い協調圧力(4×4グリッド・衝突ペナルティ8.0)・
同じ学習量(3500ep)のもとで、3系統(traj_seed=0,11,22)で検証する。

使い方:
  python3 community_signal_hybrid_prototype.py train_chunk <traj_seed> <end_ep>
  python3 community_signal_hybrid_prototype.py train_finalize <traj_seed>
  python3 community_signal_hybrid_prototype.py aggregate
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
import community_signal_v2_prototype as m  # GRID_SIZE=4, ACTIONS_COMM, GuessAgent, COLLISION_PENALTY=8.0を継承

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAJ_SEEDS = [0, 11, 22]
N_EPISODES = 3500
DECAY_EPISODES = 2500
CHECKPOINT_EPISODES = [300, 1500, 3500]
N_ROLLOUT_EPISODES = 100
ROLLOUT_EPS = 0.1
GUESS_EPS = m.GUESS_EPS

GUESS_BONUS = m.GUESS_BONUS       # 1.0、v2と同一
RECIP_WINDOW = 3
RECIP_BONUS_HYBRID = 0.15         # 前回の0.5から下げた「種火」程度の弱いボーナス


def run_episode_hybrid(env, agent0, agent1, guess0, guess1, eps0, eps1,
                        learn0=True, learn1=True, learn_guess0=True, learn_guess1=True, guess_eps=GUESS_EPS):
    obs = env.reset()
    done = False
    devs, collisions, steps = [], 0, 0
    correct0_count, correct1_count, n_guesses = 0, 0, 0
    last_signal_step = [-999, -999]
    recip_events = 0
    t = 0
    while not done:
        dom0 = env.dominant_deviation(0)
        dom1 = env.dominant_deviation(1)

        a0 = m.act(agent0, obs[0], eps0)
        a1 = m.act(agent1, obs[1], eps1)

        # --- (b) 推測ゲーム(正誤判定)ボーナス ---
        sig_for_guess0 = obs[0][6]
        sig_for_guess1 = obs[1][6]
        guess0_val = guess0.act(sig_for_guess0, guess_eps)
        guess1_val = guess1.act(sig_for_guess1, guess_eps)
        correct0 = int(guess0_val == dom1)
        correct1 = int(guess1_val == dom0)
        correct0_count += correct0
        correct1_count += correct1
        n_guesses += 1

        # --- (a) 噛み合い(正誤に無関係)ボーナス ---
        signaled_now = [a0 == "signal", a1 == "signal"]
        recip_bonus = [0.0, 0.0]
        if signaled_now[0] and signaled_now[1]:
            recip_bonus[0] += RECIP_BONUS_HYBRID
            recip_bonus[1] += RECIP_BONUS_HYBRID
            recip_events += 1
        else:
            for i in (0, 1):
                partner = 1 - i
                if signaled_now[i] and not signaled_now[partner]:
                    gap = t - last_signal_step[partner]
                    if 0 < gap <= RECIP_WINDOW:
                        recip_bonus[i] += RECIP_BONUS_HYBRID
                        recip_bonus[partner] += RECIP_BONUS_HYBRID
                        recip_events += 1
        for i in (0, 1):
            if signaled_now[i]:
                last_signal_step[i] = t

        next_obs, base_rewards, done, deviations, collided = env.step([a0, a1])

        total_r0 = base_rewards[0] + GUESS_BONUS * correct0 + GUESS_BONUS * correct1 + recip_bonus[0]
        total_r1 = base_rewards[1] + GUESS_BONUS * correct1 + GUESS_BONUS * correct0 + recip_bonus[1]

        if learn0:
            agent0.update(obs[0], a0, total_r0, next_obs[0], done)
        if learn1:
            agent1.update(obs[1], a1, total_r1, next_obs[1], done)
        if learn_guess0:
            guess0.update(sig_for_guess0, guess0_val, GUESS_BONUS * correct0)
        if learn_guess1:
            guess1.update(sig_for_guess1, guess1_val, GUESS_BONUS * correct1)

        obs = next_obs
        devs.append((deviations[0] + deviations[1]) / 2.0)
        collisions += int(collided)
        steps += 1
        t += 1

    avg_dev = float(np.mean(devs))
    coll_rate = collisions / steps
    guess_acc = (correct0_count + correct1_count) / (2 * n_guesses)
    recip_rate = recip_events / steps
    return avg_dev, coll_rate, guess_acc, recip_rate


def train_range(env, agent0, agent1, guess0, guess1, start_ep, end_ep, decay_episodes, checkpoint_eps=None):
    checkpoint_eps = checkpoint_eps or []
    checkpoint_set = set(checkpoint_eps)
    checkpoints = {}
    avg_dev_hist, coll_hist, guess_acc_hist, recip_hist = [], [], [], []
    for ep in range(start_ep, end_ep):
        eps = ib.epsilon_for_episode(ep, decay_episodes)
        avg_dev, coll_rate, guess_acc, recip_rate = run_episode_hybrid(env, agent0, agent1, guess0, guess1, eps, eps)
        avg_dev_hist.append(avg_dev)
        coll_hist.append(coll_rate)
        guess_acc_hist.append(guess_acc)
        recip_hist.append(recip_rate)
        if (ep + 1) in checkpoint_set:
            checkpoints[ep + 1] = (dict(agent0.q), dict(agent1.q), dict(guess0.q), dict(guess1.q))
    return avg_dev_hist, coll_hist, guess_acc_hist, recip_hist, checkpoints


def train_chunk(traj_seed, end_ep):
    state_file = f"hybrid_state_seed{traj_seed}.pkl"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env = state["env"]
        agent0, agent1 = state["agent0"], state["agent1"]
        guess0, guess1 = state["guess0"], state["guess1"]
        avg_dev_hist, coll_hist, guess_acc_hist, recip_hist = (
            state["avg_dev_hist"], state["coll_hist"], state["guess_acc_hist"], state["recip_hist"]
        )
        all_checkpoints = state["checkpoints"]
        start_ep = state["last_ep"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
        print(f"[ハイブリッドseed={traj_seed}] 再開: {start_ep}ep目から{end_ep}epまで学習")
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        agent0, agent1 = QLearningAgent(), QLearningAgent()
        guess0, guess1 = m.GuessAgent(), m.GuessAgent()
        avg_dev_hist, coll_hist, guess_acc_hist, recip_hist = [], [], [], []
        all_checkpoints = {}
        start_ep = 0
        print(f"[ハイブリッドseed={traj_seed}] 新規開始: 0ep目から{end_ep}epまで学習"
              f"(GUESS_BONUS={GUESS_BONUS} + RECIP_BONUS={RECIP_BONUS_HYBRID})")

    dev_h, coll_h, gacc_h, recip_h, checkpoints = train_range(
        env, agent0, agent1, guess0, guess1, start_ep, end_ep, DECAY_EPISODES, checkpoint_eps=CHECKPOINT_EPISODES
    )
    avg_dev_hist.extend(dev_h); coll_hist.extend(coll_h); guess_acc_hist.extend(gacc_h); recip_hist.extend(recip_h)
    all_checkpoints.update(checkpoints)

    state = {
        "env": env, "agent0": agent0, "agent1": agent1, "guess0": guess0, "guess1": guess1,
        "avg_dev_hist": avg_dev_hist, "coll_hist": coll_hist, "guess_acc_hist": guess_acc_hist, "recip_hist": recip_hist,
        "checkpoints": all_checkpoints, "last_ep": end_ep,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)
    print(f"[ハイブリッドseed={traj_seed}] {end_ep}epまで完了・保存 "
          f"(直近100ep衝突率={np.mean(coll_hist[-100:]):.4f}, 推測精度={np.mean(guess_acc_hist[-100:]):.4f}, "
          f"噛み合い率={np.mean(recip_hist[-100:]):.4f})")


def train_finalize(traj_seed):
    state_file = f"hybrid_state_seed{traj_seed}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    agent0, agent1 = state["agent0"], state["agent1"]
    guess0, guess1 = state["guess0"], state["guess1"]
    avg_dev_hist, coll_hist, guess_acc_hist, recip_hist = (
        state["avg_dev_hist"], state["coll_hist"], state["guess_acc_hist"], state["recip_hist"]
    )
    checkpoints = state["checkpoints"]

    print(f"[ハイブリッドseed={traj_seed}] === 土台: 衝突回避タスク自体の改善 ===")
    print(f"[ハイブリッドseed={traj_seed}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[:500]):.4f}, 推測精度={np.mean(guess_acc_hist[:500]):.4f}, "
          f"噛み合い率={np.mean(recip_hist[:500]):.4f}")
    print(f"[ハイブリッドseed={traj_seed}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[-500:]):.4f}, 推測精度={np.mean(guess_acc_hist[-500:]):.4f}, "
          f"噛み合い率={np.mean(recip_hist[-500:]):.4f}")

    mi_by_checkpoint = {}
    for n_ep in CHECKPOINT_EPISODES:
        q0, q1, gq0, gq1 = checkpoints[n_ep]
        random.seed(traj_seed * 11000 + n_ep)
        np.random.seed(traj_seed * 11000 + n_ep)
        rollout_env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        records, guess_correct = m.rollout_for_signal_analysis(
            rollout_env, dict(q0), dict(q1), dict(gq0), dict(gq1), N_ROLLOUT_EPISODES, ROLLOUT_EPS
        )
        mi, signal_rate, cond_dist, marg_dist = m.mutual_info_signal_vs_class(records)
        guess_acc = float(np.mean(guess_correct))
        mi_by_checkpoint[str(n_ep)] = {
            "mi": mi, "signal_rate": signal_rate, "cond_dist_given_signal": cond_dist,
            "marginal_dist": marg_dist, "guess_accuracy": guess_acc,
        }
        print(f"[ハイブリッドseed={traj_seed}] {n_ep}ep: I(signal;dominant_dev)={mi:.4f}bit, "
              f"信号送信率={signal_rate:.4f}, 推測精度={guess_acc:.4f}(チャンス=0.333), "
              f"signal時分布={cond_dist}, 全体分布={marg_dist}")

    result = {
        "traj_seed": traj_seed,
        "avg_dev_history": avg_dev_hist, "collision_rate_history": coll_hist,
        "guess_acc_history": guess_acc_hist, "recip_rate_history": recip_hist,
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(f"hybrid_train_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved hybrid_train_seed{traj_seed}.json")


def aggregate():
    data = [json.load(open(f"hybrid_train_seed{s}.json")) for s in TRAJ_SEEDS]
    recip_data, v2_data = None, None
    try:
        recip_data = [json.load(open(f"recip_train_seed{s}.json")) for s in TRAJ_SEEDS]
    except FileNotFoundError:
        pass
    try:
        v2_data = [json.load(open(f"community_v2_train_seed{s}.json")) for s in TRAJ_SEEDS]
    except FileNotFoundError:
        pass

    print("=== (0) 土台: 衝突回避タスク自体の改善(n=3の平均±標準偏差) ===")
    coll_early = [np.mean(d["collision_rate_history"][:500]) for d in data]
    coll_late = [np.mean(d["collision_rate_history"][-500:]) for d in data]
    print(f"衝突率: 序盤(最初500ep)={np.mean(coll_early):.4f}±{np.std(coll_early):.4f}, "
          f"終盤(最後500ep)={np.mean(coll_late):.4f}±{np.std(coll_late):.4f}")

    print("\n=== (1) 信号と内部状態のMI・信号送信率・推測精度(チェックポイント別、n=3の平均±標準偏差) ===")
    mi_summary = {}
    for n_ep in CHECKPOINT_EPISODES:
        key = str(n_ep)
        mis = [d["mi_by_checkpoint"][key]["mi"] for d in data]
        rates = [d["mi_by_checkpoint"][key]["signal_rate"] for d in data]
        gaccs = [d["mi_by_checkpoint"][key]["guess_accuracy"] for d in data]
        mi_summary[n_ep] = {
            "mi_mean": float(np.mean(mis)), "mi_std": float(np.std(mis)),
            "rate_mean": float(np.mean(rates)), "rate_std": float(np.std(rates)),
            "gacc_mean": float(np.mean(gaccs)), "gacc_std": float(np.std(gaccs)),
        }
        print(f"{n_ep}ep: MI={np.mean(mis):.4f}±{np.std(mis):.4f}bit, 信号送信率={np.mean(rates):.4f}±{np.std(rates):.4f}, "
              f"推測精度={np.mean(gaccs):.4f}±{np.std(gaccs):.4f}")

    print("\n=== (2) 目標達成状況 ===")
    print(f"目標1(300epのMIが噛み合い単独0.1028bitに近いか): ハイブリッド300ep MI={mi_summary[300]['mi_mean']:.4f}bit")
    print(f"目標2(3500epのMIが推測ゲーム単独0.0224bit以上か): ハイブリッド3500ep MI={mi_summary[3500]['mi_mean']:.4f}bit")

    summary = {
        "collision_early_mean": float(np.mean(coll_early)), "collision_early_std": float(np.std(coll_early)),
        "collision_late_mean": float(np.mean(coll_late)), "collision_late_std": float(np.std(coll_late)),
        "mi_by_checkpoint": {str(k): v for k, v in mi_summary.items()},
    }
    with open("hybrid_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved hybrid_summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ns = CHECKPOINT_EPISODES
    mi_means = [mi_summary[n]["mi_mean"] for n in ns]
    mi_stds = [mi_summary[n]["mi_std"] for n in ns]
    axes[0].errorbar(ns, mi_means, yerr=mi_stds, marker="o", color="#4472C4", label="ハイブリッド(本実験)")
    if recip_data is not None:
        r_means = [np.mean([d["mi_by_checkpoint"][str(cp)]["mi"] for d in recip_data]) for cp in ns]
        axes[0].plot(ns, r_means, "^--", color="#C0504D", label="噛み合い報酬単独")
    if v2_data is not None:
        v_means = [np.mean([d["mi_by_checkpoint"][str(cp)]["mi"] for d in v2_data]) for cp in ns]
        axes[0].plot(ns, v_means, "s--", color="#9BBB59", label="推測ゲーム単独(v2)")
    axes[0].set_xlabel("学習量(episode数)")
    axes[0].set_ylabel("I(signal;dominant_dev)[bit]")
    axes[0].set_title("3種類の報酬設計でのMI推移比較")
    axes[0].legend(fontsize=8)

    x = np.arange(2)
    axes[1].bar(x, [np.mean(coll_early), np.mean(coll_late)],
                yerr=[np.std(coll_early), np.std(coll_late)], color=["#BFBFBF", "#4472C4"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["序盤(最初500ep)", "終盤(最後500ep)"])
    axes[1].set_ylabel("衝突率")
    axes[1].set_title("衝突率: 序盤 vs 終盤(ハイブリッド)")

    fig.suptitle("要件6: ハイブリッド報酬(噛み合い+推測ゲーム)での信号創発の検証")
    fig.tight_layout()
    fig.savefig("community_signal_hybrid_comparison.png", dpi=150)
    print("グラフを community_signal_hybrid_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train_chunk":
        train_chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "train_finalize":
        train_finalize(int(sys.argv[2]))
    elif cmd == "aggregate":
        aggregate()
