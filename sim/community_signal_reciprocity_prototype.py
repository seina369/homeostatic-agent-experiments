"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 「噛み合った」ことへの漠然とした報酬での信号創発の再検証
==========================================================

二度目の成功実験(community_signal_v2_prototype.py)は、推測ゲーム(相手の
内部状態を正しく当てたら双方にボーナス)という、評価指標そのものに近い
厳密な直接報酬に依存していた。これは人間の社会的コミュニケーションが持つ
「至近要因」(オキシトシン・ドーパミンによる、うまく関わり合えたこと自体への
漠然とした社会的報酬)を模した設計として正当化できなくはないが、実際の
社会的報酬は「相手の心的状態を正確に言い当てられたか」のような厳密な正誤
判定ではなく、もっと漠然とした「やり取りが噛み合った」という感覚に近い
はずである。

本プロトタイプは、GUESS_BONUS(正誤判定に基づく厳密な直接報酬、GuessAgent)を
完全に廃止し、代わりに「相互のやり取りが噛み合ったこと」への報酬
(RECIPROCITY_BONUS)に置き換える。内部状態の正誤には一切言及しない設計:

  - 同じステップで両者がsignal行動を選んだ場合(信号のやり取りが即座に
    往復した場合)、両者に小さなボーナスを与える。
  - あるいは、片方がsignalを送った後、RECIP_WINDOW(=3)ステップ以内に
    もう片方も(過去に)signalを送っていた場合、「呼びかけ-応答」が
    成立したとみなし、両者に同じボーナスを与える。

この緩やかな報酬だけで、信号と内部状態のMIが学習を通じて創発するか
(チャンスレートを上回るか)を、二度目の実験と同じ強い協調圧力(4×4グリッド・
衝突ペナルティ8.0)のもとで検証する。学習量は二度目の実験・直接報酬なし
実験とも比較しやすいよう3500epに揃えた。

環境・エージェントクラスはcommunity_signal_v2_prototype.pyのものを再利用する
(4x4グリッド・衝突ペナルティ8.0への変更は同モジュールのimport時に適用済み)。
GuessAgent・GUESS_BONUSは一切使わない。

学習系列の乱数(traj_seed=0,11,22)を変えた3系統で確認する。45秒のbash呼び出し
制限に対応するため、学習を1750epずつ2チャンクに分割する(合計3500ep)。

使い方:
  python3 community_signal_reciprocity_prototype.py train_chunk <traj_seed> <end_ep>
  python3 community_signal_reciprocity_prototype.py train_finalize <traj_seed>
  python3 community_signal_reciprocity_prototype.py aggregate
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
N_EPISODES = 3500
DECAY_EPISODES = 2500
CHECKPOINT_EPISODES = [300, 1500, 3500]
N_ROLLOUT_EPISODES = 100
ROLLOUT_EPS = 0.1

RECIP_WINDOW = 3     # 「呼びかけ」から何ステップ以内の信号を「応答」とみなすか
RECIP_BONUS = 0.5    # 噛み合ったことへの小さなボーナス(内部状態の正誤には一切言及しない)


def run_episode_reciprocity(env, agent0, agent1, eps0, eps1, learn0=True, learn1=True):
    """内部状態の正誤には一切言及せず、信号のやり取りが「噛み合ったこと」
    自体に小さなボーナスを与える学習エピソード。"""
    obs = env.reset()
    done = False
    devs, collisions, steps = [], 0, 0
    last_signal_step = [-999, -999]  # 各エージェントが最後にsignalを送ったステップ
    t = 0
    recip_events = 0
    while not done:
        a0 = m.act(agent0, obs[0], eps0)
        a1 = m.act(agent1, obs[1], eps1)
        signaled_now = [a0 == "signal", a1 == "signal"]

        bonus = [0.0, 0.0]
        if signaled_now[0] and signaled_now[1]:
            # 同時にsignalを送り合った(即座の往復)
            bonus[0] += RECIP_BONUS
            bonus[1] += RECIP_BONUS
            recip_events += 1
        else:
            for i in (0, 1):
                partner = 1 - i
                if signaled_now[i] and not signaled_now[partner]:
                    gap = t - last_signal_step[partner]
                    if 0 < gap <= RECIP_WINDOW:
                        # 相手の直近の「呼びかけ」に、時間差はあるが応答した
                        bonus[i] += RECIP_BONUS
                        bonus[partner] += RECIP_BONUS
                        recip_events += 1

        for i in (0, 1):
            if signaled_now[i]:
                last_signal_step[i] = t

        next_obs, base_rewards, done, deviations, collided = env.step([a0, a1])
        total_r0 = base_rewards[0] + bonus[0]
        total_r1 = base_rewards[1] + bonus[1]

        if learn0:
            agent0.update(obs[0], a0, total_r0, next_obs[0], done)
        if learn1:
            agent1.update(obs[1], a1, total_r1, next_obs[1], done)

        obs = next_obs
        devs.append((deviations[0] + deviations[1]) / 2.0)
        collisions += int(collided)
        steps += 1
        t += 1

    avg_dev = float(np.mean(devs))
    coll_rate = collisions / steps
    recip_rate = recip_events / steps
    return avg_dev, coll_rate, recip_rate


def train_range(env, agent0, agent1, start_ep, end_ep, decay_episodes, checkpoint_eps=None):
    checkpoint_eps = checkpoint_eps or []
    checkpoint_set = set(checkpoint_eps)
    checkpoints = {}
    avg_dev_hist, coll_hist, recip_hist = [], [], []
    for ep in range(start_ep, end_ep):
        eps = ib.epsilon_for_episode(ep, decay_episodes)
        avg_dev, coll_rate, recip_rate = run_episode_reciprocity(env, agent0, agent1, eps, eps)
        avg_dev_hist.append(avg_dev)
        coll_hist.append(coll_rate)
        recip_hist.append(recip_rate)
        if (ep + 1) in checkpoint_set:
            checkpoints[ep + 1] = (dict(agent0.q), dict(agent1.q))
    return avg_dev_hist, coll_hist, recip_hist, checkpoints


def train_chunk(traj_seed, end_ep):
    state_file = f"recip_state_seed{traj_seed}.pkl"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env = state["env"]
        agent0, agent1 = state["agent0"], state["agent1"]
        avg_dev_hist, coll_hist, recip_hist = state["avg_dev_hist"], state["coll_hist"], state["recip_hist"]
        all_checkpoints = state["checkpoints"]
        start_ep = state["last_ep"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
        print(f"[噛み合いseed={traj_seed}] 再開: {start_ep}ep目から{end_ep}epまで学習")
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        agent0, agent1 = QLearningAgent(), QLearningAgent()
        avg_dev_hist, coll_hist, recip_hist = [], [], []
        all_checkpoints = {}
        start_ep = 0
        print(f"[噛み合いseed={traj_seed}] 新規開始: 0ep目から{end_ep}epまで学習"
              f"(GUESS_BONUSなし、RECIP_BONUS={RECIP_BONUS}, window={RECIP_WINDOW})")

    dev_h, coll_h, recip_h, checkpoints = train_range(
        env, agent0, agent1, start_ep, end_ep, DECAY_EPISODES, checkpoint_eps=CHECKPOINT_EPISODES
    )
    avg_dev_hist.extend(dev_h); coll_hist.extend(coll_h); recip_hist.extend(recip_h)
    all_checkpoints.update(checkpoints)

    state = {
        "env": env, "agent0": agent0, "agent1": agent1,
        "avg_dev_hist": avg_dev_hist, "coll_hist": coll_hist, "recip_hist": recip_hist,
        "checkpoints": all_checkpoints, "last_ep": end_ep,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)
    print(f"[噛み合いseed={traj_seed}] {end_ep}epまで完了・保存 "
          f"(直近100ep衝突率={np.mean(coll_hist[-100:]):.4f}, 噛み合い率={np.mean(recip_hist[-100:]):.4f})")


def train_finalize(traj_seed):
    state_file = f"recip_state_seed{traj_seed}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    agent0, agent1 = state["agent0"], state["agent1"]
    avg_dev_hist, coll_hist, recip_hist = state["avg_dev_hist"], state["coll_hist"], state["recip_hist"]
    checkpoints = state["checkpoints"]

    print(f"[噛み合いseed={traj_seed}] === 土台: 衝突回避タスク自体の改善 ===")
    print(f"[噛み合いseed={traj_seed}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[:500]):.4f}, 噛み合い率={np.mean(recip_hist[:500]):.4f}")
    print(f"[噛み合いseed={traj_seed}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[-500:]):.4f}, 噛み合い率={np.mean(recip_hist[-500:]):.4f}")

    mi_by_checkpoint = {}
    for n_ep in CHECKPOINT_EPISODES:
        q0, q1 = checkpoints[n_ep]
        random.seed(traj_seed * 9000 + n_ep)
        np.random.seed(traj_seed * 9000 + n_ep)
        rollout_env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        records = []
        agent0_r = QLearningAgent(); agent0_r.q = dict(q0)
        agent1_r = QLearningAgent(); agent1_r.q = dict(q1)
        for _ in range(N_ROLLOUT_EPISODES):
            obs = rollout_env.reset()
            done = False
            while not done:
                dom0 = rollout_env.dominant_deviation(0)
                dom1 = rollout_env.dominant_deviation(1)
                a0 = m.act(agent0_r, obs[0], ROLLOUT_EPS)
                a1 = m.act(agent1_r, obs[1], ROLLOUT_EPS)
                records.append((dom0, 1 if a0 == "signal" else 0))
                records.append((dom1, 1 if a1 == "signal" else 0))
                next_obs, rewards, done, deviations, collided = rollout_env.step([a0, a1])
                obs = next_obs
        mi, signal_rate, cond_dist, marg_dist = m.mutual_info_signal_vs_class(records)
        mi_by_checkpoint[str(n_ep)] = {
            "mi": mi, "signal_rate": signal_rate, "cond_dist_given_signal": cond_dist, "marginal_dist": marg_dist,
        }
        print(f"[噛み合いseed={traj_seed}] {n_ep}ep: I(signal;dominant_dev)={mi:.4f}bit, "
              f"信号送信率={signal_rate:.4f}, signal時分布={cond_dist}, 全体分布={marg_dist}")

    result = {
        "traj_seed": traj_seed,
        "avg_dev_history": avg_dev_hist, "collision_rate_history": coll_hist, "recip_rate_history": recip_hist,
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(f"recip_train_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved recip_train_seed{traj_seed}.json")


def aggregate():
    data = [json.load(open(f"recip_train_seed{s}.json")) for s in TRAJ_SEEDS]

    print("=== (0) 土台: 衝突回避タスク自体の改善(n=3の平均±標準偏差) ===")
    coll_early = [np.mean(d["collision_rate_history"][:500]) for d in data]
    coll_late = [np.mean(d["collision_rate_history"][-500:]) for d in data]
    recip_early = [np.mean(d["recip_rate_history"][:500]) for d in data]
    recip_late = [np.mean(d["recip_rate_history"][-500:]) for d in data]
    print(f"衝突率: 序盤(最初500ep)={np.mean(coll_early):.4f}±{np.std(coll_early):.4f}, "
          f"終盤(最後500ep)={np.mean(coll_late):.4f}±{np.std(coll_late):.4f}")
    print(f"噛み合い率: 序盤={np.mean(recip_early):.4f}±{np.std(recip_early):.4f}, "
          f"終盤={np.mean(recip_late):.4f}±{np.std(recip_late):.4f}")

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
        print(f"{n_ep}ep: MI={np.mean(mis):.4f}±{np.std(mis):.4f}bit, 信号送信率={np.mean(rates):.4f}±{np.std(rates):.4f}")

    v2_data, noreward_data = None, None
    try:
        v2_data = [json.load(open(f"community_v2_train_seed{s}.json")) for s in TRAJ_SEEDS]
    except FileNotFoundError:
        pass
    try:
        noreward_data = [json.load(open(f"noreward_train_seed{s}.json")) for s in TRAJ_SEEDS]
    except FileNotFoundError:
        pass

    summary = {
        "collision_early_mean": float(np.mean(coll_early)), "collision_early_std": float(np.std(coll_early)),
        "collision_late_mean": float(np.mean(coll_late)), "collision_late_std": float(np.std(coll_late)),
        "recip_early_mean": float(np.mean(recip_early)), "recip_late_mean": float(np.mean(recip_late)),
        "mi_by_checkpoint": {str(k): v for k, v in mi_summary.items()},
    }
    with open("recip_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved recip_summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ns = CHECKPOINT_EPISODES
    mi_means = [mi_summary[n]["mi_mean"] for n in ns]
    mi_stds = [mi_summary[n]["mi_std"] for n in ns]
    axes[0].errorbar(ns, mi_means, yerr=mi_stds, marker="o", color="#4472C4", label="噛み合い報酬のみ(本実験)")
    if v2_data is not None:
        v2_mi_means = [np.mean([d["mi_by_checkpoint"][str(cp)]["mi"] for d in v2_data]) for cp in ns]
        v2_mi_stds = [np.std([d["mi_by_checkpoint"][str(cp)]["mi"] for d in v2_data]) for cp in ns]
        axes[0].errorbar(ns, v2_mi_means, yerr=v2_mi_stds, marker="s", color="#9BBB59", label="推測ゲーム直接報酬(v2)")
    if noreward_data is not None:
        nr_cps = [500, 1500, 3000, 5000, 8000]
        nr_mi_means = [np.mean([d["mi_by_checkpoint"][str(cp)]["mi"] for d in noreward_data]) for cp in nr_cps]
        axes[0].plot(nr_cps, nr_mi_means, "^--", color="#C0504D", label="直接報酬なし(8000epまで)")
    axes[0].set_xlabel("学習量(episode数)")
    axes[0].set_ylabel("I(signal;dominant_dev)[bit]")
    axes[0].set_title("報酬設計とMIの比較")
    axes[0].legend(fontsize=8)

    x = np.arange(2)
    axes[1].bar(x - 0.2, [np.mean(coll_early), np.mean(coll_late)], width=0.35,
                yerr=[np.std(coll_early), np.std(coll_late)], color="#4472C4", label="衝突率")
    ax2 = axes[1].twinx()
    ax2.bar(x + 0.2, [np.mean(recip_early), np.mean(recip_late)], width=0.35,
            color="#9BBB59", label="噛み合い率")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["序盤(最初500ep)", "終盤(最後500ep)"])
    axes[1].set_ylabel("衝突率", color="#4472C4")
    ax2.set_ylabel("噛み合い率", color="#9BBB59")
    axes[1].set_title("衝突率・噛み合い率: 序盤 vs 終盤")

    fig.suptitle("要件6: 「噛み合った」ことへの漠然とした報酬での信号創発の検証")
    fig.tight_layout()
    fig.savefig("community_signal_reciprocity_comparison.png", dpi=150)
    print("グラフを community_signal_reciprocity_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train_chunk":
        train_chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "train_finalize":
        train_finalize(int(sys.argv[2]))
    elif cmd == "aggregate":
        aggregate()
