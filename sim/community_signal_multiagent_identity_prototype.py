"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 集団規模(3体)+送り手身元付き推測表での信号創発の検証
==========================================================

集団規模(3体)実験(community_signal_multiagent_prototype.py)で、3500ep時点の
MIが2体(v2)の0.0224bitの約7分の1(0.0032bit)に低下するという、言語進化
文献の予測と正反対の結果が得られた。その原因として「受け手のGuessAgentが
送り手の身元を区別しない単一の対応表を持っているため、互いに収束していない
複数の送り手の符号化方針を平均して解釈してしまい、規約が強化ではなく希薄化
した」という仮説を立てた。

本プロトタイプは、この仮説を検証するため、GuessAgentの状態表現に「どの
相手からの信号か」という送り手の身元(identity)を追加した
IdentityGuessAgentを実装する。各エージェントは、隣人ごとに別々の
(sender_id, signal) -> guessの対応表を学習できるようになる。

それ以外の条件は集団規模実験と完全に同一: 4×4グリッド、衝突ペナルティ8.0、
推測ゲームによる直接報酬(GUESS_BONUS)、3体構成、3500ep(v2と直接比較の
ため)、学習系列の乱数(traj_seed=0,11,22)を変えた3系統。

検証したいのは、この修正だけで各ペア単位のMI(iが隣人jの信号から隣人の
支配的逸脱クラスを推測する際のMI、jの身元ごとに分解)が2体構成(0.0224bit)
並みの水準まで回復するかどうか。

使い方:
  python3 community_signal_multiagent_identity_prototype.py train_chunk <traj_seed> <end_ep>
  python3 community_signal_multiagent_identity_prototype.py train_finalize <traj_seed>
  python3 community_signal_multiagent_identity_prototype.py aggregate
"""

import sys, json, pickle, random
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import homeostasis_prototype as hp
from homeostasis_prototype import QLearningAgent
import instinct_bias_prototype as ib
import community_signal_v2_prototype as m  # GRID_SIZE=4, ACTIONS_COMM, GUESS_CLASSES, ALPHA, mutual_info_signal_vs_class等を再利用
import community_signal_multiagent_prototype as ma  # MultiAgentHomeostasisEnvN(身元非依存版との比較用)

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

N_AGENTS = 3
TRAJ_SEEDS = [0, 11, 22]
N_EPISODES = 3500
DECAY_EPISODES = 2500
CHECKPOINT_EPISODES = [300, 1500, 3500]
COLLISION_PENALTY = m.COLLISION_PENALTY  # 8.0、v2・身元非依存版と同一
GUESS_BONUS = m.GUESS_BONUS
GUESS_EPS = m.GUESS_EPS
N_ROLLOUT_EPISODES = 100
ROLLOUT_EPS = 0.1


class IdentityGuessAgent:
    """送り手の身元(sender_id)を状態に含めるGuessAgent。相手ごとに別々の
    (sender_id, signal) -> guessの対応表を学習できる。GuessAgentとの唯一の
    違いは状態表現にsender_idが追加された点で、学習アルゴリズム(バンディット
    型のQ値更新)自体はGuessAgentと同一。"""

    def __init__(self):
        self.q = {}

    def q_value(self, sender_id, sig, guess):
        return self.q.get((sender_id, sig, guess), 0.0)

    def best_guess(self, sender_id, sig):
        values = [self.q_value(sender_id, sig, g) for g in m.GUESS_CLASSES]
        return m.GUESS_CLASSES[int(np.argmax(values))]

    def act(self, sender_id, sig, eps):
        if random.random() < eps:
            return random.choice(m.GUESS_CLASSES)
        return self.best_guess(sender_id, sig)

    def update(self, sender_id, sig, guess, reward):
        current = self.q_value(sender_id, sig, guess)
        self.q[(sender_id, sig, guess)] = current + m.ALPHA * (reward - current)


def run_episode(env, agents, guesses, eps_list, learn=True, learn_guess=True, guess_eps=GUESS_EPS):
    obs = env.reset()
    done = False
    devs, collisions, steps = [], 0, 0
    correct_counts = [0] * env.n
    n_guesses = 0
    while not done:
        doms = [env.dominant_deviation(i) for i in range(env.n)]
        neighbors = [env.nearest_agent(i) for i in range(env.n)]
        actions = [m.act(agents[i], obs[i], eps_list[i]) for i in range(env.n)]

        guess_vals = []
        corrects = []
        for i in range(env.n):
            j = neighbors[i]
            sig_for_guess = obs[i][6]  # 隣人jの直前signal
            gv = guesses[i].act(j, sig_for_guess, guess_eps)  # 身元jを状態に含めて推測
            guess_vals.append(gv)
            corrects.append(int(gv == doms[j]))
        correct_counts = [c + corr for c, corr in zip(correct_counts, corrects)]
        n_guesses += 1

        next_obs, base_rewards, done, deviations, any_collision, collided_flags = env.step(actions)

        bonus = [0.0] * env.n
        for i in range(env.n):
            bonus[i] += GUESS_BONUS * corrects[i]  # iが隣人jを当てた
        for k in range(env.n):
            j = neighbors[k]
            bonus[j] += GUESS_BONUS * corrects[k]  # jはkに当てられたボーナスも受け取る

        total_rewards = [base_rewards[i] + bonus[i] for i in range(env.n)]

        if learn:
            for i in range(env.n):
                agents[i].update(obs[i], actions[i], total_rewards[i], next_obs[i], done)
        if learn_guess:
            for i in range(env.n):
                j = neighbors[i]
                guesses[i].update(j, obs[i][6], guess_vals[i], GUESS_BONUS * corrects[i])

        obs = next_obs
        devs.append(float(np.mean(deviations)))
        collisions += int(any_collision)
        steps += 1

    avg_dev = float(np.mean(devs))
    coll_rate = collisions / steps
    guess_acc = sum(correct_counts) / (env.n * n_guesses)
    return avg_dev, coll_rate, guess_acc


def train_range(env, agents, guesses, start_ep, end_ep, decay_episodes, checkpoint_eps=None):
    checkpoint_eps = checkpoint_eps or []
    checkpoint_set = set(checkpoint_eps)
    checkpoints = {}
    avg_dev_hist, coll_hist, guess_acc_hist = [], [], []
    for ep in range(start_ep, end_ep):
        eps = ib.epsilon_for_episode(ep, decay_episodes)
        eps_list = [eps] * env.n
        avg_dev, coll_rate, guess_acc = run_episode(env, agents, guesses, eps_list)
        avg_dev_hist.append(avg_dev)
        coll_hist.append(coll_rate)
        guess_acc_hist.append(guess_acc)
        if (ep + 1) in checkpoint_set:
            checkpoints[ep + 1] = ([dict(a.q) for a in agents], [dict(g.q) for g in guesses])
    return avg_dev_hist, coll_hist, guess_acc_hist, checkpoints


def train_chunk(traj_seed, end_ep):
    state_file = f"multiagent_identity_state_seed{traj_seed}.pkl"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env = state["env"]
        agents = state["agents"]
        guesses = state["guesses"]
        avg_dev_hist, coll_hist, guess_acc_hist = state["avg_dev_hist"], state["coll_hist"], state["guess_acc_hist"]
        all_checkpoints = state["checkpoints"]
        start_ep = state["last_ep"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
        print(f"[身元付きN={N_AGENTS}体 seed={traj_seed}] 再開: {start_ep}ep目から{end_ep}epまで学習")
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        env = ma.MultiAgentHomeostasisEnvN(random.Random(m.TRAIN_SEED), n_agents=N_AGENTS)
        agents = [QLearningAgent() for _ in range(N_AGENTS)]
        guesses = [IdentityGuessAgent() for _ in range(N_AGENTS)]
        avg_dev_hist, coll_hist, guess_acc_hist = [], [], []
        all_checkpoints = {}
        start_ep = 0
        print(f"[身元付きN={N_AGENTS}体 seed={traj_seed}] 新規開始: 0ep目から{end_ep}epまで学習")

    dev_h, coll_h, gacc_h, checkpoints = train_range(
        env, agents, guesses, start_ep, end_ep, DECAY_EPISODES, checkpoint_eps=CHECKPOINT_EPISODES
    )
    avg_dev_hist.extend(dev_h); coll_hist.extend(coll_h); guess_acc_hist.extend(gacc_h)
    all_checkpoints.update(checkpoints)

    state = {
        "env": env, "agents": agents, "guesses": guesses,
        "avg_dev_hist": avg_dev_hist, "coll_hist": coll_hist, "guess_acc_hist": guess_acc_hist,
        "checkpoints": all_checkpoints, "last_ep": end_ep,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)
    print(f"[身元付きN={N_AGENTS}体 seed={traj_seed}] {end_ep}epまで完了・保存 "
          f"(直近100ep衝突率={np.mean(coll_hist[-100:]):.4f}, 推測精度={np.mean(guess_acc_hist[-100:]):.4f})")


def rollout_for_signal_analysis_pairwise(env, agent_qs, guess_qs, n_episodes, eps):
    """身元非依存版と同じ全体プールMI(比較用)に加え、送り手身元ごとに分解した
    ペア単位のMIも計算できるよう、(sender_id, dominant_dev, signal)のレコードを集める。"""
    agents = []
    guesses = []
    for q in agent_qs:
        a = QLearningAgent(); a.q = dict(q); agents.append(a)
    for q in guess_qs:
        g = IdentityGuessAgent(); g.q = dict(q); guesses.append(g)
    pooled_records = []          # (dom, signal) 身元非依存版と同じ形式(全体プール)
    pairwise_records = {}        # sender_id -> [(dom, signal), ...] 送り手ごとの記録
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            doms = [env.dominant_deviation(i) for i in range(env.n)]
            actions = [m.act(agents[i], obs[i], eps) for i in range(env.n)]
            for i in range(env.n):
                sig = 1 if actions[i] == "signal" else 0
                pooled_records.append((doms[i], sig))
                pairwise_records.setdefault(i, []).append((doms[i], sig))
            next_obs, rewards, done, deviations, any_collision, collided_flags = env.step(actions)
            obs = next_obs
    return pooled_records, pairwise_records


def train_finalize(traj_seed):
    state_file = f"multiagent_identity_state_seed{traj_seed}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    agents, guesses = state["agents"], state["guesses"]
    avg_dev_hist, coll_hist, guess_acc_hist = state["avg_dev_hist"], state["coll_hist"], state["guess_acc_hist"]
    checkpoints = state["checkpoints"]

    print(f"[身元付きN={N_AGENTS}体 seed={traj_seed}] === 土台: 衝突回避タスク自体の改善 ===")
    print(f"[身元付きN={N_AGENTS}体 seed={traj_seed}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[:500]):.4f}, 推測精度={np.mean(guess_acc_hist[:500]):.4f}")
    print(f"[身元付きN={N_AGENTS}体 seed={traj_seed}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[-500:]):.4f}, 推測精度={np.mean(guess_acc_hist[-500:]):.4f}")

    mi_by_checkpoint = {}
    for n_ep in CHECKPOINT_EPISODES:
        agent_qs, guess_qs = checkpoints[n_ep]
        random.seed(traj_seed * 7000 + n_ep)
        np.random.seed(traj_seed * 7000 + n_ep)
        rollout_env = ma.MultiAgentHomeostasisEnvN(random.Random(m.TRAIN_SEED), n_agents=N_AGENTS)
        pooled_records, pairwise_records = rollout_for_signal_analysis_pairwise(
            rollout_env, agent_qs, guess_qs, N_ROLLOUT_EPISODES, ROLLOUT_EPS
        )
        mi_pooled, rate_pooled, cond_pooled, marg_pooled = m.mutual_info_signal_vs_class(pooled_records)

        pairwise_mi = {}
        for sender_id, recs in pairwise_records.items():
            if len(recs) < 10:
                continue
            mi_p, rate_p, cond_p, marg_p = m.mutual_info_signal_vs_class(recs)
            pairwise_mi[str(sender_id)] = {"mi": mi_p, "signal_rate": rate_p, "n_records": len(recs)}

        mi_by_checkpoint[str(n_ep)] = {
            "mi_pooled": mi_pooled, "signal_rate_pooled": rate_pooled,
            "cond_dist_given_signal": cond_pooled, "marginal_dist": marg_pooled,
            "pairwise_mi": pairwise_mi,
        }
        pw_str = ", ".join(f"送り手{k}: {v['mi']:.4f}bit" for k, v in pairwise_mi.items())
        print(f"[身元付きN={N_AGENTS}体 seed={traj_seed}] {n_ep}ep: プール全体MI={mi_pooled:.4f}bit, "
              f"送信率={rate_pooled:.4f} | 送り手別MI: {pw_str}")

    result = {
        "traj_seed": traj_seed, "n_agents": N_AGENTS,
        "avg_dev_history": avg_dev_hist, "collision_rate_history": coll_hist, "guess_acc_history": guess_acc_hist,
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(f"multiagent_identity_train_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved multiagent_identity_train_seed{traj_seed}.json")


def aggregate():
    data = [json.load(open(f"multiagent_identity_train_seed{s}.json")) for s in TRAJ_SEEDS]

    print(f"=== (0) 土台: 衝突回避タスク自体の改善(身元付きN={N_AGENTS}体、n=3系統の平均±標準偏差) ===")
    coll_early = [np.mean(d["collision_rate_history"][:500]) for d in data]
    coll_late = [np.mean(d["collision_rate_history"][-500:]) for d in data]
    print(f"衝突率: 序盤(最初500ep)={np.mean(coll_early):.4f}±{np.std(coll_early):.4f}, "
          f"終盤(最後500ep)={np.mean(coll_late):.4f}±{np.std(coll_late):.4f}")

    print(f"\n=== (1) プール全体MI(チェックポイント別、身元付きN={N_AGENTS}体、n=3の平均±標準偏差) ===")
    mi_pooled_summary = {}
    for n_ep in CHECKPOINT_EPISODES:
        key = str(n_ep)
        mis = [d["mi_by_checkpoint"][key]["mi_pooled"] for d in data]
        rates = [d["mi_by_checkpoint"][key]["signal_rate_pooled"] for d in data]
        mi_pooled_summary[n_ep] = {
            "mi_mean": float(np.mean(mis)), "mi_std": float(np.std(mis)),
            "rate_mean": float(np.mean(rates)), "rate_std": float(np.std(rates)),
        }
        print(f"{n_ep}ep: プール全体MI={np.mean(mis):.4f}±{np.std(mis):.4f}bit, 送信率={np.mean(rates):.4f}±{np.std(rates):.4f}")

    print(f"\n=== (2) 送り手別ペア単位MI(チェックポイント別、全系統・全送り手の平均±標準偏差) ===")
    pairwise_summary = {}
    for n_ep in CHECKPOINT_EPISODES:
        key = str(n_ep)
        all_pair_mis = []
        for d in data:
            for sender_id, v in d["mi_by_checkpoint"][key]["pairwise_mi"].items():
                all_pair_mis.append(v["mi"])
        pairwise_summary[n_ep] = {"mi_mean": float(np.mean(all_pair_mis)), "mi_std": float(np.std(all_pair_mis)), "n": len(all_pair_mis)}
        print(f"{n_ep}ep: ペア単位MI(送り手別、n={len(all_pair_mis)}件の平均)={np.mean(all_pair_mis):.4f}±{np.std(all_pair_mis):.4f}bit")

    # 2体(v2)・身元非依存3体との比較
    v2_data, ma_data = None, None
    try:
        v2_data = [json.load(open(f"community_v2_train_seed{s}.json")) for s in TRAJ_SEEDS]
    except FileNotFoundError:
        pass
    try:
        ma_data = [json.load(open(f"multiagent_train_seed{s}.json")) for s in TRAJ_SEEDS]
    except FileNotFoundError:
        pass

    summary = {
        "n_agents": N_AGENTS,
        "collision_early_mean": float(np.mean(coll_early)), "collision_early_std": float(np.std(coll_early)),
        "collision_late_mean": float(np.mean(coll_late)), "collision_late_std": float(np.std(coll_late)),
        "mi_pooled_by_checkpoint": {str(k): v for k, v in mi_pooled_summary.items()},
        "pairwise_mi_by_checkpoint": {str(k): v for k, v in pairwise_summary.items()},
    }
    with open("multiagent_identity_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved multiagent_identity_summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ns = CHECKPOINT_EPISODES

    pw_means = [pairwise_summary[n]["mi_mean"] for n in ns]
    pw_stds = [pairwise_summary[n]["mi_std"] for n in ns]
    axes[0].errorbar(ns, pw_means, yerr=pw_stds, marker="D", color="#2E7D32", label="N=3体・身元付き(ペア単位MI)")

    pooled_means = [mi_pooled_summary[n]["mi_mean"] for n in ns]
    pooled_stds = [mi_pooled_summary[n]["mi_std"] for n in ns]
    axes[0].errorbar(ns, pooled_means, yerr=pooled_stds, marker="^", color="#7030A0", linestyle="--",
                      label="N=3体・身元付き(プール全体MI)")

    if ma_data is not None:
        ma_means = [np.mean([d["mi_by_checkpoint"][str(cp)]["mi"] for d in ma_data]) for cp in ns]
        ma_stds = [np.std([d["mi_by_checkpoint"][str(cp)]["mi"] for d in ma_data]) for cp in ns]
        axes[0].errorbar(ns, ma_means, yerr=ma_stds, marker="x", color="#808080",
                          label="N=3体・身元非依存(前回)")

    if v2_data is not None:
        v2_cps = m.CHECKPOINT_EPISODES
        v2_means = [np.mean([d["mi_by_checkpoint"][str(cp)]["mi"] for d in v2_data]) for cp in v2_cps]
        v2_stds = [np.std([d["mi_by_checkpoint"][str(cp)]["mi"] for d in v2_data]) for cp in v2_cps]
        axes[0].errorbar(v2_cps, v2_means, yerr=v2_stds, marker="s", color="#C0504D", label="N=2体(v2)")

    axes[0].set_xlabel("学習量(episode数)")
    axes[0].set_ylabel("I(signal;dominant_dev)[bit]")
    axes[0].set_title("送り手身元付き推測表によるMIの回復検証")
    axes[0].legend(fontsize=7)

    x = np.arange(2)
    axes[1].bar(x, [np.mean(coll_early), np.mean(coll_late)],
                yerr=[np.std(coll_early), np.std(coll_late)], color=["#BFBFBF", "#4472C4"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["序盤(最初500ep)", "終盤(最後500ep)"])
    axes[1].set_ylabel("衝突率")
    axes[1].set_title(f"衝突率: 序盤 vs 終盤(身元付きN={N_AGENTS}体)")

    fig.suptitle("要件6: 送り手身元付き推測表によるMI回復の検証(3体)")
    fig.tight_layout()
    fig.savefig("community_signal_multiagent_identity_comparison.png", dpi=150)
    print("グラフを community_signal_multiagent_identity_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train_chunk":
        train_chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "train_finalize":
        train_finalize(int(sys.argv[2]))
    elif cmd == "aggregate":
        aggregate()
