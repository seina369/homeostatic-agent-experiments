"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 集団規模(3体)での信号創発の検証
==========================================================

要件6の逸脱エージェント実験で、信号を受け取る側(解釈規則)には規範性を示す
証拠が得られたが、信号を送る側(符号化規則)の対応関係はもともと弱く
(MI≈0.02bit程度)、規範性テストの効果もはっきりしなかった。言語進化の
研究では、2体だけのペアより3体以上の集団の方が、複数の相手全員と意思疎通
する必要があるという圧力から、より体系的で安定した規約に収束しやすいことが
知られている。本プロトタイプは、community_signal_v2_prototype.pyの設定
(4×4グリッド・衝突ペナルティ8.0・推測ゲームによる直接報酬)はそのままに、
エージェント数を2体から3体に増やして同じ学習を行い、信号と内部状態のMI
(特に送信側の対応関係の強さ)が2体の場合より強く・安定するかを確認する。

**N体への一般化**: 各エージェントiは、他の全エージェントの信号・相対方向を
同時に観測するのではなく(組み合わせ爆発を避けるため)、各時点で最も近い
他エージェント(マンハッタン距離最小、以下「隣人」)の直前信号・相対方向のみを
観測する。推測ゲームも同様に、各エージェントは自分の隣人の支配的逸脱クラスを
隣人の信号から推測する(隣人は移動のたびに動的に変わりうる)。これにより、
各エージェントは学習を通じて「(誰が隣人であっても通用する)一貫した符号化・
解釈規則」を採用する圧力にさらされる。これは2体ペアにはない、複数の相手
全員との相互理解可能性という言語進化文献の核心的な圧力に対応する。

衝突は、あるエージェントが他の**いずれか**のエージェントと同じタイルを
占有した場合にペナルティを受ける形で一般化した(4×4グリッドに3体がいる
ため、2体の場合より衝突リスク自体が自然に高まる)。

各エージェントは独立したQ学習エージェント+独立したGuessAgentを持つ
(パラメータ共有はしない。共有すれば収束が自明になってしまうため、2体
実験と同様に、独立学習者が相互作用を通じて収束するかを見る)。

学習量はcommunity_signal_v2_prototype.pyと同じ3500ep(直接比較のため)。
学習系列の乱数(traj_seed=0,11,22)を変えた3系統で確認する。45秒のbash呼び出し
制限に対応するため、1000epずつ4チャンクに分割する(合計4000epまで確保し、
3500epで打ち切る)。

使い方:
  python3 community_signal_multiagent_prototype.py train_chunk <traj_seed> <end_ep>
  python3 community_signal_multiagent_prototype.py train_finalize <traj_seed>
  python3 community_signal_multiagent_prototype.py aggregate
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
import community_signal_v2_prototype as m  # GRID_SIZE=4, ACTIONS_COMM, GuessAgent, mutual_info_signal_vs_class等を再利用

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
COLLISION_PENALTY = m.COLLISION_PENALTY  # 8.0、v2と同一
GUESS_BONUS = m.GUESS_BONUS
GUESS_EPS = m.GUESS_EPS
N_ROLLOUT_EPISODES = 100
ROLLOUT_EPS = 0.1


class MultiAgentHomeostasisEnvN:
    """community_signal_v2のMultiAgentHomeostasisEnvをN体に一般化した版。
    各エージェントは「最も近い他エージェント(隣人)」の信号・相対方向のみを
    観測する(組み合わせ爆発を避けるため)。"""

    def __init__(self, rng, n_agents=N_AGENTS, collision_penalty=COLLISION_PENALTY):
        self.n = n_agents
        self.rng = rng
        self.food_tiles = hp.random_tiles(3, rng)
        self.shelter_tiles = hp.random_tiles(3, rng)
        self.hazard_tiles = hp.random_tiles(4, rng)
        self.collision_penalty = collision_penalty
        self.reset()

    def reset(self):
        # N体をグリッド上に分散配置(初期位置が重ならないようにする)
        gs = hp.GRID_SIZE
        candidates = [(x, y) for x in range(gs) for y in range(gs)]
        self.rng.shuffle(candidates)
        self.pos = list(candidates[:self.n])
        self.energy = [100.0] * self.n
        self.temperature = [hp.OPTIMAL_TEMP] * self.n
        self.damage = [0.0] * self.n
        self.last_signal = [0] * self.n
        self.t = 0
        return [self.observe(i) for i in range(self.n)]

    def _nearest_dir(self, pos, tiles):
        if not tiles:
            return (0, 0)
        x, y = pos
        best = min(tiles, key=lambda t: abs(t[0] - x) + abs(t[1] - y))
        return (int(np.sign(best[0] - x)), int(np.sign(best[1] - y)))

    def _rel_dir(self, pos_from, pos_to):
        return (int(np.sign(pos_to[0] - pos_from[0])), int(np.sign(pos_to[1] - pos_from[1])))

    def nearest_agent(self, i):
        best_j, best_d = None, None
        for j in range(self.n):
            if j == i:
                continue
            d = abs(self.pos[i][0] - self.pos[j][0]) + abs(self.pos[i][1] - self.pos[j][1])
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        return best_j

    def observe(self, i):
        j = self.nearest_agent(i)
        e_bin = int(np.clip(self.energy[i] // 20, 0, 5))
        t_bin = int(np.clip((self.temperature[i] - hp.OPTIMAL_TEMP + 15) // 6, 0, 5))
        d_bin = int(np.clip(self.damage[i] // 20, 0, 5))
        food_dir = self._nearest_dir(self.pos[i], self.food_tiles)
        shelter_dir = self._nearest_dir(self.pos[i], self.shelter_tiles)
        hazard_dir = self._nearest_dir(self.pos[i], self.hazard_tiles)
        neighbor_dir = self._rel_dir(self.pos[i], self.pos[j])
        return (food_dir, shelter_dir, hazard_dir, e_bin, t_bin, d_bin, self.last_signal[j], neighbor_dir)

    def dominant_deviation(self, i):
        dev_energy = abs(self.energy[i] - hp.OPTIMAL_ENERGY) / 100.0
        dev_temp = abs(self.temperature[i] - hp.OPTIMAL_TEMP) / 30.0
        dev_damage = abs(self.damage[i] - hp.OPTIMAL_DAMAGE) / 100.0
        return int(np.argmax([dev_energy, dev_temp, dev_damage]))

    def step(self, actions):
        new_pos = list(self.pos)
        new_signal = [0] * self.n
        for i, a in enumerate(actions):
            if a == "signal":
                new_signal[i] = 1
            else:
                dx, dy = m.MOVES[a]
                x, y = self.pos[i]
                new_pos[i] = (int(np.clip(x + dx, 0, hp.GRID_SIZE - 1)), int(np.clip(y + dy, 0, hp.GRID_SIZE - 1)))
        self.pos = new_pos
        self.last_signal = new_signal

        rewards = [0.0] * self.n
        deviations = [0.0] * self.n
        occupied = {}
        for i in range(self.n):
            occupied.setdefault(self.pos[i], []).append(i)

        for i in range(self.n):
            self.energy[i] -= hp.ENERGY_DECAY_PER_STEP
            self.temperature[i] += np.random.randn() * hp.TEMP_DRIFT_STD
            self.damage[i] = max(0.0, self.damage[i] - hp.DAMAGE_HEAL_PER_STEP)
            if self.pos[i] in self.food_tiles:
                self.energy[i] += 40.0
            if self.pos[i] in self.shelter_tiles:
                self.temperature[i] += (hp.OPTIMAL_TEMP - self.temperature[i]) * 0.5
            if self.pos[i] in self.hazard_tiles:
                self.damage[i] += 30.0
            self.energy[i] = float(np.clip(self.energy[i], 0.0, 100.0))
            self.damage[i] = float(np.clip(self.damage[i], 0.0, 100.0))

            dev_energy = abs(self.energy[i] - hp.OPTIMAL_ENERGY) / 100.0
            dev_temp = abs(self.temperature[i] - hp.OPTIMAL_TEMP) / 30.0
            dev_damage = abs(self.damage[i] - hp.OPTIMAL_DAMAGE) / 100.0
            deviations[i] = dev_energy + dev_temp + dev_damage
            rewards[i] = -deviations[i]

        collided_flags = [False] * self.n
        n_collision_events = 0
        for tile, occs in occupied.items():
            if len(occs) > 1:
                n_collision_events += 1
                for i in occs:
                    collided_flags[i] = True
                    rewards[i] -= self.collision_penalty

        self.t += 1
        done = self.t >= hp.MAX_STEPS or any(e <= 0.0 for e in self.energy)
        any_collision = any(collided_flags)
        return [self.observe(i) for i in range(self.n)], rewards, done, deviations, any_collision, collided_flags


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
            sig_for_guess = obs[i][6]  # 隣人の直前signal
            gv = guesses[i].act(sig_for_guess, guess_eps)
            guess_vals.append(gv)
            corrects.append(int(gv == doms[neighbors[i]]))
        correct_counts = [c + corr for c, corr in zip(correct_counts, corrects)]
        n_guesses += 1

        next_obs, base_rewards, done, deviations, any_collision, collided_flags = env.step(actions)

        # ボーナス: 各エージェントiは「自分が隣人を当てた」ボーナス+「隣人からiが当てられた」ボーナスの和を受け取る
        bonus = [0.0] * env.n
        for i in range(env.n):
            bonus[i] += GUESS_BONUS * corrects[i]  # iが隣人を当てた
        # iが「他の誰か(kがiを隣人としていた場合)」に当てられたボーナスも加える
        for k in range(env.n):
            j = neighbors[k]
            bonus[j] += GUESS_BONUS * corrects[k]

        total_rewards = [base_rewards[i] + bonus[i] for i in range(env.n)]

        if learn:
            for i in range(env.n):
                agents[i].update(obs[i], actions[i], total_rewards[i], next_obs[i], done)
        if learn_guess:
            for i in range(env.n):
                guesses[i].update(obs[i][6], guess_vals[i], GUESS_BONUS * corrects[i])

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
    state_file = f"multiagent_state_seed{traj_seed}.pkl"
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
        print(f"[N={N_AGENTS}体 seed={traj_seed}] 再開: {start_ep}ep目から{end_ep}epまで学習")
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        env = MultiAgentHomeostasisEnvN(random.Random(m.TRAIN_SEED), n_agents=N_AGENTS)
        agents = [QLearningAgent() for _ in range(N_AGENTS)]
        guesses = [m.GuessAgent() for _ in range(N_AGENTS)]
        avg_dev_hist, coll_hist, guess_acc_hist = [], [], []
        all_checkpoints = {}
        start_ep = 0
        print(f"[N={N_AGENTS}体 seed={traj_seed}] 新規開始: 0ep目から{end_ep}epまで学習")

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
    print(f"[N={N_AGENTS}体 seed={traj_seed}] {end_ep}epまで完了・保存 "
          f"(直近100ep衝突率={np.mean(coll_hist[-100:]):.4f}, 推測精度={np.mean(guess_acc_hist[-100:]):.4f})")


def rollout_for_signal_analysis(env, agent_qs, guess_qs, n_episodes, eps):
    agents = []
    guesses = []
    for q in agent_qs:
        a = QLearningAgent(); a.q = dict(q); agents.append(a)
    for q in guess_qs:
        g = m.GuessAgent(); g.q = dict(q); guesses.append(g)
    records = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            doms = [env.dominant_deviation(i) for i in range(env.n)]
            actions = [m.act(agents[i], obs[i], eps) for i in range(env.n)]
            for i in range(env.n):
                records.append((doms[i], 1 if actions[i] == "signal" else 0))
            next_obs, rewards, done, deviations, any_collision, collided_flags = env.step(actions)
            obs = next_obs
    return records


def train_finalize(traj_seed):
    state_file = f"multiagent_state_seed{traj_seed}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    agents, guesses = state["agents"], state["guesses"]
    avg_dev_hist, coll_hist, guess_acc_hist = state["avg_dev_hist"], state["coll_hist"], state["guess_acc_hist"]
    checkpoints = state["checkpoints"]

    print(f"[N={N_AGENTS}体 seed={traj_seed}] === 土台: 衝突回避タスク自体の改善 ===")
    print(f"[N={N_AGENTS}体 seed={traj_seed}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[:500]):.4f}, 推測精度={np.mean(guess_acc_hist[:500]):.4f}")
    print(f"[N={N_AGENTS}体 seed={traj_seed}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[-500:]):.4f}, 推測精度={np.mean(guess_acc_hist[-500:]):.4f}")

    mi_by_checkpoint = {}
    for n_ep in CHECKPOINT_EPISODES:
        agent_qs, guess_qs = checkpoints[n_ep]
        random.seed(traj_seed * 7000 + n_ep)
        np.random.seed(traj_seed * 7000 + n_ep)
        rollout_env = MultiAgentHomeostasisEnvN(random.Random(m.TRAIN_SEED), n_agents=N_AGENTS)
        records = rollout_for_signal_analysis(rollout_env, agent_qs, guess_qs, N_ROLLOUT_EPISODES, ROLLOUT_EPS)
        mi, signal_rate, cond_dist, marg_dist = m.mutual_info_signal_vs_class(records)
        mi_by_checkpoint[str(n_ep)] = {
            "mi": mi, "signal_rate": signal_rate, "cond_dist_given_signal": cond_dist, "marginal_dist": marg_dist,
        }
        print(f"[N={N_AGENTS}体 seed={traj_seed}] {n_ep}ep: I(signal;dominant_dev)={mi:.4f}bit, "
              f"信号送信率={signal_rate:.4f}, signal時分布={cond_dist}, 全体分布={marg_dist}")

    result = {
        "traj_seed": traj_seed, "n_agents": N_AGENTS,
        "avg_dev_history": avg_dev_hist, "collision_rate_history": coll_hist, "guess_acc_history": guess_acc_hist,
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(f"multiagent_train_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved multiagent_train_seed{traj_seed}.json")


def aggregate():
    data = [json.load(open(f"multiagent_train_seed{s}.json")) for s in TRAJ_SEEDS]

    print(f"=== (0) 土台: 衝突回避タスク自体の改善(N={N_AGENTS}体、n=3系統の平均±標準偏差) ===")
    coll_early = [np.mean(d["collision_rate_history"][:500]) for d in data]
    coll_late = [np.mean(d["collision_rate_history"][-500:]) for d in data]
    print(f"衝突率: 序盤(最初500ep)={np.mean(coll_early):.4f}±{np.std(coll_early):.4f}, "
          f"終盤(最後500ep)={np.mean(coll_late):.4f}±{np.std(coll_late):.4f}")

    print(f"\n=== (1) 信号と内部状態のMI・信号送信率(チェックポイント別、N={N_AGENTS}体、n=3の平均±標準偏差) ===")
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

    # 2体(v2)との比較
    v2_data = None
    try:
        v2_data = [json.load(open(f"community_v2_train_seed{s}.json")) for s in TRAJ_SEEDS]
    except FileNotFoundError:
        pass

    summary = {
        "n_agents": N_AGENTS,
        "collision_early_mean": float(np.mean(coll_early)), "collision_early_std": float(np.std(coll_early)),
        "collision_late_mean": float(np.mean(coll_late)), "collision_late_std": float(np.std(coll_late)),
        "mi_by_checkpoint": {str(k): v for k, v in mi_summary.items()},
    }
    with open("multiagent_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved multiagent_summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    ns = CHECKPOINT_EPISODES
    mi_means = [mi_summary[n]["mi_mean"] for n in ns]
    mi_stds = [mi_summary[n]["mi_std"] for n in ns]
    axes[0].errorbar(ns, mi_means, yerr=mi_stds, marker="o", color="#4472C4", label=f"N={N_AGENTS}体(本実験)")
    if v2_data is not None:
        v2_cps = m.CHECKPOINT_EPISODES
        v2_mi_means = [np.mean([d["mi_by_checkpoint"][str(cp)]["mi"] for d in v2_data]) for cp in v2_cps]
        v2_mi_stds = [np.std([d["mi_by_checkpoint"][str(cp)]["mi"] for d in v2_data]) for cp in v2_cps]
        axes[0].errorbar(v2_cps, v2_mi_means, yerr=v2_mi_stds, marker="s", color="#C0504D", label="N=2体(v2)")
    axes[0].set_xlabel("学習量(episode数)")
    axes[0].set_ylabel("I(signal;dominant_dev)[bit]")
    axes[0].set_title("集団規模とMIの比較")
    axes[0].legend(fontsize=8)

    x = np.arange(2)
    axes[1].bar(x, [np.mean(coll_early), np.mean(coll_late)],
                yerr=[np.std(coll_early), np.std(coll_late)], color=["#BFBFBF", "#4472C4"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["序盤(最初500ep)", "終盤(最後500ep)"])
    axes[1].set_ylabel("衝突率")
    axes[1].set_title(f"衝突率: 序盤 vs 終盤(N={N_AGENTS}体)")

    fig.suptitle(f"要件6: 集団規模({N_AGENTS}体)での信号創発の検証")
    fig.tight_layout()
    fig.savefig("community_signal_multiagent_comparison.png", dpi=150)
    print("グラフを community_signal_multiagent_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train_chunk":
        train_chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "train_finalize":
        train_finalize(int(sys.argv[2]))
    elif cmd == "aggregate":
        aggregate()
