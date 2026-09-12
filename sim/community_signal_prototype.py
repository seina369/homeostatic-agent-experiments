"""
感情AIプロジェクト フェーズ6 プロトタイプ: 複数個体による共同体形成(要件6)
==========================================================

要件6は「単なる複製数ではなく、相互の状態帰属という実践を生み出す」ことを狙う、
これまで完全に手つかずだった要件。最小構成として、同じグリッドワールドに2体の
エージェントを同時に配置し、移動行動(up/down/left/right/stay)に加えて
「signal」という新しい行動を1つ追加した(MultiAgentHomeostasisEnv)。

信号自体に固定の意味は与えない。signalを選ぶとその歩は移動できない(移動を
犠牲にする、という自然なコスト)以外は特別扱いせず、報酬は信号そのものにではなく
タスク上の成果、具体的には「同じ食料タイルへの同時アクセスを避けられたか」
(collision_penalty)にのみ紐づけた。2体を独立Q学習(自分の行動のみ更新)で
同時に学習させ、各エージェントの状態には、自分のセンサー情報に加えて
「相手が直前のステップでsignalを送ったか」「相手の相対方向」を含める。

検証したい2点:

  (1) 信号と送り手の内部状態(3センサーのうちどれが支配的に逸脱しているか)との
      相互情報量I(signal;dominant_deviation)を、要件7のモニタ実験
      (monitor_action_diversity_prototype.py)と同じ考え方で計算し、学習の
      チェックポイント(300ep/1200ep/2500ep)でその推移を見る。安定した対応関係が
      学習を通じて生まれるかどうかを確認する。
  (2) 収束後のペアの一方(agent0)のQテーブルを固定し(=既存の文化の担い手)、
      新しいエージェント(agent_C、Qテーブルは空)をそれと組ませて追加学習させる。
      これが、ゼロから2体を組ませて学習させた場合(=このペア自身の学習初期、
      両者ともQテーブルが空だった時期)と比べて、衝突率がより速く下がるかどうかを
      比較する。既存の信号の意味を新参者がより速く学習できれば、そのペアに固有の
      偶然の癖ではなく、伝達可能な「共有された実践」だと言える。

学習系列の乱数(traj_seed=0,11,22)を変えた3系統で確認する。処理を2回のbash呼び出し
に分割する: train(ペア本体の学習+チェックポイント保存)→newcomer(新参者の追加学習+
信号分析)。

使い方:
  python3 community_signal_prototype.py train <traj_seed>
  python3 community_signal_prototype.py newcomer <traj_seed>
  python3 community_signal_prototype.py aggregate
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

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

ACTIONS_COMM = ["up", "down", "left", "right", "stay", "signal"]
hp.ACTIONS[:] = ACTIONS_COMM  # QLearningAgentは homeostasis_prototype.ACTIONS を参照するため差し替える
hp.GRID_SIZE = 5  # 8x8のままだと2体が同じ資源タイルに同時に居合わせる頻度が低すぎ、
                  # 衝突がほぼ発生せず(実測0.0001)学習信号として機能しなかったため、
                  # グリッドを縮小して接触頻度を上げる(最小構成の共同体テストとして妥当な簡略化)

MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0), "stay": (0, 0), "signal": (0, 0)}

TRAIN_SEED = 0
TRAJ_SEEDS = [0, 11, 22]
N_EPISODES = 2500
DECAY_EPISODES = 1500
CHECKPOINT_EPISODES = [300, 1200, 2500]
COLLISION_PENALTY = 2.0
ROLLOUT_EPS = 0.1
N_ROLLOUT_EPISODES = 100
N_NEWCOMER_EPISODES = 800
NEWCOMER_PARTNER_EPS = 0.05  # 固定された既存個体(agent0)の振る舞い(成熟後の標準的な探索率)


class MultiAgentHomeostasisEnv:
    """2体のエージェントを同じマップに置く、signal行動付きの拡張版グリッドワールド。
    信号自体には固定の意味を与えず、報酬は自分のセンサー恒常性維持と、相手との
    食料タイル同時アクセス(衝突)の回避にのみ紐づける。"""

    def __init__(self, rng, collision_penalty=COLLISION_PENALTY):
        self.rng = rng
        self.food_tiles = hp.random_tiles(3, rng)
        self.shelter_tiles = hp.random_tiles(3, rng)
        self.hazard_tiles = hp.random_tiles(4, rng)
        self.collision_penalty = collision_penalty
        self.reset()

    def reset(self):
        c = hp.GRID_SIZE // 2
        self.pos = [(c - 1, c), (c + 1, c)]
        self.energy = [100.0, 100.0]
        self.temperature = [hp.OPTIMAL_TEMP, hp.OPTIMAL_TEMP]
        self.damage = [0.0, 0.0]
        self.last_signal = [0, 0]
        self.t = 0
        return [self.observe(0), self.observe(1)]

    def _nearest_dir(self, pos, tiles):
        if not tiles:
            return (0, 0)
        x, y = pos
        best = min(tiles, key=lambda t: abs(t[0] - x) + abs(t[1] - y))
        return (int(np.sign(best[0] - x)), int(np.sign(best[1] - y)))

    def _rel_dir(self, pos_from, pos_to):
        return (int(np.sign(pos_to[0] - pos_from[0])), int(np.sign(pos_to[1] - pos_from[1])))

    def observe(self, i):
        j = 1 - i
        e_bin = int(np.clip(self.energy[i] // 20, 0, 5))
        t_bin = int(np.clip((self.temperature[i] - hp.OPTIMAL_TEMP + 15) // 6, 0, 5))
        d_bin = int(np.clip(self.damage[i] // 20, 0, 5))
        food_dir = self._nearest_dir(self.pos[i], self.food_tiles)
        shelter_dir = self._nearest_dir(self.pos[i], self.shelter_tiles)
        hazard_dir = self._nearest_dir(self.pos[i], self.hazard_tiles)
        partner_dir = self._rel_dir(self.pos[i], self.pos[j])
        return (food_dir, shelter_dir, hazard_dir, e_bin, t_bin, d_bin, self.last_signal[j], partner_dir)

    def dominant_deviation(self, i):
        dev_energy = abs(self.energy[i] - hp.OPTIMAL_ENERGY) / 100.0
        dev_temp = abs(self.temperature[i] - hp.OPTIMAL_TEMP) / 30.0
        dev_damage = abs(self.damage[i] - hp.OPTIMAL_DAMAGE) / 100.0
        return int(np.argmax([dev_energy, dev_temp, dev_damage]))

    def step(self, actions):
        new_pos = list(self.pos)
        new_signal = [0, 0]
        for i, a in enumerate(actions):
            if a == "signal":
                new_signal[i] = 1
            else:
                dx, dy = MOVES[a]
                x, y = self.pos[i]
                new_pos[i] = (int(np.clip(x + dx, 0, hp.GRID_SIZE - 1)), int(np.clip(y + dy, 0, hp.GRID_SIZE - 1)))
        self.pos = new_pos
        self.last_signal = new_signal

        rewards = [0.0, 0.0]
        deviations = [0.0, 0.0]
        for i in range(2):
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

        collided = self.pos[0] == self.pos[1]  # 同じタイルへの重複アクセス全般を「衝突」として扱う
                                                # (資源タイルに限定すると発生頻度が低すぎたための簡略化)
        if collided:
            rewards[0] -= self.collision_penalty
            rewards[1] -= self.collision_penalty

        self.t += 1
        done = self.t >= hp.MAX_STEPS or self.energy[0] <= 0.0 or self.energy[1] <= 0.0
        return [self.observe(0), self.observe(1)], rewards, done, deviations, collided


def act(agent, obs_i, eps):
    if random.random() < eps:
        return random.choice(ACTIONS_COMM)
    return agent.best_action(obs_i)


def train_pair(env, agent0, agent1, n_episodes, decay_episodes, checkpoint_eps=None,
               agent1_frozen=False, agent1_fixed_eps=None):
    checkpoint_eps = checkpoint_eps or []
    checkpoint_set = set(checkpoint_eps)
    checkpoints = {}
    avg_dev_history, collision_rate_history = [], []
    for ep in range(n_episodes):
        obs = env.reset()
        eps = ib.epsilon_for_episode(ep, decay_episodes)
        eps1 = agent1_fixed_eps if agent1_frozen else eps
        done = False
        devs, collisions, steps = [], 0, 0
        while not done:
            a0 = act(agent0, obs[0], eps)
            a1 = act(agent1, obs[1], eps1)
            next_obs, rewards, done, deviations, collided = env.step([a0, a1])
            agent0.update(obs[0], a0, rewards[0], next_obs[0], done)
            if not agent1_frozen:
                agent1.update(obs[1], a1, rewards[1], next_obs[1], done)
            obs = next_obs
            devs.append((deviations[0] + deviations[1]) / 2.0)
            collisions += int(collided)
            steps += 1
        avg_dev_history.append(float(np.mean(devs)))
        collision_rate_history.append(collisions / steps)
        if (ep + 1) in checkpoint_set:
            checkpoints[ep + 1] = (dict(agent0.q), dict(agent1.q))
    return avg_dev_history, collision_rate_history, checkpoints


def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def rollout_for_signal_analysis(env, q0, q1, n_episodes, eps):
    agent0 = QLearningAgent(); agent0.q = q0
    agent1 = QLearningAgent(); agent1.q = q1
    records = []  # (dominant_class, signaled)
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a0 = act(agent0, obs[0], eps)
            a1 = act(agent1, obs[1], eps)
            dom0 = env.dominant_deviation(0)
            dom1 = env.dominant_deviation(1)
            records.append((dom0, 1 if a0 == "signal" else 0))
            records.append((dom1, 1 if a1 == "signal" else 0))
            next_obs, rewards, done, deviations, collided = env.step([a0, a1])
            obs = next_obs
    return records


def mutual_info_signal_vs_class(records, n_classes=3):
    n = len(records)
    joint = np.zeros((2, n_classes))
    for cls, sig in records:
        joint[sig, cls] += 1
    joint_p = joint / n
    p_sig = joint_p.sum(axis=1)
    p_cls = joint_p.sum(axis=0)
    mi = 0.0
    for s in range(2):
        for c in range(n_classes):
            if joint_p[s, c] > 0 and p_sig[s] > 0 and p_cls[c] > 0:
                mi += joint_p[s, c] * np.log2(joint_p[s, c] / (p_sig[s] * p_cls[c]))
    signal_rate = float(p_sig[1])
    cond_dist = (joint[1] / joint[1].sum()).tolist() if joint[1].sum() > 0 else [None, None, None]
    marg_dist = p_cls.tolist()
    return float(mi), signal_rate, cond_dist, marg_dist


def run_train(traj_seed):
    random.seed(traj_seed)
    np.random.seed(traj_seed)
    env = MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
    agent0 = QLearningAgent()
    agent1 = QLearningAgent()
    avg_dev_hist, coll_hist, checkpoints = train_pair(
        env, agent0, agent1, N_EPISODES, DECAY_EPISODES, checkpoint_eps=CHECKPOINT_EPISODES
    )
    print(f"[seed={traj_seed}] 終盤(最後100ep)平均逸脱={np.mean(avg_dev_hist[-100:]):.4f}, "
          f"終盤衝突率={np.mean(coll_hist[-100:]):.4f}")
    print(f"[seed={traj_seed}] 序盤(最初100ep)平均逸脱={np.mean(avg_dev_hist[:100]):.4f}, "
          f"序盤衝突率={np.mean(coll_hist[:100]):.4f}")

    # 各チェックポイントでロールアウトし、信号と内部状態のMIを計算
    mi_by_checkpoint = {}
    for n_ep in CHECKPOINT_EPISODES:
        q0, q1 = checkpoints[n_ep]
        random.seed(traj_seed * 7000 + n_ep)
        np.random.seed(traj_seed * 7000 + n_ep)
        rollout_env = MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
        records = rollout_for_signal_analysis(rollout_env, dict(q0), dict(q1), N_ROLLOUT_EPISODES, ROLLOUT_EPS)
        mi, signal_rate, cond_dist, marg_dist = mutual_info_signal_vs_class(records)
        mi_by_checkpoint[str(n_ep)] = {
            "mi": mi, "signal_rate": signal_rate, "cond_dist_given_signal": cond_dist, "marginal_dist": marg_dist,
        }
        print(f"[seed={traj_seed}] {n_ep}ep: I(signal;dominant_dev)={mi:.4f}bit, 信号送信率={signal_rate:.4f}, "
              f"signal時のdominant分布={cond_dist}, 全体分布={marg_dist}")

    # 保存: 最終Qテーブル(agent0を新参者実験の「既存文化」として使う)、履歴、MI結果
    with open(f"community_qtables_seed{traj_seed}.pkl", "wb") as f:
        pickle.dump({"agent0_q": dict(agent0.q), "agent1_q": dict(agent1.q)}, f)

    result = {
        "traj_seed": traj_seed,
        "avg_dev_history": avg_dev_hist,
        "collision_rate_history": coll_hist,
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(f"community_train_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved community_train_seed{traj_seed}.json, community_qtables_seed{traj_seed}.pkl")


def run_newcomer(traj_seed):
    with open(f"community_qtables_seed{traj_seed}.pkl", "rb") as f:
        qtables = pickle.load(f)
    frozen_q0 = qtables["agent0_q"]

    random.seed(traj_seed * 13 + 1)
    np.random.seed(traj_seed * 13 + 1)
    env = MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
    frozen_agent0 = QLearningAgent(); frozen_agent0.q = dict(frozen_q0)
    newcomer_c = QLearningAgent()

    avg_dev_hist, coll_hist, _ = train_pair(
        env, frozen_agent0, newcomer_c, N_NEWCOMER_EPISODES, DECAY_EPISODES,
        agent1_frozen=True, agent1_fixed_eps=NEWCOMER_PARTNER_EPS,
    )
    print(f"[seed={traj_seed}] 新参者C 序盤(最初100ep)衝突率={np.mean(coll_hist[:100]):.4f}, "
          f"序盤平均逸脱={np.mean(avg_dev_hist[:100]):.4f}")

    result = {
        "traj_seed": traj_seed,
        "newcomer_avg_dev_history": avg_dev_hist,
        "newcomer_collision_rate_history": coll_hist,
    }
    with open(f"community_newcomer_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved community_newcomer_seed{traj_seed}.json")


def aggregate():
    train_data, newcomer_data = [], []
    for seed in TRAJ_SEEDS:
        with open(f"community_train_seed{seed}.json") as f:
            train_data.append(json.load(f))
        with open(f"community_newcomer_seed{seed}.json") as f:
            newcomer_data.append(json.load(f))

    print("=== (1) 信号と内部状態のMI(チェックポイント別、n=3の平均±標準偏差) ===")
    mi_summary = {}
    for n_ep in CHECKPOINT_EPISODES:
        key = str(n_ep)
        mis = [d["mi_by_checkpoint"][key]["mi"] for d in train_data]
        rates = [d["mi_by_checkpoint"][key]["signal_rate"] for d in train_data]
        mi_summary[n_ep] = {"mi_mean": float(np.mean(mis)), "mi_std": float(np.std(mis)),
                             "rate_mean": float(np.mean(rates)), "rate_std": float(np.std(rates))}
        print(f"{n_ep}ep: MI={np.mean(mis):.4f}±{np.std(mis):.4f}bit, 信号送信率={np.mean(rates):.4f}±{np.std(rates):.4f}")

    print("\n=== (1b) 全チェックポイント平均でのsignal時のdominant分布(参考、系統ごとの生値) ===")
    for d in train_data:
        for n_ep in CHECKPOINT_EPISODES:
            cd = d["mi_by_checkpoint"][str(n_ep)]["cond_dist_given_signal"]
            md = d["mi_by_checkpoint"][str(n_ep)]["marginal_dist"]
            print(f"  seed={d['traj_seed']} {n_ep}ep: P(dominant|signal)={cd}, P(dominant)全体={md}")

    print("\n=== (2) 新参者Cの学習速度 vs ペア自身の学習初期(n=3の平均±標準偏差) ===")
    windows = [100, 300, 800]
    comparison = {}
    for w in windows:
        newcomer_rates = [np.mean(d["newcomer_collision_rate_history"][:w]) for d in newcomer_data]
        original_rates = [np.mean(d["collision_rate_history"][:w]) for d in train_data]
        comparison[w] = {
            "newcomer_mean": float(np.mean(newcomer_rates)), "newcomer_std": float(np.std(newcomer_rates)),
            "original_mean": float(np.mean(original_rates)), "original_std": float(np.std(original_rates)),
        }
        print(f"最初{w}ep平均衝突率: 新参者C(既存agent0と組)={np.mean(newcomer_rates):.4f}±{np.std(newcomer_rates):.4f}, "
              f"ペア自身の学習初期(ゼロから)={np.mean(original_rates):.4f}±{np.std(original_rates):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ns = CHECKPOINT_EPISODES
    mi_means = [mi_summary[n]["mi_mean"] for n in ns]
    mi_stds = [mi_summary[n]["mi_std"] for n in ns]
    axes[0].errorbar(ns, mi_means, yerr=mi_stds, marker="o", color="#4472C4")
    axes[0].set_xlabel("学習量(episode数)")
    axes[0].set_ylabel("I(signal; dominant_deviation) [bit]")
    axes[0].set_title("(1) 信号と内部状態のMIの推移")

    rate_means = [mi_summary[n]["rate_mean"] for n in ns]
    rate_stds = [mi_summary[n]["rate_std"] for n in ns]
    axes[1].errorbar(ns, rate_means, yerr=rate_stds, marker="s", color="#C0504D")
    axes[1].set_xlabel("学習量(episode数)")
    axes[1].set_ylabel("信号送信率")
    axes[1].set_title("(1) 信号送信頻度の推移")

    x = np.arange(len(windows))
    width = 0.35
    newcomer_means = [comparison[w]["newcomer_mean"] for w in windows]
    newcomer_stds = [comparison[w]["newcomer_std"] for w in windows]
    original_means = [comparison[w]["original_mean"] for w in windows]
    original_stds = [comparison[w]["original_std"] for w in windows]
    axes[2].bar(x - width / 2, newcomer_means, width, yerr=newcomer_stds, label="新参者C(既存agent0と組)", color="#4472C4")
    axes[2].bar(x + width / 2, original_means, width, yerr=original_stds, label="ペア自身の学習初期(ゼロから)", color="#BFBFBF")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([f"最初{w}ep" for w in windows])
    axes[2].set_ylabel("平均衝突率")
    axes[2].set_title("(2) 新参者の学習速度比較")
    axes[2].legend(fontsize=8)

    fig.suptitle("要件6: 複数個体による共同体形成(signal行動の検証)")
    fig.tight_layout()
    fig.savefig("community_signal_comparison.png", dpi=150)
    print("グラフを community_signal_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "aggregate":
        aggregate()
    elif cmd == "train":
        run_train(int(sys.argv[2]))
    elif cmd == "newcomer":
        run_newcomer(int(sys.argv[2]))
