"""
感情AIプロジェクト フェーズ6 プロトタイプ: 複数個体による共同体形成(要件6) 再挑戦
==========================================================

前回(community_signal_prototype.py)は、(a)衝突がまれで低リスクだったため協調への
圧力が弱すぎた、(b)信号の有用性が報酬に直接結びついておらず、送り手・受け手が
同時にゼロから学習する「鶏と卵」問題を解く手がかりが乏しかった、という2点により、
信号と内部状態の対応関係の創発を確認できなかった。今回はこの2点を直接強化する。

  (1) 協調圧力の強化: グリッドを5x5→4x4へさらに縮小し(衝突の遭遇頻度を上げる)、
      衝突ペナルティを2.0→8.0へ引き上げた。
  (2) 信号の有用性を報酬に直結: 新たに「推測(guess)」という副次的な判断を各
      エージェントに追加した。各エージェントは、相手が直前ステップで信号を
      送ったかどうか(0/1)だけを手がかりに、相手の支配的な逸脱センサー(3クラス)を
      推測する。推測が当たれば、推測した側(受け手)・信号を送った側(送り手)の
      双方にボーナス報酬(GUESS_BONUS)を与える。これにより「信号を送ることに
      意味がある(=相手が正しく推測できれば自分も得をする)」という直接の学習圧力を
      作った。推測の当てずっぽうの手がかりは信号1ビットのみに限定し(相手の
      センサー生値や位置は見せない)、推測精度の向上が信号の情報量に起因すると
      言えるようにした。推測自体は信号値と欲求クラスの対応を事前に固定しない
      (GuessAgentという独立した小さな文脈付きバンディットが、報酬を通じてどの
      信号値がどのクラスを意味するかを学習する)。

学習量もN_EPISODES=3500(前回2500)に増やした。まず土台となる衝突回避タスク自体が
改善するかを確認したうえで、信号とのMIを見る。

新参者テスト(要件6の(2))も、前回の交絡要因(既存個体が探索率0.05に固定されており、
単に動きが予測しやすいだけで有利だった可能性)を避けるため、3群比較に変更した。

  - Arm1(比較対象、前回と同じ): ペア自身の学習初期(両者ともQテーブル空、
    探索率はep数に応じて通常通り減衰)
  - Arm2(対照群): 新参者Cを、Qテーブルが空の「新人」パートナー(学習はしないが
    探索率スケジュールはCと完全に一致)と組ませる
  - Arm3(処置群): 新参者Cを、収束済みのQテーブルを持つ既存agent0(学習はしないが
    探索率スケジュールはCと完全に一致)と組ませる

Arm2とArm3は探索率スケジュールが完全に同一なため、両者の差は純粋に
「パートナーが学習済みの内容を持っているかどうか」に起因する。Arm3がArm2を
上回れば、既存個体の学習内容(信号の使い方を含む)が新参者に有利に働く
「伝達可能な実践」であることの、前回より厳密な証拠になる。

学習系列の乱数(traj_seed=0,11,22)を変えた3系統で確認する。処理を2回のbash呼び出し
に分割する: train→newcomer。

使い方:
  python3 community_signal_v2_prototype.py train <traj_seed>
  python3 community_signal_v2_prototype.py newcomer <traj_seed>
  python3 community_signal_v2_prototype.py aggregate
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
hp.ACTIONS[:] = ACTIONS_COMM
hp.GRID_SIZE = 4  # 前回の5x5よりさらに縮小し、遭遇頻度を上げる

MOVES = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0), "stay": (0, 0), "signal": (0, 0)}
GUESS_CLASSES = [0, 1, 2]
ALPHA = hp.ALPHA

TRAIN_SEED = 0
TRAJ_SEEDS = [0, 11, 22]
N_EPISODES = 3500
DECAY_EPISODES = 2500
CHECKPOINT_EPISODES = [300, 1500, 3500]
COLLISION_PENALTY = 8.0   # 前回2.0から引き上げ
GUESS_BONUS = 1.0
ROLLOUT_EPS = 0.1
N_ROLLOUT_EPISODES = 100
N_NEWCOMER_EPISODES = 800
GUESS_EPS = 0.2  # 推測側にも一定の探索を持たせ、全クラスの手がかりを学習できるようにする


class GuessAgent:
    """相手の直前signalの有無(0/1)だけを手がかりに、相手の支配的な逸脱センサー
    (3クラス)を当てる、状態遷移を持たない文脈付きバンディット。信号値と欲求
    クラスの対応は固定せず、報酬(推測が当たったか)を通じて学習する。"""

    def __init__(self):
        self.q = {}

    def q_value(self, sig, guess):
        return self.q.get((sig, guess), 0.0)

    def best_guess(self, sig):
        values = [self.q_value(sig, g) for g in GUESS_CLASSES]
        return GUESS_CLASSES[int(np.argmax(values))]

    def act(self, sig, eps):
        if random.random() < eps:
            return random.choice(GUESS_CLASSES)
        return self.best_guess(sig)

    def update(self, sig, guess, reward):
        current = self.q_value(sig, guess)
        self.q[(sig, guess)] = current + ALPHA * (reward - current)


class MultiAgentHomeostasisEnv:
    def __init__(self, rng, collision_penalty=COLLISION_PENALTY):
        self.rng = rng
        self.food_tiles = hp.random_tiles(3, rng)
        self.shelter_tiles = hp.random_tiles(3, rng)
        self.hazard_tiles = hp.random_tiles(4, rng)
        self.collision_penalty = collision_penalty
        self.reset()

    def reset(self):
        c = hp.GRID_SIZE // 2
        self.pos = [(max(0, c - 1), c), (min(hp.GRID_SIZE - 1, c), c)]
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

        collided = self.pos[0] == self.pos[1]
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


def run_episode(env, agent0, agent1, guess0, guess1, eps0, eps1,
                 learn0=True, learn1=True, learn_guess0=True, learn_guess1=True, guess_eps=GUESS_EPS):
    obs = env.reset()
    done = False
    devs, collisions, steps = [], 0, 0
    correct0_count, correct1_count, n_guesses = 0, 0, 0
    while not done:
        dom0 = env.dominant_deviation(0)
        dom1 = env.dominant_deviation(1)

        a0 = act(agent0, obs[0], eps0)
        a1 = act(agent1, obs[1], eps1)

        sig_for_guess0 = obs[0][6]  # agent0が見る「相手(agent1)の直前signal」
        sig_for_guess1 = obs[1][6]  # agent1が見る「相手(agent0)の直前signal」
        guess0_val = guess0.act(sig_for_guess0, guess_eps)
        guess1_val = guess1.act(sig_for_guess1, guess_eps)
        correct0 = int(guess0_val == dom1)  # agent0がagent1の状態を当てられたか
        correct1 = int(guess1_val == dom0)  # agent1がagent0の状態を当てられたか
        correct0_count += correct0
        correct1_count += correct1
        n_guesses += 1

        next_obs, base_rewards, done, deviations, collided = env.step([a0, a1])

        total_r0 = base_rewards[0] + GUESS_BONUS * correct0 + GUESS_BONUS * correct1
        total_r1 = base_rewards[1] + GUESS_BONUS * correct1 + GUESS_BONUS * correct0

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

    avg_dev = float(np.mean(devs))
    coll_rate = collisions / steps
    guess_acc = (correct0_count + correct1_count) / (2 * n_guesses)
    return avg_dev, coll_rate, guess_acc


def train_pair(env, agent0, agent1, guess0, guess1, n_episodes, decay_episodes, checkpoint_eps=None):
    checkpoint_eps = checkpoint_eps or []
    checkpoint_set = set(checkpoint_eps)
    checkpoints = {}
    avg_dev_hist, coll_hist, guess_acc_hist = [], [], []
    for ep in range(n_episodes):
        eps = ib.epsilon_for_episode(ep, decay_episodes)
        avg_dev, coll_rate, guess_acc = run_episode(env, agent0, agent1, guess0, guess1, eps, eps)
        avg_dev_hist.append(avg_dev)
        coll_hist.append(coll_rate)
        guess_acc_hist.append(guess_acc)
        if (ep + 1) in checkpoint_set:
            checkpoints[ep + 1] = (dict(agent0.q), dict(agent1.q), dict(guess0.q), dict(guess1.q))
    return avg_dev_hist, coll_hist, guess_acc_hist, checkpoints


def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def rollout_for_signal_analysis(env, q0, q1, gq0, gq1, n_episodes, eps):
    agent0 = QLearningAgent(); agent0.q = q0
    agent1 = QLearningAgent(); agent1.q = q1
    guess0 = GuessAgent(); guess0.q = gq0
    guess1 = GuessAgent(); guess1.q = gq1
    records = []       # (dominant_class, signaled) 送り手側のMI用
    guess_correct = []  # 推測精度用(受け手0・1両方をプール)
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            dom0 = env.dominant_deviation(0)
            dom1 = env.dominant_deviation(1)
            a0 = act(agent0, obs[0], eps)
            a1 = act(agent1, obs[1], eps)
            sig_for_guess0 = obs[0][6]
            sig_for_guess1 = obs[1][6]
            guess0_val = guess0.best_guess(sig_for_guess0)
            guess1_val = guess1.best_guess(sig_for_guess1)
            guess_correct.append(int(guess0_val == dom1))
            guess_correct.append(int(guess1_val == dom0))
            records.append((dom0, 1 if a0 == "signal" else 0))
            records.append((dom1, 1 if a1 == "signal" else 0))
            next_obs, rewards, done, deviations, collided = env.step([a0, a1])
            obs = next_obs
    return records, guess_correct


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


def train_pair_range(env, agent0, agent1, guess0, guess1, start_ep, end_ep, decay_episodes, checkpoint_eps=None):
    """45秒のbash呼び出し制限に収めるため、学習をエピソード範囲で分割実行できるようにした版。
    start_ep/end_epは通し番号(0起点)で、eps_for_episodeは常にこの通し番号を使うため、
    分割してもepsilonの減衰スケジュールは連続した1回の学習と同じになる。"""
    checkpoint_eps = checkpoint_eps or []
    checkpoint_set = set(checkpoint_eps)
    checkpoints = {}
    avg_dev_hist, coll_hist, guess_acc_hist = [], [], []
    for ep in range(start_ep, end_ep):
        eps = ib.epsilon_for_episode(ep, decay_episodes)
        avg_dev, coll_rate, guess_acc = run_episode(env, agent0, agent1, guess0, guess1, eps, eps)
        avg_dev_hist.append(avg_dev)
        coll_hist.append(coll_rate)
        guess_acc_hist.append(guess_acc)
        if (ep + 1) in checkpoint_set:
            checkpoints[ep + 1] = (dict(agent0.q), dict(agent1.q), dict(guess0.q), dict(guess1.q))
    return avg_dev_hist, coll_hist, guess_acc_hist, checkpoints


def run_train_chunk(traj_seed, end_ep):
    state_file = f"community_v2_state_seed{traj_seed}.pkl"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env = state["env"]
        agent0, agent1 = state["agent0"], state["agent1"]
        guess0, guess1 = state["guess0"], state["guess1"]
        avg_dev_hist, coll_hist, guess_acc_hist = state["avg_dev_hist"], state["coll_hist"], state["guess_acc_hist"]
        all_checkpoints = state["checkpoints"]
        start_ep = state["last_ep"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
        print(f"[seed={traj_seed}] 再開: {start_ep}ep目から{end_ep}epまで学習")
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        env = MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
        agent0, agent1 = QLearningAgent(), QLearningAgent()
        guess0, guess1 = GuessAgent(), GuessAgent()
        avg_dev_hist, coll_hist, guess_acc_hist = [], [], []
        all_checkpoints = {}
        start_ep = 0
        print(f"[seed={traj_seed}] 新規開始: 0ep目から{end_ep}epまで学習")

    dev_h, coll_h, gacc_h, checkpoints = train_pair_range(
        env, agent0, agent1, guess0, guess1, start_ep, end_ep, DECAY_EPISODES, checkpoint_eps=CHECKPOINT_EPISODES
    )
    avg_dev_hist.extend(dev_h); coll_hist.extend(coll_h); guess_acc_hist.extend(gacc_h)
    all_checkpoints.update(checkpoints)

    state = {
        "env": env, "agent0": agent0, "agent1": agent1, "guess0": guess0, "guess1": guess1,
        "avg_dev_hist": avg_dev_hist, "coll_hist": coll_hist, "guess_acc_hist": guess_acc_hist,
        "checkpoints": all_checkpoints, "last_ep": end_ep,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)
    print(f"[seed={traj_seed}] {end_ep}epまで完了・保存 (直近100ep衝突率={np.mean(coll_hist[-100:]):.4f}, "
          f"推測精度={np.mean(guess_acc_hist[-100:]):.4f})")


def run_train_finalize(traj_seed):
    state_file = f"community_v2_state_seed{traj_seed}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    agent0, agent1 = state["agent0"], state["agent1"]
    guess0, guess1 = state["guess0"], state["guess1"]
    avg_dev_hist, coll_hist, guess_acc_hist = state["avg_dev_hist"], state["coll_hist"], state["guess_acc_hist"]
    checkpoints = state["checkpoints"]

    print(f"[seed={traj_seed}] === 土台: 衝突回避タスク自体の改善 ===")
    print(f"[seed={traj_seed}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[:500]):.4f}, 推測精度={np.mean(guess_acc_hist[:500]):.4f}")
    print(f"[seed={traj_seed}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[-500:]):.4f}, 推測精度={np.mean(guess_acc_hist[-500:]):.4f}")

    mi_by_checkpoint = {}
    for n_ep in CHECKPOINT_EPISODES:
        q0, q1, gq0, gq1 = checkpoints[n_ep]
        random.seed(traj_seed * 7000 + n_ep)
        np.random.seed(traj_seed * 7000 + n_ep)
        rollout_env = MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
        records, guess_correct = rollout_for_signal_analysis(
            rollout_env, dict(q0), dict(q1), dict(gq0), dict(gq1), N_ROLLOUT_EPISODES, ROLLOUT_EPS
        )
        mi, signal_rate, cond_dist, marg_dist = mutual_info_signal_vs_class(records)
        guess_acc = float(np.mean(guess_correct))
        mi_by_checkpoint[str(n_ep)] = {
            "mi": mi, "signal_rate": signal_rate, "cond_dist_given_signal": cond_dist,
            "marginal_dist": marg_dist, "guess_accuracy": guess_acc,
        }
        print(f"[seed={traj_seed}] {n_ep}ep: I(signal;dominant_dev)={mi:.4f}bit, 信号送信率={signal_rate:.4f}, "
              f"推測精度={guess_acc:.4f}(チャンス=0.333), signal時分布={cond_dist}, 全体分布={marg_dist}")

    with open(f"community_v2_qtables_seed{traj_seed}.pkl", "wb") as f:
        pickle.dump({
            "agent0_q": dict(agent0.q), "agent1_q": dict(agent1.q),
            "guess0_q": dict(guess0.q), "guess1_q": dict(guess1.q),
        }, f)

    result = {
        "traj_seed": traj_seed,
        "avg_dev_history": avg_dev_hist,
        "collision_rate_history": coll_hist,
        "guess_acc_history": guess_acc_hist,
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(f"community_v2_train_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved community_v2_train_seed{traj_seed}.json")


def run_newcomer(traj_seed):
    with open(f"community_v2_qtables_seed{traj_seed}.pkl", "rb") as f:
        qt = pickle.load(f)

    # Arm2(対照群): 新参者C_control + 空のQテーブルの新人パートナー(学習なし、探索率は一致)
    random.seed(traj_seed * 13 + 1)
    np.random.seed(traj_seed * 13 + 1)
    env2 = MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
    fresh_partner = QLearningAgent()
    fresh_partner_guess = GuessAgent()
    c_control = QLearningAgent()
    c_control_guess = GuessAgent()
    avg_dev2, coll2, gacc2 = [], [], []
    for ep in range(N_NEWCOMER_EPISODES):
        eps = ib.epsilon_for_episode(ep, DECAY_EPISODES)
        avg_dev, coll_rate, guess_acc = run_episode(
            env2, fresh_partner, c_control, fresh_partner_guess, c_control_guess, eps, eps,
            learn0=False, learn1=True, learn_guess0=False, learn_guess1=True,
        )
        avg_dev2.append(avg_dev); coll2.append(coll_rate); gacc2.append(guess_acc)
    print(f"[seed={traj_seed}] Arm2(対照,新人パートナー) 最初100ep衝突率={np.mean(coll2[:100]):.4f}")

    # Arm3(処置群): 新参者C_treat + 収束済みagent0(学習なし、探索率は一致)
    random.seed(traj_seed * 13 + 2)
    np.random.seed(traj_seed * 13 + 2)
    env3 = MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
    established_partner = QLearningAgent(); established_partner.q = dict(qt["agent0_q"])
    established_partner_guess = GuessAgent(); established_partner_guess.q = dict(qt["guess0_q"])
    c_treat = QLearningAgent()
    c_treat_guess = GuessAgent()
    avg_dev3, coll3, gacc3 = [], [], []
    for ep in range(N_NEWCOMER_EPISODES):
        eps = ib.epsilon_for_episode(ep, DECAY_EPISODES)
        avg_dev, coll_rate, guess_acc = run_episode(
            env3, established_partner, c_treat, established_partner_guess, c_treat_guess, eps, eps,
            learn0=False, learn1=True, learn_guess0=False, learn_guess1=True,
        )
        avg_dev3.append(avg_dev); coll3.append(coll_rate); gacc3.append(guess_acc)
    print(f"[seed={traj_seed}] Arm3(処置,既存agent0) 最初100ep衝突率={np.mean(coll3[:100]):.4f}")

    result = {
        "traj_seed": traj_seed,
        "arm2_control_collision_rate_history": coll2, "arm2_control_avg_dev_history": avg_dev2,
        "arm2_control_guess_acc_history": gacc2,
        "arm3_treatment_collision_rate_history": coll3, "arm3_treatment_avg_dev_history": avg_dev3,
        "arm3_treatment_guess_acc_history": gacc3,
    }
    with open(f"community_v2_newcomer_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved community_v2_newcomer_seed{traj_seed}.json")


def aggregate():
    train_data, newcomer_data = [], []
    for seed in TRAJ_SEEDS:
        with open(f"community_v2_train_seed{seed}.json") as f:
            train_data.append(json.load(f))
        with open(f"community_v2_newcomer_seed{seed}.json") as f:
            newcomer_data.append(json.load(f))

    print("=== (0) 土台: 衝突回避タスク自体の改善(n=3の平均±標準偏差) ===")
    coll_early = [np.mean(d["collision_rate_history"][:500]) for d in train_data]
    coll_late = [np.mean(d["collision_rate_history"][-500:]) for d in train_data]
    dev_early = [np.mean(d["avg_dev_history"][:500]) for d in train_data]
    dev_late = [np.mean(d["avg_dev_history"][-500:]) for d in train_data]
    gacc_early = [np.mean(d["guess_acc_history"][:500]) for d in train_data]
    gacc_late = [np.mean(d["guess_acc_history"][-500:]) for d in train_data]
    print(f"衝突率: 序盤(最初500ep)={np.mean(coll_early):.4f}±{np.std(coll_early):.4f}, "
          f"終盤(最後500ep)={np.mean(coll_late):.4f}±{np.std(coll_late):.4f}")
    print(f"平均逸脱: 序盤={np.mean(dev_early):.4f}±{np.std(dev_early):.4f}, "
          f"終盤={np.mean(dev_late):.4f}±{np.std(dev_late):.4f}")
    print(f"学習中の推測精度: 序盤={np.mean(gacc_early):.4f}±{np.std(gacc_early):.4f}, "
          f"終盤={np.mean(gacc_late):.4f}±{np.std(gacc_late):.4f}(チャンス=0.333)")

    print("\n=== (1) 信号と内部状態のMI・推測精度(チェックポイント別、n=3の平均±標準偏差) ===")
    mi_summary = {}
    for n_ep in CHECKPOINT_EPISODES:
        key = str(n_ep)
        mis = [d["mi_by_checkpoint"][key]["mi"] for d in train_data]
        rates = [d["mi_by_checkpoint"][key]["signal_rate"] for d in train_data]
        gaccs = [d["mi_by_checkpoint"][key]["guess_accuracy"] for d in train_data]
        mi_summary[n_ep] = {
            "mi_mean": float(np.mean(mis)), "mi_std": float(np.std(mis)),
            "rate_mean": float(np.mean(rates)), "rate_std": float(np.std(rates)),
            "gacc_mean": float(np.mean(gaccs)), "gacc_std": float(np.std(gaccs)),
        }
        print(f"{n_ep}ep: MI={np.mean(mis):.4f}±{np.std(mis):.4f}bit, 信号送信率={np.mean(rates):.4f}±{np.std(rates):.4f}, "
              f"推測精度(ロールアウト)={np.mean(gaccs):.4f}±{np.std(gaccs):.4f}")

    print("\n=== (2) 新参者3群比較(n=3の平均±標準偏差、平均衝突率) ===")
    windows = [100, 300, 800]
    comparison = {}
    for w in windows:
        arm1 = [np.mean(d["collision_rate_history"][:w]) for d in train_data]
        arm2 = [np.mean(d["arm2_control_collision_rate_history"][:w]) for d in newcomer_data]
        arm3 = [np.mean(d["arm3_treatment_collision_rate_history"][:w]) for d in newcomer_data]
        comparison[w] = {
            "arm1_mean": float(np.mean(arm1)), "arm1_std": float(np.std(arm1)),
            "arm2_mean": float(np.mean(arm2)), "arm2_std": float(np.std(arm2)),
            "arm3_mean": float(np.mean(arm3)), "arm3_std": float(np.std(arm3)),
        }
        print(f"最初{w}ep平均衝突率: Arm1(ゼロからペア)={np.mean(arm1):.4f}±{np.std(arm1):.4f}, "
              f"Arm2(新人パートナー対照)={np.mean(arm2):.4f}±{np.std(arm2):.4f}, "
              f"Arm3(既存agent0処置)={np.mean(arm3):.4f}±{np.std(arm3):.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ep_idx = np.arange(len(train_data[0]["collision_rate_history"]))
    window = 100
    for d in train_data:
        arr = np.array(d["collision_rate_history"])
        smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
        axes[0, 0].plot(smoothed, alpha=0.5, color="#4472C4")
    axes[0, 0].set_xlabel("エピソード")
    axes[0, 0].set_ylabel("衝突率(移動平均100ep)")
    axes[0, 0].set_title("(0) 衝突率の学習推移(3系統)")

    ns = CHECKPOINT_EPISODES
    mi_means = [mi_summary[n]["mi_mean"] for n in ns]
    mi_stds = [mi_summary[n]["mi_std"] for n in ns]
    gacc_means = [mi_summary[n]["gacc_mean"] for n in ns]
    gacc_stds = [mi_summary[n]["gacc_std"] for n in ns]
    ax2 = axes[0, 1].twinx()
    axes[0, 1].errorbar(ns, mi_means, yerr=mi_stds, marker="o", color="#4472C4", label="MI(左軸)")
    ax2.errorbar(ns, gacc_means, yerr=gacc_stds, marker="s", color="#C0504D", label="推測精度(右軸)")
    ax2.axhline(1 / 3, color="gray", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("学習量(episode数)")
    axes[0, 1].set_ylabel("I(signal;dominant_dev)[bit]", color="#4472C4")
    ax2.set_ylabel("推測精度", color="#C0504D")
    axes[0, 1].set_title("(1) 信号のMIと推測精度の推移")
    lines1, labs1 = axes[0, 1].get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    axes[0, 1].legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    x = np.arange(2)
    width = 0.35
    axes[1, 0].bar(x - width / 2, [np.mean(coll_early), np.mean(coll_late)],
                    width, yerr=[np.std(coll_early), np.std(coll_late)], color=["#BFBFBF", "#4472C4"])
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(["序盤(最初500ep)", "終盤(最後500ep)"])
    axes[1, 0].set_ylabel("衝突率")
    axes[1, 0].set_title("(0) 衝突率: 序盤 vs 終盤")

    xw = np.arange(len(windows))
    width = 0.25
    arm1_means = [comparison[w]["arm1_mean"] for w in windows]
    arm1_stds = [comparison[w]["arm1_std"] for w in windows]
    arm2_means = [comparison[w]["arm2_mean"] for w in windows]
    arm2_stds = [comparison[w]["arm2_std"] for w in windows]
    arm3_means = [comparison[w]["arm3_mean"] for w in windows]
    arm3_stds = [comparison[w]["arm3_std"] for w in windows]
    axes[1, 1].bar(xw - width, arm1_means, width, yerr=arm1_stds, label="Arm1:ゼロからペア", color="#BFBFBF")
    axes[1, 1].bar(xw, arm2_means, width, yerr=arm2_stds, label="Arm2:新人パートナー対照", color="#9BBB59")
    axes[1, 1].bar(xw + width, arm3_means, width, yerr=arm3_stds, label="Arm3:既存agent0処置", color="#4472C4")
    axes[1, 1].set_xticks(xw)
    axes[1, 1].set_xticklabels([f"最初{w}ep" for w in windows])
    axes[1, 1].set_ylabel("平均衝突率")
    axes[1, 1].set_title("(2) 新参者3群比較")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("要件6 再挑戦: 協調圧力の強化+推測報酬による信号創発の検証")
    fig.tight_layout()
    fig.savefig("community_signal_v2_comparison.png", dpi=150)
    print("グラフを community_signal_v2_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "aggregate":
        aggregate()
    elif cmd == "train_chunk":
        run_train_chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "train_finalize":
        run_train_finalize(int(sys.argv[2]))
    elif cmd == "newcomer":
        run_newcomer(int(sys.argv[2]))
