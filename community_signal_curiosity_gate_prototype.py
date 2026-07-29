"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 協調限定の新奇性ゲート×好奇心報酬による信号創発の検証
==========================================================

これまでの信号創発実験はすべて、推測ゲーム(相手の内部状態を正しく当てたら
ボーナス)という、測定したい量(信号の意味)そのものに近い直接報酬に依存
していた。これはGoodhart的な設計依存(測定対象を目的関数に組み込んでいる)
という懸念がある。本プロトタイプは、信号の正誤に一切報酬を与えず、「新しい
行動・状態の発見」への内発的動機(好奇心)だけから、結果として協調と信号
創発が生まれるかを検証する。

**環境改修(協調ゲート)**: 4×4の中核グリッドに加え、6×6の拡張グリッドを
定義する。中核グリッドの対角(S1=(0,0)、S2=(3,3))にスイッチマスを置き、
agent0はS1、agent1はS2が「担当」スイッチとなる(観測にswitch_dir=担当
スイッチへの方向を追加)。両エージェントが同一ステップでそれぞれの担当
スイッチに同時に乗ったときだけ、GATE_DURATION(15)ステップの間だけ拡張
グリッド(外側のリング領域)への移動が解禁され、そこには中核グリッドには
ない追加の食料・シェルタータイルが配置されている(=新しい状態・行動の
組み合わせが一時的に出現する)。ゲートが閉じている間は中核グリッドの外に
出られず(移動がクリップされる)、ゲートが閉じる瞬間にリング領域にいた
エージェントは強制的に中核へ押し戻される。観測にはgate_open(0/1)も含める。

**報酬設計**: 各エージェントは独立した(状態,行動)訪問回数表N(s,a)を持ち、
好奇心報酬 bonus = BETA_CURIOSITY / sqrt(N(s,a)+1) を通常の恒常性維持報酬
(逸脱の負値、衝突ペナルティ)に上乗せする。GuessAgent・推測ゲームは一切
存在せず、信号(signal)は他の5行動と全く同じ「1つの行動選択肢」として
扱われ、特別な報酬は一切紐付かない。

**3条件×2学習量**:
  - 提案設計: gate_enabled=True, curiosity_enabled=True
  - 対照A: gate_enabled=True, curiosity_enabled=False(通常報酬のみ、ゲートの
    存在自体が偶然の協調を生むかを見る)
  - 対照B: gate_enabled=False(現行4×4環境), curiosity_enabled=True(好奇心
    報酬単独が既存の漠然報酬と同様のプーリング均衡希薄化に陥るかを見る)
各条件についてN_EPISODES=3500版と8000版(好奇心報酬は発見に時間がかかり
やすいため)を並行して実行する。まずtraj_seed=0,11,22のn=3で予備検証し、
有望ならn=15へ拡大する。

**指標**: 信号と内部状態のMI(従来と同じ定義)、協調ゲート到達率・到達までの
平均ステップ数、好奇心ボーナス平均値の推移、信号送信タイミングと協調ゲート
到達成功の相関(協調ゲート到達の直前3ステップ以内に信号が送られていた
割合 vs 通常時の信号送信率、のリフト比)。

45秒のbash呼び出し制限に対応するため、community_signal_iterated_v2_
prototype.pyのgen_multi_chunkと同じ「時間主導」のチャンク実行方式を採用する。

使い方:
  python3 community_signal_curiosity_gate_prototype.py multi_chunk <condition> <traj_seed> <n_episodes>
  python3 community_signal_curiosity_gate_prototype.py aggregate <n_episodes>
  条件(condition): proposed(gate+curiosity) / controlA(gate only) / controlB(curiosity only)
"""

import sys, json, pickle, random, time
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
import community_signal_v2_prototype as m  # MOVES, act, mutual_info_signal_vs_class等を再利用

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

ACTIONS_COMM = m.ACTIONS_COMM  # ["up","down","left","right","stay","signal"]
CORE_GRID_SIZE = 4
EXT_GRID_SIZE = 6
S1 = (0, 0)   # agent0の担当スイッチ
S2 = (3, 3)   # agent1の担当スイッチ
GATE_DURATION = 15
BETA_CURIOSITY = 1.0
COLLISION_PENALTY = m.COLLISION_PENALTY  # 8.0、既存実験と同一
DECAY_EPISODES = 2500
TRAJ_SEEDS_PRELIM = [0, 11, 22]
CONDITIONS = ["proposed", "controlA", "controlB"]
CONDITION_FLAGS = {
    "proposed": {"gate_enabled": True, "curiosity_enabled": True},
    "controlA": {"gate_enabled": True, "curiosity_enabled": False},
    "controlB": {"gate_enabled": False, "curiosity_enabled": True},
}
N_ROLLOUT_EPISODES = 100
ROLLOUT_EPS = 0.1


def checkpoint_episodes_for(n_episodes):
    if n_episodes <= 3500:
        return [300, 1500, 3500]
    return [500, 1500, 3000, 5000, 8000]


class GateCuriosityEnv:
    """community_signal_v2のMultiAgentHomeostasisEnvを、協調限定の新奇性
    ゲート(gate_enabled時)を持つように拡張した2体環境。"""

    def __init__(self, rng, gate_enabled=True, collision_penalty=COLLISION_PENALTY):
        self.rng = rng
        self.gate_enabled = gate_enabled
        self.collision_penalty = collision_penalty
        self.food_tiles = hp.random_tiles(3, rng)     # 中核グリッド内
        self.shelter_tiles = hp.random_tiles(3, rng)
        self.hazard_tiles = hp.random_tiles(4, rng)
        if gate_enabled:
            self.ring_food_tiles = self._random_ring_tiles(3, rng)
            self.ring_shelter_tiles = self._random_ring_tiles(2, rng)
        else:
            self.ring_food_tiles = set()
            self.ring_shelter_tiles = set()
        self.reset()

    def _random_ring_tiles(self, n, rng):
        candidates = [(x, y) for x in range(EXT_GRID_SIZE) for y in range(EXT_GRID_SIZE)
                      if x >= CORE_GRID_SIZE or y >= CORE_GRID_SIZE]
        idx = rng.sample(range(len(candidates)), min(n, len(candidates)))
        return set(candidates[i] for i in idx)

    def reset(self):
        c = CORE_GRID_SIZE // 2
        self.pos = [(max(0, c - 1), c), (min(CORE_GRID_SIZE - 1, c), c)]
        self.energy = [100.0, 100.0]
        self.temperature = [hp.OPTIMAL_TEMP, hp.OPTIMAL_TEMP]
        self.damage = [0.0, 0.0]
        self.last_signal = [0, 0]
        self.gate_timer = 0
        self.gate_reached_this_episode = False
        self.gate_first_reach_step = None
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

    def _accessible_food_tiles(self):
        if self.gate_enabled and self.gate_timer > 0:
            return self.food_tiles | self.ring_food_tiles
        return self.food_tiles

    def _accessible_shelter_tiles(self):
        if self.gate_enabled and self.gate_timer > 0:
            return self.shelter_tiles | self.ring_shelter_tiles
        return self.shelter_tiles

    def observe(self, i):
        j = 1 - i
        e_bin = int(np.clip(self.energy[i] // 20, 0, 5))
        t_bin = int(np.clip((self.temperature[i] - hp.OPTIMAL_TEMP + 15) // 6, 0, 5))
        d_bin = int(np.clip(self.damage[i] // 20, 0, 5))
        food_dir = self._nearest_dir(self.pos[i], self._accessible_food_tiles())
        shelter_dir = self._nearest_dir(self.pos[i], self._accessible_shelter_tiles())
        hazard_dir = self._nearest_dir(self.pos[i], self.hazard_tiles)
        partner_dir = self._rel_dir(self.pos[i], self.pos[j])
        if self.gate_enabled:
            my_switch = S1 if i == 0 else S2
            switch_dir = self._rel_dir(self.pos[i], my_switch)
            gate_open = 1 if self.gate_timer > 0 else 0
        else:
            switch_dir = (0, 0)
            gate_open = 0
        return (food_dir, shelter_dir, hazard_dir, e_bin, t_bin, d_bin,
                self.last_signal[j], partner_dir, switch_dir, gate_open)

    def dominant_deviation(self, i):
        dev_energy = abs(self.energy[i] - hp.OPTIMAL_ENERGY) / 100.0
        dev_temp = abs(self.temperature[i] - hp.OPTIMAL_TEMP) / 30.0
        dev_damage = abs(self.damage[i] - hp.OPTIMAL_DAMAGE) / 100.0
        return int(np.argmax([dev_energy, dev_temp, dev_damage]))

    def step(self, actions):
        bound = EXT_GRID_SIZE if (self.gate_enabled and self.gate_timer > 0) else CORE_GRID_SIZE
        new_pos = list(self.pos)
        new_signal = [0, 0]
        for i, a in enumerate(actions):
            if a == "signal":
                new_signal[i] = 1
            else:
                dx, dy = m.MOVES[a]
                x, y = self.pos[i]
                new_pos[i] = (int(np.clip(x + dx, 0, bound - 1)), int(np.clip(y + dy, 0, bound - 1)))
        self.pos = new_pos
        self.last_signal = new_signal

        gate_triggered = False
        if self.gate_enabled:
            reached = (self.pos[0] == S1 and self.pos[1] == S2)
            if reached:
                self.gate_timer = GATE_DURATION
                gate_triggered = True
                if not self.gate_reached_this_episode:
                    self.gate_reached_this_episode = True
                    self.gate_first_reach_step = self.t + 1
            elif self.gate_timer > 0:
                self.gate_timer -= 1
                if self.gate_timer == 0:
                    # ゲートが閉じた瞬間、リング領域にいたら中核へ強制的に押し戻す
                    self.pos = [(min(p[0], CORE_GRID_SIZE - 1), min(p[1], CORE_GRID_SIZE - 1)) for p in self.pos]

        rewards = [0.0, 0.0]
        deviations = [0.0, 0.0]
        for i in range(2):
            self.energy[i] -= hp.ENERGY_DECAY_PER_STEP
            self.temperature[i] += np.random.randn() * hp.TEMP_DRIFT_STD
            self.damage[i] = max(0.0, self.damage[i] - hp.DAMAGE_HEAL_PER_STEP)
            if self.pos[i] in self._accessible_food_tiles():
                self.energy[i] += 40.0
            if self.pos[i] in self._accessible_shelter_tiles():
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
        return [self.observe(0), self.observe(1)], rewards, done, deviations, collided, gate_triggered


def curiosity_bonus(visit_counts, obs, action):
    key = (obs, action)
    n = visit_counts.get(key, 0)
    bonus = BETA_CURIOSITY / np.sqrt(n + 1)
    visit_counts[key] = n + 1
    return bonus


def run_episode(env, agent0, agent1, eps0, eps1, curiosity_enabled,
                 visits0=None, visits1=None, learn0=True, learn1=True):
    obs = env.reset()
    done = False
    devs, collisions, steps = [], 0, 0
    curiosity_hist = []
    signal_before_gate_records = []  # (either_signaled_this_step, gate_triggered_this_step)
    while not done:
        a0 = m.act(agent0, obs[0], eps0)
        a1 = m.act(agent1, obs[1], eps1)
        either_signaled = (a0 == "signal") or (a1 == "signal")

        next_obs, base_rewards, done, deviations, collided, gate_triggered = env.step([a0, a1])
        signal_before_gate_records.append((either_signaled, gate_triggered))

        total_r0, total_r1 = base_rewards[0], base_rewards[1]
        if curiosity_enabled:
            b0 = curiosity_bonus(visits0, obs[0], a0)
            b1 = curiosity_bonus(visits1, obs[1], a1)
            total_r0 += b0
            total_r1 += b1
            curiosity_hist.append((b0 + b1) / 2.0)

        if learn0:
            agent0.update(obs[0], a0, total_r0, next_obs[0], done)
        if learn1:
            agent1.update(obs[1], a1, total_r1, next_obs[1], done)

        obs = next_obs
        devs.append((deviations[0] + deviations[1]) / 2.0)
        collisions += int(collided)
        steps += 1

    avg_dev = float(np.mean(devs))
    coll_rate = collisions / steps
    avg_curiosity = float(np.mean(curiosity_hist)) if curiosity_hist else 0.0

    # 信号タイミングとゲート到達の相関(直前3ステップ以内に信号があったか)
    lift_records = []
    for t_idx in range(len(signal_before_gate_records)):
        _, gate_triggered = signal_before_gate_records[t_idx]
        window_start = max(0, t_idx - 2)
        signaled_recently = any(signal_before_gate_records[k][0] for k in range(window_start, t_idx + 1))
        lift_records.append((signaled_recently, gate_triggered))

    return avg_dev, coll_rate, avg_curiosity, env.gate_reached_this_episode, env.gate_first_reach_step, lift_records


def train_range(env, agent0, agent1, condition, start_ep, end_ep, decay_episodes,
                 visits0, visits1, checkpoint_eps=None):
    checkpoint_eps = checkpoint_eps or []
    checkpoint_set = set(checkpoint_eps)
    checkpoints = {}
    avg_dev_hist, coll_hist, curiosity_hist, gate_reached_hist, gate_step_hist = [], [], [], [], []
    lift_signal, lift_no_signal, lift_trig_signal, lift_trig_no_signal = 0, 0, 0, 0
    curiosity_enabled = CONDITION_FLAGS[condition]["curiosity_enabled"]
    for ep in range(start_ep, end_ep):
        eps = ib.epsilon_for_episode(ep, decay_episodes)
        avg_dev, coll_rate, avg_cur, gate_reached, gate_step, lift_records = run_episode(
            env, agent0, agent1, eps, eps, curiosity_enabled, visits0, visits1
        )
        avg_dev_hist.append(avg_dev)
        coll_hist.append(coll_rate)
        curiosity_hist.append(avg_cur)
        gate_reached_hist.append(int(gate_reached))
        gate_step_hist.append(gate_step)
        for signaled_recently, triggered in lift_records:
            if signaled_recently:
                lift_signal += 1
                lift_trig_signal += int(triggered)
            else:
                lift_no_signal += 1
                lift_trig_no_signal += int(triggered)
        if (ep + 1) in checkpoint_set:
            checkpoints[ep + 1] = (dict(agent0.q), dict(agent1.q), dict(visits0), dict(visits1))
    lift_stats = {
        "lift_signal_steps": lift_signal, "lift_trig_given_signal": lift_trig_signal,
        "lift_no_signal_steps": lift_no_signal, "lift_trig_given_no_signal": lift_trig_no_signal,
    }
    return avg_dev_hist, coll_hist, curiosity_hist, gate_reached_hist, gate_step_hist, lift_stats, checkpoints


def rollout_for_signal_analysis(env, agent0, agent1, n_episodes, eps):
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
            next_obs, rewards, done, deviations, collided, gate_triggered = env.step([a0, a1])
            obs = next_obs
    return records


def multi_chunk(condition, traj_seed, target_end_ep, time_budget=36.0, sub_step=250):
    t_start = time.time()
    flags = CONDITION_FLAGS[condition]
    checkpoint_eps = checkpoint_episodes_for(target_end_ep)
    state_file = f"gatecur_state_{condition}_seed{traj_seed}_n{target_end_ep}.pkl"
    n_sub_chunks = 0
    while True:
        try:
            with open(state_file, "rb") as f:
                state = pickle.load(f)
            cur_last_ep = state["last_ep"]
        except FileNotFoundError:
            cur_last_ep = -1
        if cur_last_ep >= target_end_ep:
            break
        if (time.time() - t_start) > time_budget:
            print(f"[{condition} seed={traj_seed} N={target_end_ep}] 時間予算({time_budget}s)到達、"
                  f"{max(cur_last_ep,0)}epで一旦終了")
            return

        if cur_last_ep == -1:
            random.seed(traj_seed * 100000 + hash(condition) % 1000)
            np.random.seed((traj_seed * 100000 + hash(condition) % 1000) % (2**32 - 1))
            env = GateCuriosityEnv(random.Random(m.TRAIN_SEED), gate_enabled=flags["gate_enabled"])
            agent0, agent1 = QLearningAgent(), QLearningAgent()
            visits0, visits1 = {}, {}
            avg_dev_hist, coll_hist, curiosity_hist, gate_reached_hist, gate_step_hist = [], [], [], [], []
            all_checkpoints = {}
            all_lift = {"lift_signal_steps": 0, "lift_trig_given_signal": 0,
                        "lift_no_signal_steps": 0, "lift_trig_given_no_signal": 0}
            start_ep = 0
            print(f"[{condition} seed={traj_seed} N={target_end_ep}] 新規開始")
        else:
            env = state["env"]
            agent0, agent1 = state["agent0"], state["agent1"]
            visits0, visits1 = state["visits0"], state["visits1"]
            avg_dev_hist, coll_hist, curiosity_hist = state["avg_dev_hist"], state["coll_hist"], state["curiosity_hist"]
            gate_reached_hist, gate_step_hist = state["gate_reached_hist"], state["gate_step_hist"]
            all_checkpoints = state["checkpoints"]
            all_lift = state["lift"]
            start_ep = cur_last_ep
            random.setstate(state["random_state"])
            np.random.set_state(state["np_random_state"])

        next_ep = min(start_ep + sub_step, target_end_ep)
        dev_h, coll_h, cur_h, gr_h, gs_h, lift_stats, checkpoints = train_range(
            env, agent0, agent1, condition, start_ep, next_ep, DECAY_EPISODES,
            visits0, visits1, checkpoint_eps=checkpoint_eps
        )
        avg_dev_hist.extend(dev_h); coll_hist.extend(coll_h); curiosity_hist.extend(cur_h)
        gate_reached_hist.extend(gr_h); gate_step_hist.extend(gs_h)
        all_checkpoints.update(checkpoints)
        for k in all_lift:
            all_lift[k] += lift_stats[k]

        state = {
            "env": env, "agent0": agent0, "agent1": agent1, "visits0": visits0, "visits1": visits1,
            "avg_dev_hist": avg_dev_hist, "coll_hist": coll_hist, "curiosity_hist": curiosity_hist,
            "gate_reached_hist": gate_reached_hist, "gate_step_hist": gate_step_hist,
            "checkpoints": all_checkpoints, "lift": all_lift, "last_ep": next_ep,
            "random_state": random.getstate(), "np_random_state": np.random.get_state(),
        }
        with open(state_file, "wb") as f:
            pickle.dump(state, f)
        print(f"[{condition} seed={traj_seed} N={target_end_ep}] {next_ep}epまで完了 "
              f"(直近100ep衝突率={np.mean(coll_hist[-100:]):.4f}, ゲート到達率={np.mean(gate_reached_hist[-100:]):.4f}, "
              f"好奇心ボーナス平均={np.mean(curiosity_hist[-100:]) if curiosity_hist else 0:.4f})")
        n_sub_chunks += 1

    print(f"[{condition} seed={traj_seed} N={target_end_ep}] target_end_epに到達、finalizeを実行")
    finalize(condition, traj_seed, target_end_ep)


def finalize(condition, traj_seed, n_episodes):
    state_file = f"gatecur_state_{condition}_seed{traj_seed}_n{n_episodes}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    checkpoints = state["checkpoints"]
    coll_hist, curiosity_hist = state["coll_hist"], state["curiosity_hist"]
    gate_reached_hist, gate_step_hist = state["gate_reached_hist"], state["gate_step_hist"]
    lift = state["lift"]
    flags = CONDITION_FLAGS[condition]

    print(f"[{condition} seed={traj_seed} N={n_episodes}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"ゲート到達率={np.mean(gate_reached_hist[:500]):.4f}")
    print(f"[{condition} seed={traj_seed} N={n_episodes}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
          f"ゲート到達率={np.mean(gate_reached_hist[-500:]):.4f}")

    reach_steps = [s for s in gate_step_hist[-500:] if s is not None]
    avg_reach_step = float(np.mean(reach_steps)) if reach_steps else None

    p_trig_given_signal = (lift["lift_trig_given_signal"] / lift["lift_signal_steps"]) if lift["lift_signal_steps"] > 0 else None
    p_trig_given_no_signal = (lift["lift_trig_given_no_signal"] / lift["lift_no_signal_steps"]) if lift["lift_no_signal_steps"] > 0 else None
    lift_ratio = (p_trig_given_signal / p_trig_given_no_signal) if (p_trig_given_signal and p_trig_given_no_signal) else None

    checkpoint_eps = checkpoint_episodes_for(n_episodes)
    mi_by_checkpoint = {}
    for n_ep in checkpoint_eps:
        q0, q1, v0, v1 = checkpoints[n_ep]
        random.seed(traj_seed * 700000 + n_ep)
        np.random.seed((traj_seed * 700000 + n_ep) % (2**32 - 1))
        rollout_env = GateCuriosityEnv(random.Random(m.TRAIN_SEED), gate_enabled=flags["gate_enabled"])
        agent0 = QLearningAgent(); agent0.q = dict(q0)
        agent1 = QLearningAgent(); agent1.q = dict(q1)
        records = rollout_for_signal_analysis(rollout_env, agent0, agent1, N_ROLLOUT_EPISODES, ROLLOUT_EPS)
        mi, signal_rate, cond_dist, marg_dist = m.mutual_info_signal_vs_class(records)
        mi_by_checkpoint[str(n_ep)] = {"mi": mi, "signal_rate": signal_rate}
        print(f"[{condition} seed={traj_seed} N={n_episodes}] {n_ep}ep: I(signal;dominant_dev)={mi:.4f}bit, "
              f"信号送信率={signal_rate:.4f}")

    result = {
        "condition": condition, "traj_seed": traj_seed, "n_episodes": n_episodes,
        "collision_early": float(np.mean(coll_hist[:500])), "collision_late": float(np.mean(coll_hist[-500:])),
        "gate_reach_rate_early": float(np.mean(gate_reached_hist[:500])) if flags["gate_enabled"] else None,
        "gate_reach_rate_late": float(np.mean(gate_reached_hist[-500:])) if flags["gate_enabled"] else None,
        "avg_gate_first_reach_step_late": avg_reach_step,
        "curiosity_bonus_early": float(np.mean(curiosity_hist[:500])) if curiosity_hist else None,
        "curiosity_bonus_late": float(np.mean(curiosity_hist[-500:])) if curiosity_hist else None,
        "p_trigger_given_signal_recent": p_trig_given_signal,
        "p_trigger_given_no_signal_recent": p_trig_given_no_signal,
        "signal_gate_lift_ratio": lift_ratio,
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(f"gatecur_result_{condition}_seed{traj_seed}_n{n_episodes}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved gatecur_result_{condition}_seed{traj_seed}_n{n_episodes}.json")


def aggregate(n_episodes):
    print(f"=== 要件6: 協調限定の新奇性ゲート×好奇心報酬による信号創発の検証(N={n_episodes}) ===")
    checkpoint_eps = checkpoint_episodes_for(n_episodes)
    final_ep = checkpoint_eps[-1]
    summary = {}
    for cond in CONDITIONS:
        data = [json.load(open(f"gatecur_result_{cond}_seed{s}_n{n_episodes}.json")) for s in TRAJ_SEEDS_PRELIM]
        mis = [d["mi_by_checkpoint"][str(final_ep)]["mi"] for d in data]
        rates = [d["mi_by_checkpoint"][str(final_ep)]["signal_rate"] for d in data]
        gate_rates = [d["gate_reach_rate_late"] for d in data if d["gate_reach_rate_late"] is not None]
        reach_steps = [d["avg_gate_first_reach_step_late"] for d in data if d["avg_gate_first_reach_step_late"] is not None]
        cur_early = [d["curiosity_bonus_early"] for d in data if d["curiosity_bonus_early"] is not None]
        cur_late = [d["curiosity_bonus_late"] for d in data if d["curiosity_bonus_late"] is not None]
        lift_ratios = [d["signal_gate_lift_ratio"] for d in data if d["signal_gate_lift_ratio"] is not None]

        summary[cond] = {
            "mi_mean": float(np.mean(mis)), "mi_std": float(np.std(mis)), "mi_values": mis,
            "signal_rate_mean": float(np.mean(rates)),
            "gate_reach_rate_mean": float(np.mean(gate_rates)) if gate_rates else None,
            "avg_gate_reach_step_mean": float(np.mean(reach_steps)) if reach_steps else None,
            "curiosity_bonus_early_mean": float(np.mean(cur_early)) if cur_early else None,
            "curiosity_bonus_late_mean": float(np.mean(cur_late)) if cur_late else None,
            "signal_gate_lift_ratio_mean": float(np.mean(lift_ratios)) if lift_ratios else None,
        }
        print(f"\n[{cond}] MI({final_ep}ep)={np.mean(mis):.4f}±{np.std(mis):.4f}bit(系統別: {['%.4f'%x for x in mis]}), "
              f"信号送信率={np.mean(rates):.4f}")
        if gate_rates:
            print(f"  ゲート到達率(終盤)={np.mean(gate_rates):.4f}, 到達までの平均ステップ={summary[cond]['avg_gate_reach_step_mean']}")
        if cur_early:
            print(f"  好奇心ボーナス平均: 序盤={np.mean(cur_early):.4f} -> 終盤={np.mean(cur_late):.4f}")
        if lift_ratios:
            print(f"  信号タイミング->ゲート到達のリフト比: {np.mean(lift_ratios):.3f}(1.0=無相関)")

    # 提案設計 vs 対照A・対照Bの比較(Mann-WhitneyのU検定、n=3では参考程度)
    from scipy import stats as sstats
    mi_proposed = summary["proposed"]["mi_values"]
    mi_a = summary["controlA"]["mi_values"]
    mi_b = summary["controlB"]["mi_values"]
    try:
        u_a, p_a = sstats.mannwhitneyu(mi_proposed, mi_a, alternative="greater")
        u_b, p_b = sstats.mannwhitneyu(mi_proposed, mi_b, alternative="greater")
        print(f"\n提案 vs 対照A(片側マン・ホイットニー): U={u_a:.2f}, p={p_a:.4f}")
        print(f"提案 vs 対照B(片側マン・ホイットニー): U={u_b:.2f}, p={p_b:.4f}")
    except Exception as e:
        print(f"検定計算エラー(n=3では分解能不足の可能性): {e}")
        p_a, p_b = None, None

    with open(f"gatecur_summary_n{n_episodes}.json", "w") as f:
        json.dump({"summary": summary, "test_proposed_vs_A_p": p_a, "test_proposed_vs_B_p": p_b},
                   f, ensure_ascii=False, indent=2)
    print(f"saved gatecur_summary_n{n_episodes}.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(CONDITIONS))
    mi_means = [summary[c]["mi_mean"] for c in CONDITIONS]
    mi_stds = [summary[c]["mi_std"] for c in CONDITIONS]
    axes[0].bar(x, mi_means, yerr=mi_stds, color=["#2E7D32", "#4472C4", "#C0504D"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["提案(gate+curiosity)", "対照A(gateのみ)", "対照B(curiosityのみ)"], fontsize=8)
    axes[0].set_ylabel(f"I(signal;dominant_dev)[bit]({final_ep}ep)")
    axes[0].set_title(f"3条件でのMI比較(N={n_episodes})")

    gate_rates_plot = [summary[c]["gate_reach_rate_mean"] or 0 for c in ["proposed", "controlA"]]
    axes[1].bar(np.arange(2), gate_rates_plot, color=["#2E7D32", "#4472C4"])
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(["提案", "対照A"], fontsize=8)
    axes[1].set_ylabel("ゲート到達率(終盤500ep)")
    axes[1].set_title("協調ゲート到達率(gate_enabled条件のみ)")

    fig.suptitle(f"要件6: 協調限定の新奇性ゲート×好奇心報酬(N={n_episodes})")
    fig.tight_layout()
    fig.savefig(f"community_signal_curiosity_gate_n{n_episodes}_comparison.png", dpi=150)
    print(f"グラフを community_signal_curiosity_gate_n{n_episodes}_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "multi_chunk":
        multi_chunk(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "aggregate":
        aggregate(int(sys.argv[2]))
