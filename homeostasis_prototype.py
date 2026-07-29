"""
感情AIプロジェクト フェーズ1 縮小版プロトタイプ
=====================================

計画書「感情を持つAIプロジェクト計画書」フェーズ1(基盤研究:センサー・恒常性システム)の
考え方を、実機を使わず小さなグリッドワールドのシミュレーションで検証する学習用プログラム。

やっていること:
  1. エージェントに3つの仮想センサーを持たせる(エネルギー・体温・損傷)
  2. 各センサーの「最適値からの逸脱」を毎ステップ計算する(恒常性モデル)
  3. 逸脱の大きさをそのまま罰(負の報酬)に変換する(要件1・要件2に対応する最小構成)
  4. Q学習でエージェントに、逸脱を減らす行動(食料へ向かう、日陰へ入る、危険地帯を避ける)を
     学習させる
  5. 学習の進み具合をグラフにして、恒常性からの逸脱がエピソードを重ねるごとに
     小さくなっていくことを確認する

注意: これは概念実証のためのごく小さな例であり、計画書のフェーズ3(外部監督組織)を
経ていない。罰の閾値超過による「不可逆な削除」のような、本物の実存的な賭け金は
一切実装していない(意図的に、シミュレーションのエピソードが一定歩数で終わるだけで、
個体が消滅するわけではない)。
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

random.seed(0)
np.random.seed(0)

# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------
GRID_SIZE = 8
N_EPISODES = 3000
MAX_STEPS = 120
ACTIONS = ["up", "down", "left", "right", "stay"]

OPTIMAL_ENERGY = 100.0
OPTIMAL_TEMP = 20.0
OPTIMAL_DAMAGE = 0.0

ENERGY_DECAY_PER_STEP = 2.0
TEMP_DRIFT_STD = 1.5
DAMAGE_HEAL_PER_STEP = 1.0

ALPHA = 0.2      # 学習率
GAMMA = 0.95     # 割引率
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_EPISODES = 2000


def random_tiles(n, rng):
    return set((rng.randint(0, GRID_SIZE - 1), rng.randint(0, GRID_SIZE - 1)) for _ in range(n))


class HomeostasisEnv:
    """センサー・恒常性システムを持つ、ごく小さなグリッドワールド。"""

    def __init__(self, rng):
        self.rng = rng
        self.food_tiles = random_tiles(3, rng)      # エネルギーを補給する場所
        self.shelter_tiles = random_tiles(3, rng)    # 体温を最適値へ近づける場所
        self.hazard_tiles = random_tiles(4, rng)     # 損傷を与える危険地帯
        self.reset()

    def reset(self):
        self.pos = (GRID_SIZE // 2, GRID_SIZE // 2)
        self.energy = 100.0
        self.temperature = OPTIMAL_TEMP
        self.damage = 0.0
        self.t = 0
        return self.discrete_state()

    def _nearest_dir(self, tiles):
        # 現在地から最も近いタイルへの向き(-1/0/1のみ)を返す。
        # 絶対座標ではなく相対的な向きを状態にすることで、
        # 「グリッド上のどこにいても同じ関係なら同じ行動を学べる」ようにする。
        if not tiles:
            return (0, 0)
        x, y = self.pos
        best = min(tiles, key=lambda t: abs(t[0] - x) + abs(t[1] - y))
        dx = np.sign(best[0] - x)
        dy = np.sign(best[1] - y)
        return (int(dx), int(dy))

    def discrete_state(self):
        # Q学習の表引きのため、連続値のセンサーを粗く離散化する
        e_bin = int(np.clip(self.energy // 20, 0, 5))
        t_bin = int(np.clip((self.temperature - OPTIMAL_TEMP + 15) // 6, 0, 5))
        d_bin = int(np.clip(self.damage // 20, 0, 5))
        food_dir = self._nearest_dir(self.food_tiles)
        shelter_dir = self._nearest_dir(self.shelter_tiles)
        hazard_dir = self._nearest_dir(self.hazard_tiles)
        return (food_dir, shelter_dir, hazard_dir, e_bin, t_bin, d_bin)

    def step(self, action):
        x, y = self.pos
        if action == "up":
            y = max(0, y - 1)
        elif action == "down":
            y = min(GRID_SIZE - 1, y + 1)
        elif action == "left":
            x = max(0, x - 1)
        elif action == "right":
            x = min(GRID_SIZE - 1, x + 1)
        self.pos = (x, y)

        # --- センサーの自然な変化(恒常性モデルの土台) ---
        self.energy -= ENERGY_DECAY_PER_STEP
        self.temperature += self.rng2_normal() * TEMP_DRIFT_STD
        self.damage = max(0.0, self.damage - DAMAGE_HEAL_PER_STEP)

        # --- 環境との相互作用 ---
        if self.pos in self.food_tiles:
            self.energy += 40.0
        if self.pos in self.shelter_tiles:
            self.temperature += (OPTIMAL_TEMP - self.temperature) * 0.5
        if self.pos in self.hazard_tiles:
            self.damage += 30.0

        self.energy = float(np.clip(self.energy, 0.0, 100.0))
        self.damage = float(np.clip(self.damage, 0.0, 100.0))

        # --- 恒常性からの逸脱 → 賞罰信号への変換(要件1・要件2の最小構成) ---
        dev_energy = abs(self.energy - OPTIMAL_ENERGY) / 100.0
        dev_temp = abs(self.temperature - OPTIMAL_TEMP) / 30.0
        dev_damage = abs(self.damage - OPTIMAL_DAMAGE) / 100.0
        deviation = dev_energy + dev_temp + dev_damage
        reward = -deviation  # 逸脱が大きいほど強い「罰」

        self.t += 1
        done = self.t >= MAX_STEPS or self.energy <= 0.0
        return self.discrete_state(), reward, done, deviation

    def rng2_normal(self):
        return np.random.randn()


class QLearningAgent:
    def __init__(self):
        self.q = {}

    def _key(self, state, action):
        return (state, action)

    def q_value(self, state, action):
        return self.q.get(self._key(state, action), 0.0)

    def best_action(self, state):
        values = [self.q_value(state, a) for a in ACTIONS]
        return ACTIONS[int(np.argmax(values))]

    def update(self, state, action, reward, next_state, done):
        current = self.q_value(state, action)
        next_max = 0.0 if done else max(self.q_value(next_state, a) for a in ACTIONS)
        target = reward + GAMMA * next_max
        self.q[self._key(state, action)] = current + ALPHA * (target - current)


def epsilon_for_episode(ep):
    frac = min(1.0, ep / EPS_DECAY_EPISODES)
    return EPS_START + (EPS_END - EPS_START) * frac


def run_training():
    rng = random.Random(0)
    env = HomeostasisEnv(rng)
    agent = QLearningAgent()

    avg_deviation_per_episode = []
    total_reward_per_episode = []

    for ep in range(N_EPISODES):
        state = env.reset()
        eps = epsilon_for_episode(ep)
        done = False
        deviations = []
        total_reward = 0.0

        while not done:
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = agent.best_action(state)

            next_state, reward, done, deviation = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            deviations.append(deviation)
            total_reward += reward

        avg_deviation_per_episode.append(float(np.mean(deviations)))
        total_reward_per_episode.append(total_reward)

    return avg_deviation_per_episode, total_reward_per_episode


def moving_average(x, window=50):
    x = np.array(x)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


if __name__ == "__main__":
    avg_dev, total_reward = run_training()

    print(f"最初の50エピソードの平均逸脱: {np.mean(avg_dev[:50]):.4f}")
    print(f"最後の50エピソードの平均逸脱: {np.mean(avg_dev[-50:]):.4f}")
    print(f"最初の50エピソードの平均報酬: {np.mean(total_reward[:50]):.2f}")
    print(f"最後の50エピソードの平均報酬: {np.mean(total_reward[-50:]):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(avg_dev, color="lightgray", linewidth=0.5, label="毎エピソード")
    axes[0].plot(
        np.arange(len(moving_average(avg_dev))) + 25,
        moving_average(avg_dev),
        color="#2F5496",
        linewidth=2,
        label="移動平均(50episode)",
    )
    axes[0].set_xlabel("エピソード")
    axes[0].set_ylabel("平均逸脱(小さいほど恒常性が保たれている)")
    axes[0].set_title("恒常性からの逸脱の推移")
    axes[0].legend()

    axes[1].plot(total_reward, color="lightgray", linewidth=0.5, label="毎エピソード")
    axes[1].plot(
        np.arange(len(moving_average(total_reward))) + 25,
        moving_average(total_reward),
        color="#C0504D",
        linewidth=2,
        label="移動平均(50episode)",
    )
    axes[1].set_xlabel("エピソード")
    axes[1].set_ylabel("累計報酬(賞罰信号の合計)")
    axes[1].set_title("報酬の推移")
    axes[1].legend()

    fig.suptitle("フェーズ1プロトタイプ: センサー逸脱 → 賞罰 → 学習された恒常性維持行動")
    fig.tight_layout()
    fig.savefig("homeostasis_learning_curve.png", dpi=150)
    print("グラフを homeostasis_learning_curve.png に保存しました。")
