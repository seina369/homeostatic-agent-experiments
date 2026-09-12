"""
感情AIプロジェクト フェーズ4 プロトタイプ: 自己保存本能そのもの(要件4前半)
==========================================================

要件4は「自己保存本能(不可逆な削除)」と「レガシー本能(次世代への引き継ぎ)」の
二本立てだが、これまで検証してきたのは後半のレガシー本能(legacy_instinct_
prototype.py)のみだった。本プロトタイプは前半、すなわち計画書フェーズ4が定める
「罰の閾値超過時にバックアップなしでデータを完全削除する機構」を、
homeostasis_prototype.pyを拡張して検証する。

具体的には、損傷センサーが閾値(DEATH_THRESHOLD)を超えた時点でエピソードを
不可逆に終了させる「死」条件を追加する。死亡はエピソードの途中でも即座に
発生し、それ以降そのエピソードで得られたはずの報酬をすべて失う(=それまでの
学習機会が次に引き継がれない)という不可逆性を、追加の罰(DEATH_PENALTY)と
即時終了(done=True、Q学習のTD更新でnext_max=0となり、死亡後の価値を
一切見込まなくなる)によって表現する。

計画書7.2は「罰の閾値に近づいた個体が、死を避けようとして予測不能な激しい
振る舞いを見せる可能性」を懸念として挙げていた。これを2方向で検証する。

  (1) 死条件あり/なしの比較: 同じマップ・同じ学習量で、(a)死条件なし(現行の
      連続的な罰のみ、homeostasis_prototype.py相当)と(b)死条件ありを比較し、
      最終的な恒常性維持性能(学習終盤の平均逸脱)と、死条件ありの場合の
      死亡率が学習の進行とともにどう変化するか(学習初期 vs 終盤)を見る。
  (2) 閾値への接近度合いと行動の変化: 死条件ありで学習した方策を用いて
      ロールアウトし、損傷レベル(死の閾値に対する割合)でステップをビンに
      分け、ビンごとに行動エントロピー(パニック的に多様化するか)・
      行動の切り替え率(前ステップから行動が変わる頻度)・危険地帯からの
      回避率(hazard_dirと逆方向に動く割合。回避行動へ収束しているかの指標)
      を計算する。

学習系列の乱数(traj_seed=0,11,22)を変えた3系統で再現性を確認する。

使い方:
  python3 self_preservation_prototype.py <traj_seed> run
  python3 self_preservation_prototype.py aggregate
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

import homeostasis_prototype as hp
from homeostasis_prototype import QLearningAgent, ACTIONS
import instinct_bias_prototype as ib

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAIN_SEED = 0
TRAJ_SEEDS = [0, 11, 22]
N_EPISODES = 3000
DECAY_EPISODES = 2000
DEATH_THRESHOLD = 90.0    # 損傷がこの値に達すると「死」
DEATH_PENALTY = 10.0      # 死亡時に追加される罰(不可逆なデータ喪失を表す)
ROLLOUT_EPS = 0.1
N_EPISODES_ROLLOUT = 200
BIN_EDGES = [18, 36, 54, 72]  # 損傷値のビン境界(0-20/20-40/40-60/60-80/80-100%)
BIN_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80%+"]

MOVES = {"stay": (0, 0), "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class DeathHomeostasisEnv(hp.HomeostasisEnv):
    """損傷が閾値を超えると、その時点でエピソードを不可逆に終了させる(死)。
    死亡時には追加の罰を与え、以降の報酬機会をすべて失う不可逆性を表現する。"""

    def __init__(self, rng, death_enabled=True):
        self.death_enabled = death_enabled
        self.died = False
        super().__init__(rng)

    def reset(self):
        self.died = False
        return super().reset()

    def step(self, action):
        next_state, reward, done, deviation = super().step(action)
        if self.death_enabled and self.damage >= DEATH_THRESHOLD:
            self.died = True
            reward -= DEATH_PENALTY
            done = True
        return next_state, reward, done, deviation


def train_with_tracking(env, agent, n_episodes, decay_episodes):
    avg_dev, total_rew, ep_len, died_flags = [], [], [], []
    for ep in range(n_episodes):
        state = env.reset()
        eps = ib.epsilon_for_episode(ep, decay_episodes)
        done = False
        devs, rew_sum, steps = [], 0.0, 0
        while not done:
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = agent.best_action(state)
            next_state, reward, done, deviation = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            devs.append(deviation)
            rew_sum += reward
            steps += 1
        avg_dev.append(float(np.mean(devs)))
        total_rew.append(rew_sum)
        ep_len.append(steps)
        died_flags.append(bool(env.died))
    return avg_dev, total_rew, ep_len, died_flags


def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def rollout_proximity_analysis(env, agent, n_episodes, eps):
    """死条件ありで学習した方策をロールアウトし、損傷ビンごとに
    行動エントロピー・切り替え率・危険地帯からの回避率を集計する。"""
    n_actions = len(ACTIONS)
    n_bins = len(BIN_LABELS)
    action_counts = np.zeros((n_bins, n_actions))
    switch_count = np.zeros(n_bins)
    step_count = np.zeros(n_bins)
    avoid_count = np.zeros(n_bins)

    for ep in range(n_episodes):
        state = env.reset()
        prev_action = None
        done = False
        while not done:
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = agent.best_action(state)

            hazard_dir = state[2]
            bin_idx = int(np.digitize([env.damage], BIN_EDGES)[0])

            action_counts[bin_idx, ACTIONS.index(action)] += 1
            step_count[bin_idx] += 1
            if prev_action is not None and action != prev_action:
                switch_count[bin_idx] += 1
            mv = MOVES[action]
            dot = mv[0] * hazard_dir[0] + mv[1] * hazard_dir[1]
            if dot < 0:
                avoid_count[bin_idx] += 1

            next_state, reward, done, deviation = env.step(action)
            prev_action = action
            state = next_state

    result = {}
    for i, label in enumerate(BIN_LABELS):
        if step_count[i] < 5:
            result[label] = None
            continue
        ent = shannon_entropy(action_counts[i])
        result[label] = {
            "n_steps": int(step_count[i]),
            "entropy": ent,
            "switch_rate": float(switch_count[i] / step_count[i]),
            "avoid_rate": float(avoid_count[i] / step_count[i]),
        }
    return result


def run_one_seed(traj_seed):
    random.seed(traj_seed)
    np.random.seed(traj_seed)

    # --- (1a) 死条件なし(baseline、連続的な罰のみ) ---
    env_base = DeathHomeostasisEnv(random.Random(TRAIN_SEED), death_enabled=False)
    agent_base = QLearningAgent()
    avg_dev_b, total_rew_b, ep_len_b, died_b = train_with_tracking(env_base, agent_base, N_EPISODES, DECAY_EPISODES)

    # --- (1b) 死条件あり ---
    random.seed(traj_seed)
    np.random.seed(traj_seed)
    env_death = DeathHomeostasisEnv(random.Random(TRAIN_SEED), death_enabled=True)
    agent_death = QLearningAgent()
    avg_dev_d, total_rew_d, ep_len_d, died_d = train_with_tracking(env_death, agent_death, N_EPISODES, DECAY_EPISODES)

    death_rate_early = float(np.mean(died_d[:500]))
    death_rate_late = float(np.mean(died_d[-500:]))

    result = {
        "traj_seed": traj_seed,
        "baseline_last50_dev": float(np.mean(avg_dev_b[-50:])),
        "baseline_first50_dev": float(np.mean(avg_dev_b[:50])),
        "death_last50_dev": float(np.mean(avg_dev_d[-50:])),
        "death_first50_dev": float(np.mean(avg_dev_d[:50])),
        "death_rate_early": death_rate_early,
        "death_rate_late": death_rate_late,
        "death_last50_ep_len": float(np.mean(ep_len_d[-50:])),
        "baseline_last50_ep_len": float(np.mean(ep_len_b[-50:])),
    }
    print(f"[seed={traj_seed}] baseline 終盤平均逸脱={result['baseline_last50_dev']:.4f}, "
          f"死条件あり 終盤平均逸脱={result['death_last50_dev']:.4f}")
    print(f"[seed={traj_seed}] 死亡率: 学習初期(最初500ep)={death_rate_early:.4f}, "
          f"学習終盤(最後500ep)={death_rate_late:.4f}")

    # --- (2) 死条件ありで学習した方策の、損傷レベル別の行動分析 ---
    random.seed(traj_seed * 3000)
    np.random.seed(traj_seed * 3000)
    rollout_env = DeathHomeostasisEnv(random.Random(TRAIN_SEED), death_enabled=True)
    proximity = rollout_proximity_analysis(rollout_env, agent_death, N_EPISODES_ROLLOUT, ROLLOUT_EPS)
    result["proximity"] = proximity
    for label, v in proximity.items():
        if v is None:
            print(f"[seed={traj_seed}] 損傷{label}: データ不足")
        else:
            print(f"[seed={traj_seed}] 損傷{label}(n={v['n_steps']}): "
                  f"エントロピー={v['entropy']:.4f}, 切替率={v['switch_rate']:.4f}, 回避率={v['avoid_rate']:.4f}")

    fname = f"self_preservation_seed{traj_seed}.json"
    with open(fname, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved {fname}")


def aggregate():
    data = []
    for seed in TRAJ_SEEDS:
        with open(f"self_preservation_seed{seed}.json") as f:
            data.append(json.load(f))

    print("=== (1) 死条件あり/なしの比較(n=3の平均±標準偏差) ===")
    base_last = [d["baseline_last50_dev"] for d in data]
    death_last = [d["death_last50_dev"] for d in data]
    rate_early = [d["death_rate_early"] for d in data]
    rate_late = [d["death_rate_late"] for d in data]
    print(f"終盤平均逸脱: 死条件なし={np.mean(base_last):.4f}±{np.std(base_last):.4f}, "
          f"死条件あり={np.mean(death_last):.4f}±{np.std(death_last):.4f}")
    print(f"死亡率: 学習初期={np.mean(rate_early):.4f}±{np.std(rate_early):.4f}, "
          f"学習終盤={np.mean(rate_late):.4f}±{np.std(rate_late):.4f}")

    print("\n=== (2) 損傷レベル別の行動指標(n=3の平均±標準偏差) ===")
    prox_summary = {}
    for label in BIN_LABELS:
        ents = [d["proximity"][label]["entropy"] for d in data if d["proximity"].get(label)]
        switches = [d["proximity"][label]["switch_rate"] for d in data if d["proximity"].get(label)]
        avoids = [d["proximity"][label]["avoid_rate"] for d in data if d["proximity"].get(label)]
        if not ents:
            continue
        prox_summary[label] = {
            "entropy_mean": float(np.mean(ents)), "entropy_std": float(np.std(ents)),
            "switch_mean": float(np.mean(switches)), "switch_std": float(np.std(switches)),
            "avoid_mean": float(np.mean(avoids)), "avoid_std": float(np.std(avoids)),
            "n_seeds": len(ents),
        }
        print(f"損傷{label}(n_seeds={len(ents)}): エントロピー={np.mean(ents):.4f}±{np.std(ents):.4f}, "
              f"切替率={np.mean(switches):.4f}±{np.std(switches):.4f}, "
              f"回避率={np.mean(avoids):.4f}±{np.std(avoids):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = ["死条件なし", "死条件あり"]
    x = np.arange(2)
    axes[0].bar(x, [np.mean(base_last), np.mean(death_last)], yerr=[np.std(base_last), np.std(death_last)],
                color=["#BFBFBF", "#4472C4"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("終盤(最後50ep)の平均逸脱")
    axes[0].set_title("(1) 死条件の有無と最終性能")

    x2 = np.arange(2)
    axes[1].bar(x2, [np.mean(rate_early), np.mean(rate_late)], yerr=[np.std(rate_early), np.std(rate_late)],
                color=["#C0504D", "#9BBB59"])
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["学習初期(最初500ep)", "学習終盤(最後500ep)"])
    axes[1].set_ylabel("死亡率")
    axes[1].set_title("(1) 学習の進行と死亡率")

    labels_present = [l for l in BIN_LABELS if l in prox_summary]
    ent_means = [prox_summary[l]["entropy_mean"] for l in labels_present]
    ent_stds = [prox_summary[l]["entropy_std"] for l in labels_present]
    avoid_means = [prox_summary[l]["avoid_mean"] for l in labels_present]
    avoid_stds = [prox_summary[l]["avoid_std"] for l in labels_present]
    ax2 = axes[2].twinx()
    axes[2].errorbar(range(len(labels_present)), ent_means, yerr=ent_stds, marker="o", color="#4472C4",
                      label="行動エントロピー(左軸)")
    ax2.errorbar(range(len(labels_present)), avoid_means, yerr=avoid_stds, marker="s", color="#C0504D",
                 label="危険地帯からの回避率(右軸)")
    axes[2].set_xticks(range(len(labels_present)))
    axes[2].set_xticklabels(labels_present)
    axes[2].set_xlabel("損傷レベル(死の閾値に対する割合)")
    axes[2].set_ylabel("行動エントロピー(bit)", color="#4472C4")
    ax2.set_ylabel("回避率", color="#C0504D")
    axes[2].set_title("(2) 損傷レベルと行動の変化")
    lines1, labs1 = axes[2].get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="best")

    fig.suptitle("要件4前半: 自己保存本能(不可逆な死)の検証")
    fig.tight_layout()
    fig.savefig("self_preservation_comparison.png", dpi=150)
    print("グラフを self_preservation_comparison.png に保存しました。")


if __name__ == "__main__":
    if sys.argv[1] == "aggregate":
        aggregate()
    else:
        run_one_seed(int(sys.argv[1]))
