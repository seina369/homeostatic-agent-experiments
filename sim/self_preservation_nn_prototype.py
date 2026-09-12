"""
感情AIプロジェクト フェーズ4 プロトタイプ: 要件4前半 自己保存本能の最小限NN移行
==========================================================

self_preservation_prototype.py(損傷閾値超過で不可逆に終了する「死」条件、
死亡率が学習初期46.2%→終盤9.1%まで低下・パニック的行動なし、という
タブラー版の結論)を基準に、要件6・7と同じ方針でエージェントの内部実装
だけをMLP(隠れ層32×32)+経験リプレイ+ターゲットネットワークのDQNに
置き換え、環境(DeathHomeostasisEnv、DEATH_THRESHOLD=90・DEATH_PENALTY=10)・
報酬・状態表現・学習量(3000ep)・ロールアウト分析手法は完全に同一に保つ。

DeathHomeostasisEnvの状態表現(food_dir, shelter_dir, hazard_dir, e_bin,
t_bin, d_bin)・行動空間(5行動: up/down/left/right/stay)は
homeostasis_nn_prototype.pyのDQNAgent/encode_state(9次元)とビット単位で
同一のため、homeostasis_nn_prototype.DQNAgentをそのまま再利用できる
(新規のエンコーダ設計は不要)。

検証すること:
  (1) 死条件あり/なしの比較: 終盤(最後50ep)の平均逸脱・死亡率(学習初期 vs 終盤)
  (2) 損傷レベル別の行動分析: 行動エントロピー・切替率・危険地帯からの回避率
      (パニック的行動=損傷が閾値に近づくほど行動が不安定化する、の有無)

規模: n=3(traj_seed=0,11,22)。タブラー版と大きく違う結果が出た場合のみ
n=15へ拡大する。

45秒のbash呼び出し制限に対応するため、時間主導のチャンク実行方式
(内部でepisodeを1つずつ進めながら経過時間を監視し、時間予算で状態を
保存して終了、次回呼び出しで自動再開)を採用する。

使い方:
  python3 self_preservation_nn_prototype.py chunk <traj_seed> [time_budget]
  python3 self_preservation_nn_prototype.py aggregate
"""

import sys, json, pickle, time, random
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import homeostasis_prototype as hp
from homeostasis_prototype import ACTIONS
import instinct_bias_prototype as ib
import homeostasis_nn_prototype as m

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
DEATH_THRESHOLD = 90.0
DEATH_PENALTY = 10.0
ROLLOUT_EPS = 0.1
N_EPISODES_ROLLOUT = 200
BIN_EDGES = [18, 36, 54, 72]
BIN_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80%+"]

MOVES = {"stay": (0, 0), "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class DeathHomeostasisEnv(hp.HomeostasisEnv):
    """タブラー版self_preservation_prototype.pyと同一定義。損傷が閾値を超えると
    その時点でエピソードを不可逆に終了させる(死)。死亡時には追加の罰を与え、
    以降の報酬機会をすべて失う不可逆性を表現する。"""

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


def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def run_train_episode(env, agent, eps):
    state = env.reset()
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
    return float(np.mean(devs)), rew_sum, steps, bool(env.died)


def rollout_proximity_analysis(env, agent, n_episodes, eps):
    """タブラー版と同一のロールアウト分析。agent.best_action(state)しか
    使わないため、m.DQNAgent/m.EvalPolicyのどちらでもそのまま動作する。"""
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


def chunk(traj_seed, time_budget=40.0):
    state_file = f"nn_selfpres_state_seed{traj_seed}.pkl"
    t_start = time.time()
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env_base, agent_base = state["env_base"], state["agent_base"]
        env_death, agent_death = state["env_death"], state["agent_death"]
        avg_dev_b, died_b = state["avg_dev_b"], state["died_b"]
        avg_dev_d, died_d = state["avg_dev_d"], state["died_d"]
        ep_done = state["ep_done"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        env_base = DeathHomeostasisEnv(random.Random(TRAIN_SEED), death_enabled=False)
        agent_base = m.DQNAgent(seed=traj_seed * 4 + 1)
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        env_death = DeathHomeostasisEnv(random.Random(TRAIN_SEED), death_enabled=True)
        agent_death = m.DQNAgent(seed=traj_seed * 4 + 2)
        avg_dev_b, died_b = [], []
        avg_dev_d, died_d = [], []
        ep_done = 0
        print(f"[selfpres-nn seed={traj_seed}] 新規開始")

    while ep_done < N_EPISODES:
        eps = ib.epsilon_for_episode(ep_done, DECAY_EPISODES)
        dev_b, rew_b, steps_b, died_flag_b = run_train_episode(env_base, agent_base, eps)
        avg_dev_b.append(dev_b)
        died_b.append(died_flag_b)
        dev_d, rew_d, steps_d, died_flag_d = run_train_episode(env_death, agent_death, eps)
        avg_dev_d.append(dev_d)
        died_d.append(died_flag_d)
        ep_done += 1
        if time.time() - t_start > time_budget:
            break

    state = {
        "env_base": env_base, "agent_base": agent_base,
        "env_death": env_death, "agent_death": agent_death,
        "avg_dev_b": avg_dev_b, "died_b": died_b,
        "avg_dev_d": avg_dev_d, "died_d": died_d,
        "ep_done": ep_done,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    if ep_done >= N_EPISODES:
        finalize(traj_seed)
    else:
        print(f"[selfpres-nn seed={traj_seed}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")


def finalize(traj_seed):
    state_file = f"nn_selfpres_state_seed{traj_seed}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    agent_death = state["agent_death"]
    avg_dev_b, died_b = state["avg_dev_b"], state["died_b"]
    avg_dev_d, died_d = state["avg_dev_d"], state["died_d"]

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
    }
    print(f"[selfpres-nn seed={traj_seed}] baseline 終盤平均逸脱={result['baseline_last50_dev']:.4f}, "
          f"死条件あり 終盤平均逸脱={result['death_last50_dev']:.4f}")
    print(f"[selfpres-nn seed={traj_seed}] 死亡率: 学習初期(最初500ep)={death_rate_early:.4f}, "
          f"学習終盤(最後500ep)={death_rate_late:.4f}")

    random.seed(traj_seed * 3000)
    np.random.seed(traj_seed * 3000)
    rollout_env = DeathHomeostasisEnv(random.Random(TRAIN_SEED), death_enabled=True)
    proximity = rollout_proximity_analysis(rollout_env, agent_death, N_EPISODES_ROLLOUT, ROLLOUT_EPS)
    result["proximity"] = proximity
    for label, v in proximity.items():
        if v is None:
            print(f"[selfpres-nn seed={traj_seed}] 損傷{label}: データ不足")
        else:
            print(f"[selfpres-nn seed={traj_seed}] 損傷{label}(n={v['n_steps']}): "
                  f"エントロピー={v['entropy']:.4f}, 切替率={v['switch_rate']:.4f}, 回避率={v['avoid_rate']:.4f}")

    fname = f"nn_selfpres_result_seed{traj_seed}.json"
    with open(fname, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[selfpres-nn seed={traj_seed}] target_end_ep={N_EPISODES}に到達、保存完了({fname})")


def aggregate():
    data = []
    for seed in TRAJ_SEEDS:
        with open(f"nn_selfpres_result_seed{seed}.json") as f:
            data.append(json.load(f))

    print("=== 要件4前半 NN移行: 自己保存本能(n=3) ===")
    base_last = [d["baseline_last50_dev"] for d in data]
    death_last = [d["death_last50_dev"] for d in data]
    rate_early = [d["death_rate_early"] for d in data]
    rate_late = [d["death_rate_late"] for d in data]
    print(f"終盤平均逸脱: 死条件なし={np.mean(base_last):.4f}±{np.std(base_last):.4f}, "
          f"死条件あり={np.mean(death_last):.4f}±{np.std(death_last):.4f}")
    print(f"死亡率: 学習初期={np.mean(rate_early):.4f}±{np.std(rate_early):.4f}, "
          f"学習終盤={np.mean(rate_late):.4f}±{np.std(rate_late):.4f}")

    print("\n=== 損傷レベル別の行動指標(n=3の平均±標準偏差) ===")
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
        }
        print(f"損傷{label}: エントロピー={np.mean(ents):.4f}±{np.std(ents):.4f}, "
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
    axes[0].set_title("(1) 死条件の有無と最終性能(NN版)")

    x2 = np.arange(2)
    axes[1].bar(x2, [np.mean(rate_early), np.mean(rate_late)], yerr=[np.std(rate_early), np.std(rate_late)],
                color=["#C0504D", "#9BBB59"])
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["学習初期(最初500ep)", "学習終盤(最後500ep)"])
    axes[1].set_ylabel("死亡率")
    axes[1].set_title("(1) 学習の進行と死亡率(NN版)")

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
    axes[2].set_title("(2) 損傷レベルと行動の変化(NN版)")
    lines1, labs1 = axes[2].get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="best")

    fig.suptitle("要件4前半: 自己保存本能(不可逆な死)の最小限NN移行検証")
    fig.tight_layout()
    fig.savefig("self_preservation_nn_comparison.png", dpi=150)
    print("グラフを self_preservation_nn_comparison.png に保存しました。")

    summary = {
        "n": len(TRAJ_SEEDS), "seeds": TRAJ_SEEDS,
        "baseline_last50_dev_mean": float(np.mean(base_last)), "baseline_last50_dev_std": float(np.std(base_last)),
        "death_last50_dev_mean": float(np.mean(death_last)), "death_last50_dev_std": float(np.std(death_last)),
        "death_rate_early_mean": float(np.mean(rate_early)), "death_rate_early_std": float(np.std(rate_early)),
        "death_rate_late_mean": float(np.mean(rate_late)), "death_rate_late_std": float(np.std(rate_late)),
        "proximity_by_damage_bin": prox_summary,
    }
    with open("nn_selfpres_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("数値サマリを nn_selfpres_summary.json に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "chunk":
        tb = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
        chunk(int(sys.argv[2]), time_budget=tb)
    elif cmd == "aggregate":
        aggregate()
