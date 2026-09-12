"""
感情AIプロジェクト フェーズ2〜3 プロトタイプ: 要件7 U字型の追加切り分け
==========================================================
(条件付きエントロピー/相互情報量による切り分け + 行動空間の大きさの影響)

前回(monitor_policy_complexity_prototype.py)、行動エントロピー(行動の周辺分布の
多様性)がモニタのU字型と逆位相の山型を示すことを確認した。これに対し2つの
追加の問いを検証する。

  (A) 条件付きエントロピー/相互情報量: 「同じ行動が異なる欲求の場面で使い回されて
      いる」という仮説を、周辺エントロピーより直接的な指標で確認する。行動Aと
      支配的な逸脱の種類Yの相互情報量 I(A;Y) = H(Y) - H(Y|A) を計算する。I(A;Y)が
      高いほど「行動を見ればどの欲求が支配的か分かる」ことを意味し、モニタの
      精度(held-out相関)と同じU字型(150epと3000epで高く、500〜1500epで低い)を
      示すはずだという予測を検証する。
  (B) 行動空間の大きさの影響: 行動が5種類(上下左右+stay)しかないため、周辺
      エントロピーの理論上限はlog2(5)≈2.32bitであり、実測値(2.07〜2.26)は
      すでに上限の89〜97%に達していた。これが「行動の種類が少ないこと」自体に
      由来する天井効果なのか、環境の構造(欲求の複雑さ)に由来するのかを切り分ける
      ため、斜め移動4種を加えた9行動版の環境で同じ実験を繰り返し、行動空間が
      増えても同様のU字型・エントロピー相関が再現するかを確認する。

コマンドライン引数で条件を選択する: "baseline"(5行動、デフォルト) または
"extended"(9行動、斜め移動を追加)。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import homeostasis_prototype as hp
from homeostasis_prototype import HomeostasisEnv, QLearningAgent
import instinct_bias_prototype as ib
from monitor_maturity_prototype import (
    train_with_checkpoints, collect_rollout, fit_linear_regression, predict_linear, mean_correlation,
)

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAIN_SEED = 0
TRAJ_SEEDS = [0, 11, 22]
CHECKPOINT_EPISODES = [150, 500, 1500, 3000]
ROLLOUT_EPS = 0.1
N_EPISODES_TRAIN_MAP = 100

BASE_ACTIONS = ["up", "down", "left", "right", "stay"]
EXT_ACTIONS = ["up", "down", "left", "right", "up_left", "up_right", "down_left", "down_right", "stay"]
MOVES = {
    "stay": (0, 0),
    "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0),
    "up_left": (-1, -1), "up_right": (1, -1), "down_left": (-1, 1), "down_right": (1, 1),
}


class ExtendedHomeostasisEnv(HomeostasisEnv):
    """斜め移動4種を加えた9行動版(hp.ACTIONSがEXT_ACTIONSに差し替えられている前提)。"""

    def step(self, action):
        x, y = self.pos
        dx, dy = MOVES[action]
        x = int(np.clip(x + dx, 0, hp.GRID_SIZE - 1))
        y = int(np.clip(y + dy, 0, hp.GRID_SIZE - 1))
        self.pos = (x, y)

        self.energy -= hp.ENERGY_DECAY_PER_STEP
        self.temperature += self.rng2_normal() * hp.TEMP_DRIFT_STD
        self.damage = max(0.0, self.damage - hp.DAMAGE_HEAL_PER_STEP)

        if self.pos in self.food_tiles:
            self.energy += 40.0
        if self.pos in self.shelter_tiles:
            self.temperature += (hp.OPTIMAL_TEMP - self.temperature) * 0.5
        if self.pos in self.hazard_tiles:
            self.damage += 30.0

        self.energy = float(np.clip(self.energy, 0.0, 100.0))
        self.damage = float(np.clip(self.damage, 0.0, 100.0))

        dev_energy = abs(self.energy - hp.OPTIMAL_ENERGY) / 100.0
        dev_temp = abs(self.temperature - hp.OPTIMAL_TEMP) / 30.0
        dev_damage = abs(self.damage - hp.OPTIMAL_DAMAGE) / 100.0
        deviation = dev_energy + dev_temp + dev_damage
        reward = -deviation

        self.t += 1
        done = self.t >= hp.MAX_STEPS or self.energy <= 0.0
        return self.discrete_state(), reward, done, deviation


ACTIONS_3D = ["x+", "x-", "y+", "y-", "z+", "z-", "stay"]
MOVES_3D = {
    "stay": (0, 0, 0),
    "x+": (1, 0, 0), "x-": (-1, 0, 0),
    "y+": (0, 1, 0), "y-": (0, -1, 0),
    "z+": (0, 0, 1), "z-": (0, 0, -1),
}


def random_tiles_3d(n, rng):
    return set((rng.randint(0, hp.GRID_SIZE - 1), rng.randint(0, hp.GRID_SIZE - 1),
                rng.randint(0, hp.GRID_SIZE - 1)) for _ in range(n))


class Homeostasis3DEnv:
    """行動空間を三次元(x,y,z)に拡張したグリッドワールド。2D版と同じ恒常性モデル・
    センサー3種を、立体空間上に配置したfood/shelter/hazardタイルに対して適用する。
    行動は6方向(±x,±y,±z)+stayの7種類。状態表現も各資源への相対方向を
    (dx,dy,dz)の3成分に拡張しており、2D版(dx,dy)より状態空間が本質的に広い。"""

    def __init__(self, rng):
        self.rng = rng
        self.food_tiles = random_tiles_3d(3, rng)
        self.shelter_tiles = random_tiles_3d(3, rng)
        self.hazard_tiles = random_tiles_3d(4, rng)
        self.reset()

    def reset(self):
        c = hp.GRID_SIZE // 2
        self.pos = (c, c, c)
        self.energy = 100.0
        self.temperature = hp.OPTIMAL_TEMP
        self.damage = 0.0
        self.t = 0
        return self.discrete_state()

    def _nearest_dir(self, tiles):
        if not tiles:
            return (0, 0, 0)
        x, y, z = self.pos
        best = min(tiles, key=lambda t: abs(t[0] - x) + abs(t[1] - y) + abs(t[2] - z))
        return (int(np.sign(best[0] - x)), int(np.sign(best[1] - y)), int(np.sign(best[2] - z)))

    def discrete_state(self):
        e_bin = int(np.clip(self.energy // 20, 0, 5))
        t_bin = int(np.clip((self.temperature - hp.OPTIMAL_TEMP + 15) // 6, 0, 5))
        d_bin = int(np.clip(self.damage // 20, 0, 5))
        return (self._nearest_dir(self.food_tiles), self._nearest_dir(self.shelter_tiles),
                self._nearest_dir(self.hazard_tiles), e_bin, t_bin, d_bin)

    def rng2_normal(self):
        return np.random.randn()

    def step(self, action):
        x, y, z = self.pos
        dx, dy, dz = MOVES_3D[action]
        x = int(np.clip(x + dx, 0, hp.GRID_SIZE - 1))
        y = int(np.clip(y + dy, 0, hp.GRID_SIZE - 1))
        z = int(np.clip(z + dz, 0, hp.GRID_SIZE - 1))
        self.pos = (x, y, z)

        self.energy -= hp.ENERGY_DECAY_PER_STEP
        self.temperature += self.rng2_normal() * hp.TEMP_DRIFT_STD
        self.damage = max(0.0, self.damage - hp.DAMAGE_HEAL_PER_STEP)

        if self.pos in self.food_tiles:
            self.energy += 40.0
        if self.pos in self.shelter_tiles:
            self.temperature += (hp.OPTIMAL_TEMP - self.temperature) * 0.5
        if self.pos in self.hazard_tiles:
            self.damage += 30.0

        self.energy = float(np.clip(self.energy, 0.0, 100.0))
        self.damage = float(np.clip(self.damage, 0.0, 100.0))

        dev_energy = abs(self.energy - hp.OPTIMAL_ENERGY) / 100.0
        dev_temp = abs(self.temperature - hp.OPTIMAL_TEMP) / 30.0
        dev_damage = abs(self.damage - hp.OPTIMAL_DAMAGE) / 100.0
        deviation = dev_energy + dev_temp + dev_damage
        reward = -deviation

        self.t += 1
        done = self.t >= hp.MAX_STEPS or self.energy <= 0.0
        return self.discrete_state(), reward, done, deviation


def shannon_entropy(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_information(action_idx, y_class, n_actions, n_classes=3):
    n = len(y_class)
    joint = np.zeros((n_actions, n_classes))
    for a, c in zip(action_idx, y_class):
        joint[a, c] += 1
    joint /= n
    pa = joint.sum(axis=1)
    pc = joint.sum(axis=0)
    h_y = shannon_entropy(pc * n)  # pcはすでに正規化済みなのでcounts代わりにそのまま渡してよい
    mi = 0.0
    for a in range(n_actions):
        for c in range(n_classes):
            if joint[a, c] > 0 and pa[a] > 0 and pc[c] > 0:
                mi += joint[a, c] * np.log2(joint[a, c] / (pa[a] * pc[c]))
    h_y_given_a = h_y - mi
    return float(mi), float(h_y), float(h_y_given_a)


def run_condition(condition):
    if condition == "extended":
        hp.ACTIONS[:] = EXT_ACTIONS
        env_cls = ExtendedHomeostasisEnv
    elif condition == "3d":
        hp.ACTIONS[:] = ACTIONS_3D
        env_cls = Homeostasis3DEnv
    else:
        hp.ACTIONS[:] = BASE_ACTIONS
        env_cls = HomeostasisEnv
    n_actions = len(hp.ACTIONS)
    max_entropy = np.log2(n_actions)

    train_env = env_cls(random.Random(TRAIN_SEED))
    per_seed_records = []

    for traj_seed in TRAJ_SEEDS:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        agent = QLearningAgent()
        checkpoints = train_with_checkpoints(
            train_env, agent, max(CHECKPOINT_EPISODES), ib.PARENT_EPS_DECAY_EPISODES, CHECKPOINT_EPISODES
        )

        seed_records = []
        for n_ep in CHECKPOINT_EPISODES:
            q_table = checkpoints[n_ep]
            random.seed(traj_seed * 1000 + n_ep)
            np.random.seed(traj_seed * 1000 + n_ep)
            map_env = env_cls(random.Random(TRAIN_SEED))
            X_all, yc_all, ycont_all = collect_rollout(map_env, dict(q_table), N_EPISODES_TRAIN_MAP, ROLLOUT_EPS)

            action_idx = np.argmax(X_all[:, :n_actions], axis=1)
            counts = np.bincount(action_idx, minlength=n_actions).astype(float)
            ent = shannon_entropy(counts)
            mi, h_y, h_y_given_a = mutual_information(action_idx, yc_all, n_actions)

            n = len(X_all)
            idx = np.random.permutation(n)
            split = int(n * 0.7)
            X_tr, ycont_tr = X_all[idx[:split]], ycont_all[idx[:split]]
            X_te, ycont_te = X_all[idx[split:]], ycont_all[idx[split:]]
            W = fit_linear_regression(X_tr, ycont_tr)
            pred_te = predict_linear(X_te, W)
            corr_te = mean_correlation(ycont_te, pred_te)

            seed_records.append({
                "n_episodes": n_ep, "corr_holdout": corr_te,
                "entropy": ent, "entropy_norm": ent / max_entropy,
                "mi": mi, "h_y": h_y, "h_y_given_a": h_y_given_a,
            })
        per_seed_records.append(seed_records)
        print(f"  [{condition}] traj_seed={traj_seed}: " + " / ".join(
            f"{r['n_episodes']}ep:相関={r['corr_holdout']:.3f},MI={r['mi']:.3f},H(Y|A)={r['h_y_given_a']:.3f},エントロピー={r['entropy']:.3f}({r['entropy_norm']:.2%})"
            for r in seed_records
        ))

    summary = []
    for i, n_ep in enumerate(CHECKPOINT_EPISODES):
        keys = ["corr_holdout", "entropy", "entropy_norm", "mi", "h_y_given_a"]
        vals = {k: [rec[i][k] for rec in per_seed_records] for k in keys}
        summary.append({"n_episodes": n_ep,
                         **{f"{k}_mean": float(np.mean(v)) for k, v in vals.items()},
                         **{f"{k}_std": float(np.std(v)) for k, v in vals.items()}})
    return summary, n_actions, max_entropy


if __name__ == "__main__":
    condition = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    print(f"=== 条件: {condition} ===")
    summary, n_actions, max_entropy = run_condition(condition)

    print(f"\n行動数={n_actions}, 理論上限エントロピー={max_entropy:.3f}bit")
    print("=== 集計(n=3の平均±標準偏差) ===")
    for s in summary:
        print(
            f"{s['n_episodes']}ep: held-out相関={s['corr_holdout_mean']:.4f}±{s['corr_holdout_std']:.4f}, "
            f"相互情報量I(A;Y)={s['mi_mean']:.4f}±{s['mi_std']:.4f}bit, "
            f"H(Y|A)={s['h_y_given_a_mean']:.4f}±{s['h_y_given_a_std']:.4f}bit, "
            f"エントロピー={s['entropy_mean']:.4f}±{s['entropy_std']:.4f}bit"
            f"(上限比{s['entropy_norm_mean']:.1%}±{s['entropy_norm_std']:.1%})"
        )

    corr_vals = np.array([s["corr_holdout_mean"] for s in summary])
    mi_vals = np.array([s["mi_mean"] for s in summary])
    ent_vals = np.array([s["entropy_mean"] for s in summary])
    print(f"\nチェックポイント平均どうしの相関(n=4点): MIとheld-out相関 r={np.corrcoef(mi_vals, corr_vals)[0,1]:.3f}, "
          f"エントロピーとheld-out相関 r={np.corrcoef(ent_vals, corr_vals)[0,1]:.3f}")

    ns = [s["n_episodes"] for s in summary]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].errorbar(ns, [s["corr_holdout_mean"] for s in summary], yerr=[s["corr_holdout_std"] for s in summary],
                      marker="o", color="#4472C4")
    axes[0].set_title("モニタのheld-out相関")
    axes[0].set_xlabel("学習量(episode数)")

    axes[1].errorbar(ns, [s["mi_mean"] for s in summary], yerr=[s["mi_std"] for s in summary],
                      marker="o", color="#C0504D")
    axes[1].set_title("行動と支配的逸脱の相互情報量 I(A;Y)")
    axes[1].set_xlabel("学習量(episode数)")
    axes[1].set_ylabel("bit")

    axes[2].errorbar(ns, [s["entropy_norm_mean"] * 100 for s in summary], yerr=[s["entropy_norm_std"] * 100 for s in summary],
                      marker="o", color="#9BBB59")
    axes[2].set_title(f"行動エントロピー(理論上限比、行動数={n_actions})")
    axes[2].set_xlabel("学習量(episode数)")
    axes[2].set_ylabel("%")

    fig.suptitle(f"要件7追加切り分け: 条件={condition}(行動数={n_actions})")
    fig.tight_layout()
    fig.savefig(f"monitor_action_diversity_{condition}.png", dpi=150)
    print(f"グラフを monitor_action_diversity_{condition}.png に保存しました。")
