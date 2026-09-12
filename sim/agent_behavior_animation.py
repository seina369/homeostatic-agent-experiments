"""
感情AIプロジェクト: エージェントの実際の振る舞いをアニメーション(GIF)で可視化する。

学習段階の異なる3つのチェックポイント(150ep=学習初期, 1500ep=学習中盤,
3000ep=学習成熟後)のエージェントを、同じマップ・同じロールアウト条件で
1エピソードずつ動かし、実際の移動軌跡とセンサー値の推移を並べて表示する。
これまでの実験(monitor_maturity_prototype.py, monitor_policy_complexity_prototype.py)
で「学習中盤に行動が多様化し、終盤に少数のパターンへ再収束する」という結果が
出ていたが、これを数値ではなく実際の動きとして目で見えるようにする。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm

import homeostasis_prototype as hp
from homeostasis_prototype import HomeostasisEnv, QLearningAgent, ACTIONS
import instinct_bias_prototype as ib
from monitor_maturity_prototype import train_with_checkpoints

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAIN_SEED = 0
CHECKPOINTS = [150, 1500, 3000]
DEMO_SEED = 777
ROLLOUT_EPS = 0.1
GRID = hp.GRID_SIZE

TITLES = {150: "学習初期(150ep)", 1500: "学習中盤(1500ep)", 3000: "学習成熟後(3000ep)"}


def rollout_episode(env_template, q_table, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = HomeostasisEnv(random.Random(TRAIN_SEED))  # マップは共通のTRAIN_SEEDで再構築
    agent = QLearningAgent()
    agent.q = q_table
    state = env.reset()
    frames = []
    done = False
    while not done:
        if random.random() < ROLLOUT_EPS:
            action = random.choice(ACTIONS)
        else:
            action = agent.best_action(state)
        frames.append({
            "pos": env.pos, "energy": env.energy, "temperature": env.temperature,
            "damage": env.damage, "action": action, "t": env.t,
        })
        next_state, reward, done, deviation = env.step(action)
        state = next_state
    return frames, env


if __name__ == "__main__":
    train_env = HomeostasisEnv(random.Random(TRAIN_SEED))
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    agent = QLearningAgent()
    checkpoints_q = train_with_checkpoints(
        train_env, agent, max(CHECKPOINTS), ib.PARENT_EPS_DECAY_EPISODES, CHECKPOINTS
    )
    print("学習完了。チェックポイントQエントリ数:", {k: len(v) for k, v in checkpoints_q.items()})

    episodes = {}
    env_ref = None
    for n_ep in CHECKPOINTS:
        frames, env = rollout_episode(train_env, checkpoints_q[n_ep], seed=DEMO_SEED)
        episodes[n_ep] = frames
        env_ref = env
        print(f"{n_ep}ep: エピソード長={len(frames)}ステップ, 最終エネルギー={frames[-1]['energy']:.1f}")

    max_len = max(len(f) for f in episodes.values())

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2))

    def setup_ax(ax, title):
        ax.set_xlim(-0.5, GRID - 0.5)
        ax.set_ylim(-0.5, GRID - 0.5)
        ax.set_xticks(range(GRID))
        ax.set_yticks(range(GRID))
        ax.grid(True, color="#eeeeee")
        ax.set_title(title, fontsize=13)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        for (fx, fy) in env_ref.food_tiles:
            ax.add_patch(plt.Rectangle((fx - 0.5, fy - 0.5), 1, 1, color="#9BBB59", alpha=0.55))
        for (sx, sy) in env_ref.shelter_tiles:
            ax.add_patch(plt.Rectangle((sx - 0.5, sy - 0.5), 1, 1, color="#4472C4", alpha=0.55))
        for (hx, hy) in env_ref.hazard_tiles:
            ax.add_patch(plt.Rectangle((hx - 0.5, hy - 0.5), 1, 1, color="#C0504D", alpha=0.55))

    agent_dots, trail_lines, info_texts = {}, {}, {}
    for ax, n_ep in zip(axes, CHECKPOINTS):
        setup_ax(ax, TITLES[n_ep])
        dot, = ax.plot([], [], "o", color="black", markersize=15, zorder=5)
        trail, = ax.plot([], [], "-", color="black", alpha=0.35, linewidth=1.5, zorder=4)
        txt = ax.text(0.0, -0.16, "", transform=ax.transAxes, fontsize=9.5, va="top")
        agent_dots[n_ep] = dot
        trail_lines[n_ep] = trail
        info_texts[n_ep] = txt

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#9BBB59", alpha=0.55, label="食料(エネルギー回復)"),
        plt.Rectangle((0, 0), 1, 1, color="#4472C4", alpha=0.55, label="シェルター(体温調整)"),
        plt.Rectangle((0, 0), 1, 1, color="#C0504D", alpha=0.55, label="危険地帯(損傷)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=10, frameon=False)

    def update(frame_idx):
        artists = []
        for n_ep in CHECKPOINTS:
            frames = episodes[n_ep]
            idx = min(frame_idx, len(frames) - 1)
            f = frames[idx]
            x, y = f["pos"]
            agent_dots[n_ep].set_data([x], [y])
            xs = [ff["pos"][0] for ff in frames[: idx + 1]]
            ys = [ff["pos"][1] for ff in frames[: idx + 1]]
            trail_lines[n_ep].set_data(xs, ys)
            ended = frame_idx >= len(frames) - 1 and f["energy"] <= 0.0
            status = "  [力尽きた]" if ended else ""
            info_texts[n_ep].set_text(
                f"step {f['t']}  行動: {f['action']}{status}\n"
                f"エネルギー {f['energy']:.0f} / 体温 {f['temperature']:.1f} / 損傷 {f['damage']:.0f}"
            )
            artists += [agent_dots[n_ep], trail_lines[n_ep], info_texts[n_ep]]
        return artists

    fig.suptitle("感情AIプロトタイプ: 学習段階別のエージェントの振る舞い(homeostasis_prototype.py, 同一マップ)", y=1.09, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    anim = animation.FuncAnimation(fig, update, frames=max_len, interval=140, blit=False)
    anim.save("agent_behavior_comparison.gif", writer="pillow", fps=7)
    print("saved agent_behavior_comparison.gif")
