"""
感情AIプロジェクト フェーズ2 拡張プロトタイプ: 本能レベルの初期バイアス(要件3)
==========================================================

フェーズ1(homeostasis_prototype.py)で学習済みの「親」エージェントのQ値を、
親とは異なるタイル配置を持つ「子」エージェントの初期Qテーブルへ、
強度(bias_strength: 0.0〜1.0)を変えて転写し、以下2点を確認する。

  1. 本能(=学習結果からの初期バイアス)が強いほど、白紙状態(本能なし)より
     早く恒常性を保てるようになるか
  2. 環境が変わって一部の偏りが不適切になっても、経験によって上書きされ、
     最終的な性能が損なわれずに済むか(=本能が固定的な命令ではなく、
     修正可能な傾きとして機能するか)

状態表現(食料・シェルター・危険地帯への相対方向 + センサー値の離散化ビン)が
絶対座標ではなく相対的な関係で組まれているため、親のQ値は配置の異なる
別マップにもそのまま転写できる(この設計はhomeostasis_prototype.pyの
学習不良の修正時に導入したものだが、結果的に本能の転写可能性も担保している)。

注意: bias_strength=0.0(対照群)も含め、子世代の環境マップ(食料・シェルター・
危険地帯の配置)は全条件で同一に揃えている。ただし学習ループの乱数(探索行動・
気温ドリフト)は条件ごとに複数シードで繰り返し(REPEATS回)、平均と run 間の
ばらつき(標準偏差)の両方を記録することで、単発試行のノイズと本能強度の
効果を区別できるようにしている。学術的な厳密比較ではなく、「本能バイアスが
機能しうるか」を確認するための小規模プロトタイプである。
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from homeostasis_prototype import HomeostasisEnv, QLearningAgent, ACTIONS, moving_average

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

PARENT_SEED = 0
CHILD_SEED = 1                 # 親とは別の食料/シェルター/危険地帯配置(全条件・全repeatで共通)
PARENT_EPISODES = 3000
PARENT_EPS_DECAY_EPISODES = 2000
CHILD_EPISODES = 3000          # 親と同じ長さまで伸ばし、収束するかどうかを確認する
CHILD_EPS_DECAY_EPISODES = 2000

EPS_START = 1.0
EPS_END = 0.05

BIAS_STRENGTHS = [0.0, 0.25, 0.5, 1.0]  # 0.0=本能なし(対照群), 1.0=親のQ値をそのまま初期値に
REPEATS = 3                    # 学習ループの乱数を変えて繰り返し、run間のばらつきを見る
RUN_SEEDS = [10, 20, 30]       # マップ(CHILD_SEED)は固定し、探索行動・気温ドリフトのみ変える


def epsilon_for_episode(ep, decay_episodes, eps_start=EPS_START):
    frac = min(1.0, ep / decay_episodes)
    return eps_start + (EPS_END - eps_start) * frac


def train(env, agent, n_episodes, decay_episodes, eps_start=EPS_START):
    avg_dev, total_rew = [], []
    for ep in range(n_episodes):
        state = env.reset()
        eps = epsilon_for_episode(ep, decay_episodes, eps_start)
        done = False
        devs, rew_sum = [], 0.0
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
        avg_dev.append(float(np.mean(devs)))
        total_rew.append(rew_sum)
    return avg_dev, total_rew


def make_biased_agent(parent_q, strength):
    agent = QLearningAgent()
    if strength > 0.0:
        agent.q = {k: v * strength for k, v in parent_q.items()}
    return agent


if __name__ == "__main__":
    # --- 親世代の学習(フェーズ1と同一条件) ---
    random.seed(PARENT_SEED)
    np.random.seed(PARENT_SEED)
    parent_env = HomeostasisEnv(random.Random(PARENT_SEED))
    parent_agent = QLearningAgent()
    train(parent_env, parent_agent, PARENT_EPISODES, PARENT_EPS_DECAY_EPISODES)
    print(f"親エージェント 学習後Qエントリ数: {len(parent_agent.q)}")

    # --- 子世代: 親と異なる配置の環境で、本能バイアスの強さを変えて学習 ---
    # マップはCHILD_SEEDで固定し、全条件・全repeatで共通にする
    child_env = HomeostasisEnv(random.Random(CHILD_SEED))

    results = {}
    for strength in BIAS_STRENGTHS:
        # 本能が強いほど、探索(ランダム行動)に頼らず自分のQ値を初手から信頼する
        # ことにする。eps_start=1.0固定のままだと序盤は完全ランダム行動が支配的になり、
        # 転写したQ値が行動選択に反映されず、本能の効果が序盤に一切現れないという
        # 問題が最初の実行で判明したため導入した補正。
        eps_start = 1.0 - 0.7 * strength
        dev_runs, rew_runs = [], []
        for run_seed in RUN_SEEDS:
            random.seed(run_seed)
            np.random.seed(run_seed)
            child_agent = make_biased_agent(parent_agent.q, strength)
            avg_dev, total_rew = train(child_env, child_agent, CHILD_EPISODES, CHILD_EPS_DECAY_EPISODES, eps_start)
            dev_runs.append(avg_dev)
            rew_runs.append(total_rew)
        dev_runs = np.array(dev_runs)   # shape (REPEATS, CHILD_EPISODES)
        rew_runs = np.array(rew_runs)
        results[strength] = (dev_runs, rew_runs)

        first50_dev = dev_runs[:, :50].mean(axis=1)
        last50_dev = dev_runs[:, -50:].mean(axis=1)
        first50_rew = rew_runs[:, :50].mean(axis=1)
        last50_rew = rew_runs[:, -50:].mean(axis=1)
        print(
            f"本能強度={strength:.2f} (n={REPEATS}): "
            f"最初50ep平均逸脱={first50_dev.mean():.4f}±{first50_dev.std():.4f}, "
            f"最後50ep平均逸脱={last50_dev.mean():.4f}±{last50_dev.std():.4f}, "
            f"最初50ep平均報酬={first50_rew.mean():.2f}±{first50_rew.std():.2f}, "
            f"最後50ep平均報酬={last50_rew.mean():.2f}±{last50_rew.std():.2f}"
        )

    # --- 可視化(各runの移動平均を計算した上で、run間の平均±標準偏差を帯で表示) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {0.0: "#888888", 0.25: "#9BBB59", 0.5: "#4BACC6", 1.0: "#C0504D"}
    window = 100
    for strength in BIAS_STRENGTHS:
        dev_runs, rew_runs = results[strength]
        label = "本能なし(対照群)" if strength == 0.0 else f"本能強度 {strength:.2f}"

        ma_dev = np.array([moving_average(run, window=window) for run in dev_runs])
        x_dev = np.arange(ma_dev.shape[1]) + window // 2
        mean_dev, std_dev = ma_dev.mean(axis=0), ma_dev.std(axis=0)
        axes[0].plot(x_dev, mean_dev, label=label, color=colors[strength], linewidth=2)
        axes[0].fill_between(x_dev, mean_dev - std_dev, mean_dev + std_dev, color=colors[strength], alpha=0.15)

        ma_rew = np.array([moving_average(run, window=window) for run in rew_runs])
        x_rew = np.arange(ma_rew.shape[1]) + window // 2
        mean_rew, std_rew = ma_rew.mean(axis=0), ma_rew.std(axis=0)
        axes[1].plot(x_rew, mean_rew, label=label, color=colors[strength], linewidth=2)
        axes[1].fill_between(x_rew, mean_rew - std_rew, mean_rew + std_rew, color=colors[strength], alpha=0.15)

    axes[0].set_xlabel(f"エピソード(子世代・親とは異なる環境、{REPEATS}回試行の平均±標準偏差)")
    axes[0].set_ylabel(f"平均逸脱(移動平均{window}ep)")
    axes[0].set_title("本能強度別: 恒常性からの逸脱")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel(f"エピソード(子世代・親とは異なる環境、{REPEATS}回試行の平均±標準偏差)")
    axes[1].set_ylabel(f"累計報酬(移動平均{window}ep)")
    axes[1].set_title("本能強度別: 報酬の推移")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"要件3プロトタイプ: 本能レベルの初期バイアス({CHILD_EPISODES}ep, {REPEATS}回試行の平均)")
    fig.tight_layout()
    fig.savefig("instinct_bias_comparison.png", dpi=150)
    print("グラフを instinct_bias_comparison.png に保存しました。")
