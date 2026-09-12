"""
感情AIプロジェクト フェーズ2〜3 プロトタイプ: 高階自己モニタリング層(要件7)の深掘り
==========================================================

monitor_prototype.py(単一の成熟した方策・単一の未経験マップでの検証)を、
3方向で深掘りする。

  (1) 成熟度の検出: エージェントの学習が浅い段階と十分学習した段階とで、
      モニタの精度・相関がどう変わるかを追う。計画書は「高階モニタリング層が
      安定稼働する段階に達した個体」に自己改訂権を与えるとしており、この
      「安定稼働」を数値的に検出できるかを試す。
  (2) 時間的安定性: 精度(離散クラスの正解率)だけでなく、モニタが予測する
      連続値(センサーごとの逸脱量)が、真の値となだらかに連動して変化するか
      (ノイズ的に暴れないか)を時系列で確認する。
  (3) 汎化性能の厳密化: 前回は未経験マップが1つだけで、たまたま「損傷」が
      一度も支配的にならず実質2クラス評価になっていた。複数の未経験マップで
      平均精度・ばらつきを見る。

設計変更: 前回は3クラス分類のロジスティック回帰だったが、今回はセンサーごとの
逸脱量(連続値)を予測するリッジ回帰に切り替える。予測値のargmaxを取れば従来の
分類精度も得られ、かつ「センサー入力と相関する」を文字通り相関係数で測定できる。
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from homeostasis_prototype import HomeostasisEnv, QLearningAgent, ACTIONS
import instinct_bias_prototype as ib

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

OPTIMAL_ENERGY = 100.0
OPTIMAL_TEMP = 20.0
OPTIMAL_DAMAGE = 0.0

TRAIN_SEED = 0
UNSEEN_SEEDS = [5, 6, 7]                       # 汎化性能の厳密化: 複数の未経験マップ
CHECKPOINT_EPISODES = [150, 500, 1500, 3000]   # 成熟度の各段階
ROLLOUT_EPS = 0.1
N_EPISODES_TRAIN_MAP = 100
N_EPISODES_UNSEEN_MAP = 40
HISTORY_LEN = 2


def deviations(energy, temperature, damage):
    dev_energy = abs(energy - OPTIMAL_ENERGY) / 100.0
    dev_temp = abs(temperature - OPTIMAL_TEMP) / 30.0
    dev_damage = abs(damage - OPTIMAL_DAMAGE) / 100.0
    return np.array([dev_energy, dev_temp, dev_damage])


def action_one_hot(action):
    v = np.zeros(len(ACTIONS))
    v[ACTIONS.index(action)] = 1.0
    return v


def epsilon_for_episode(ep, decay_episodes, eps_start=1.0, eps_end=0.05):
    frac = min(1.0, ep / decay_episodes)
    return eps_start + (eps_end - eps_start) * frac


def train_with_checkpoints(env, agent, n_episodes, decay_episodes, checkpoint_eps):
    checkpoints = {}
    checkpoint_set = set(checkpoint_eps)
    for ep in range(n_episodes):
        state = env.reset()
        eps = epsilon_for_episode(ep, decay_episodes)
        done = False
        while not done:
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = agent.best_action(state)
            next_state, reward, done, deviation = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        if (ep + 1) in checkpoint_set:
            checkpoints[ep + 1] = dict(agent.q)
    return checkpoints


def collect_rollout(env, q_table, n_episodes, eps):
    X, y_class, y_cont = [], [], []
    agent = QLearningAgent()
    agent.q = q_table
    for ep in range(n_episodes):
        state = env.reset()
        history = ["stay"] * HISTORY_LEN
        done = False
        while not done:
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = agent.best_action(state)

            q_values = [agent.q_value(state, a) for a in ACTIONS]
            sorted_q = sorted(q_values, reverse=True)
            chosen_q = agent.q_value(state, action)
            q_gap = sorted_q[0] - (sorted_q[1] if len(sorted_q) > 1 else sorted_q[0])

            feat = np.concatenate([
                action_one_hot(action),
                [chosen_q, q_gap],
                np.concatenate([action_one_hot(a) for a in history]),
            ])

            next_state, reward, done, deviation = env.step(action)
            dev_vec = deviations(env.energy, env.temperature, env.damage)

            X.append(feat)
            y_class.append(int(np.argmax(dev_vec)))
            y_cont.append(dev_vec)

            history = history[1:] + [action]
            state = next_state
    return np.array(X), np.array(y_class), np.array(y_cont)


def fit_linear_regression(X, Y, l2=1e-3):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    d = Xb.shape[1]
    A = Xb.T @ Xb + l2 * np.eye(d)
    B = Xb.T @ Y
    return np.linalg.solve(A, B)


def predict_linear(X, W):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return Xb @ W


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def majority_baseline(y_train, y_eval):
    majority = np.bincount(y_train).argmax()
    return accuracy(y_eval, np.full_like(y_eval, majority))


def mean_correlation(Y_true, Y_pred):
    rs = []
    for i in range(Y_true.shape[1]):
        if np.std(Y_true[:, i]) < 1e-8 or np.std(Y_pred[:, i]) < 1e-8:
            continue
        r = np.corrcoef(Y_true[:, i], Y_pred[:, i])[0, 1]
        rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


if __name__ == "__main__":
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    train_env = HomeostasisEnv(random.Random(TRAIN_SEED))
    agent = QLearningAgent()
    checkpoints = train_with_checkpoints(
        train_env, agent, max(CHECKPOINT_EPISODES), ib.PARENT_EPS_DECAY_EPISODES, CHECKPOINT_EPISODES
    )
    print("チェックポイントのQエントリ数:", {k: len(v) for k, v in checkpoints.items()})

    maturity_report = []
    final_holdout_ts = None

    for n_ep in CHECKPOINT_EPISODES:
        q_table = checkpoints[n_ep]

        random.seed(TRAIN_SEED + n_ep)
        np.random.seed(TRAIN_SEED + n_ep)
        map_env = HomeostasisEnv(random.Random(TRAIN_SEED))
        X_all, yc_all, ycont_all = collect_rollout(map_env, dict(q_table), N_EPISODES_TRAIN_MAP, ROLLOUT_EPS)
        n = len(X_all)
        idx = np.random.permutation(n)
        split = int(n * 0.7)
        X_tr, yc_tr, ycont_tr = X_all[idx[:split]], yc_all[idx[:split]], ycont_all[idx[:split]]
        X_te, yc_te, ycont_te = X_all[idx[split:]], yc_all[idx[split:]], ycont_all[idx[split:]]

        W = fit_linear_regression(X_tr, ycont_tr)
        pred_te = predict_linear(X_te, W)
        pred_class_te = np.argmax(pred_te, axis=1)
        acc_te = accuracy(yc_te, pred_class_te)
        base_te = majority_baseline(yc_tr, yc_te)
        corr_te = mean_correlation(ycont_te, pred_te)

        gen_accs, gen_corrs = [], []
        for seed in UNSEEN_SEEDS:
            random.seed(seed + n_ep)
            np.random.seed(seed + n_ep)
            unseen_env = HomeostasisEnv(random.Random(seed))
            X_u, yc_u, ycont_u = collect_rollout(unseen_env, dict(q_table), N_EPISODES_UNSEEN_MAP, ROLLOUT_EPS)
            pred_u = predict_linear(X_u, W)
            pred_class_u = np.argmax(pred_u, axis=1)
            gen_accs.append(accuracy(yc_u, pred_class_u))
            gen_corrs.append(mean_correlation(ycont_u, pred_u))

        maturity_report.append({
            "n_episodes": n_ep,
            "acc_holdout": acc_te,
            "base_holdout": base_te,
            "corr_holdout": corr_te,
            "acc_unseen_mean": float(np.mean(gen_accs)),
            "acc_unseen_std": float(np.std(gen_accs)),
            "corr_unseen_mean": float(np.mean(gen_corrs)),
            "corr_unseen_std": float(np.std(gen_corrs)),
        })
        print(
            f"[{n_ep}episode学習時点] held-out精度={acc_te:.4f}(baseline {base_te:.4f}), "
            f"held-out相関={corr_te:.4f} / "
            f"未経験マップ精度={np.mean(gen_accs):.4f}±{np.std(gen_accs):.4f}, "
            f"未経験マップ相関={np.mean(gen_corrs):.4f}±{np.std(gen_corrs):.4f}"
        )

        if n_ep == max(CHECKPOINT_EPISODES):
            random.seed(999)
            np.random.seed(999)
            ts_env = HomeostasisEnv(random.Random(TRAIN_SEED))
            X_ts, yc_ts, ycont_ts = collect_rollout(ts_env, dict(q_table), 1, 0.0)
            pred_ts = predict_linear(X_ts, W)
            final_holdout_ts = (ycont_ts, pred_ts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ns = [r["n_episodes"] for r in maturity_report]
    axes[0].plot(ns, [r["acc_holdout"] for r in maturity_report], "o-", label="学習マップ(held-out)精度", color="#4472C4")
    axes[0].plot(ns, [r["acc_unseen_mean"] for r in maturity_report], "o-", label="未経験マップ平均精度", color="#C0504D")
    axes[0].fill_between(
        ns,
        [r["acc_unseen_mean"] - r["acc_unseen_std"] for r in maturity_report],
        [r["acc_unseen_mean"] + r["acc_unseen_std"] for r in maturity_report],
        color="#C0504D", alpha=0.15,
    )
    axes[0].axhline(1 / 3, color="gray", linestyle="--", label="チャンスレート")
    axes[0].set_xlabel("エージェントの学習量(episode数)")
    axes[0].set_ylabel("モニタの精度")
    axes[0].set_title("(1) 成熟度によるモニタ精度の変化")
    axes[0].legend(fontsize=8)

    axes[1].plot(ns, [r["corr_holdout"] for r in maturity_report], "o-", label="学習マップ(held-out)相関", color="#4472C4")
    axes[1].plot(ns, [r["corr_unseen_mean"] for r in maturity_report], "o-", label="未経験マップ平均相関", color="#C0504D")
    axes[1].fill_between(
        ns,
        [r["corr_unseen_mean"] - r["corr_unseen_std"] for r in maturity_report],
        [r["corr_unseen_mean"] + r["corr_unseen_std"] for r in maturity_report],
        color="#C0504D", alpha=0.15,
    )
    axes[1].set_xlabel("エージェントの学習量(episode数)")
    axes[1].set_ylabel("真の逸脱量と予測値の相関係数(平均)")
    axes[1].set_title("(1) 成熟度によるモニタ相関の変化")
    axes[1].legend(fontsize=8)

    fig.suptitle("要件7深掘り: 成熟度とモニタの信頼性")
    fig.tight_layout()
    fig.savefig("monitor_maturity.png", dpi=150)
    print("グラフを monitor_maturity.png に保存しました。")

    ycont_ts, pred_ts = final_holdout_ts
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    sensor_names = ["エネルギー逸脱", "体温逸脱", "損傷逸脱"]
    for i, name in enumerate(sensor_names):
        axes2[i].plot(ycont_ts[:, i], label="真の値", color="#4472C4", linewidth=2)
        axes2[i].plot(pred_ts[:, i], label="モニタの推定", color="#C0504D", linewidth=1.5, linestyle="--")
        axes2[i].set_title(name)
        axes2[i].set_xlabel("ステップ")
        axes2[i].legend(fontsize=8)
    fig2.suptitle("要件7深掘り: 時間的安定性(1エピソードの推移、最も成熟した段階)")
    fig2.tight_layout()
    fig2.savefig("monitor_stability_timeseries.png", dpi=150)
    print("グラフを monitor_stability_timeseries.png に保存しました。")
