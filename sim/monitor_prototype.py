"""
感情AIプロジェクト フェーズ2〜3 プロトタイプ: 高階自己モニタリング層(要件7)
==========================================================

計画書の成功・失敗基準は、要件7(高階自己モニタリング層)が「センサー入力と
安定的に相関する内部状態を生成できるか」に懸かっている。本プロトタイプは、
この基準を測定可能な形に落とし込んだ最小検証である。

設計:
  - フェーズ1の親と同一条件で学習済みのエージェントの方策をほぼ固定し
    (探索率0.1のみ残す)、環境をロールアウトしてデータを集める。
  - 「一次過程」はエージェントの行動選択そのもの(どの行動を選んだか、
    その行動のQ値、次点行動とのQ値差、直近2手の行動)であり、センサーの
    生の値そのものではない。
  - 「高階モニタリング層」は、センサーの生の値を一切見ることなく、一次過程の
    振る舞いだけから、今どのセンサーの逸脱が支配的か
    (エネルギー/体温/損傷のどれが最も最適値から外れているか、3クラス)を
    推定するよう学習したロジスティック回帰(scikit-learn不使用、numpyで自作)
    である。
  - 学習に使ったマップ(エルダーの生育環境)とは別のマップでも同様の精度が
    出るかを確認し、特定マップの丸暗記ではなく、一次過程の振る舞いパターン
    から一般的に推定できているかを検証する。
  - この精度がチャンスレート(3クラスなら理論上約33%、実際のクラス頻度に
    基づく多数派ベースライン)を大きく上回れば、要件7の成功基準に沿う結果と
    言える。逆にチャンスレート付近にとどまれば、失敗基準に該当する可能性を
    示す。
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

TRAIN_SEED = 0     # エルダーの生育環境(方策の学習にも使った同じマップ)
TEST_SEED = 5      # 方策が経験したことのない、別のマップ
ROLLOUT_EPS = 0.1  # 完全グリーディだと行動が単調になりすぎるため、少しだけ探索を残す
N_EPISODES_TRAIN_MAP = 200
N_EPISODES_TEST_MAP = 100
HISTORY_LEN = 2    # モニタが参照する直近の行動履歴の長さ


def dominant_deviation(energy, temperature, damage):
    dev_energy = abs(energy - OPTIMAL_ENERGY) / 100.0
    dev_temp = abs(temperature - OPTIMAL_TEMP) / 30.0
    dev_damage = abs(damage - OPTIMAL_DAMAGE) / 100.0
    devs = [dev_energy, dev_temp, dev_damage]
    return int(np.argmax(devs))  # 0=エネルギー, 1=体温, 2=損傷


def action_one_hot(action):
    v = np.zeros(len(ACTIONS))
    v[ACTIONS.index(action)] = 1.0
    return v


def collect_rollout(env, agent, n_episodes, eps):
    """一次過程(行動選択)の特徴量Xと、正解ラベルy(支配的な逸脱の種類)を集める。"""
    X, y = [], []
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
            label = dominant_deviation(env.energy, env.temperature, env.damage)

            X.append(feat)
            y.append(label)

            history = history[1:] + [action]
            state = next_state
    return np.array(X), np.array(y)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_logreg(X, y, n_classes, lr=0.5, epochs=500, l2=1e-4):
    n, d = X.shape
    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y]
    for _ in range(epochs):
        logits = X @ W + b
        P = softmax(logits)
        grad_logits = (P - Y) / n
        grad_W = X.T @ grad_logits + l2 * W
        grad_b = grad_logits.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b


def predict(X, W, b):
    logits = X @ W + b
    return np.argmax(logits, axis=1)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def majority_baseline(y_train, y_eval):
    majority = np.bincount(y_train).argmax()
    return accuracy(y_eval, np.full_like(y_eval, majority))


if __name__ == "__main__":
    random.seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    train_env = HomeostasisEnv(random.Random(TRAIN_SEED))
    agent = QLearningAgent()
    ib.train(train_env, agent, ib.PARENT_EPISODES, ib.PARENT_EPS_DECAY_EPISODES)
    print(f"方策(エージェント)のQエントリ数: {len(agent.q)}")

    # --- データ収集: 学習に使ったマップ ---
    random.seed(TRAIN_SEED + 1)
    np.random.seed(TRAIN_SEED + 1)
    train_map_env = HomeostasisEnv(random.Random(TRAIN_SEED))
    X_all, y_all = collect_rollout(train_map_env, agent, N_EPISODES_TRAIN_MAP, ROLLOUT_EPS)
    n = len(X_all)
    split = int(n * 0.7)
    idx = np.random.permutation(n)
    X_tr, y_tr = X_all[idx[:split]], y_all[idx[:split]]
    X_te, y_te = X_all[idx[split:]], y_all[idx[split:]]
    print(f"学習マップでのサンプル数: {n} (train={len(X_tr)}, held-out test={len(X_te)})")
    print(f"クラス分布(全体): エネルギー={np.mean(y_all==0):.3f}, 体温={np.mean(y_all==1):.3f}, 損傷={np.mean(y_all==2):.3f}")

    W, b = train_logreg(X_tr, y_tr, n_classes=3)
    pred_te = predict(X_te, W, b)
    acc_te = accuracy(y_te, pred_te)
    base_te = majority_baseline(y_tr, y_te)
    print(f"[学習マップ・held-outテスト] モニタの精度={acc_te:.4f}, 多数派ベースライン={base_te:.4f}, チャンスレート=0.333")

    # --- 未経験のマップでの汎化テスト ---
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    test_map_env = HomeostasisEnv(random.Random(TEST_SEED))
    X_new, y_new = collect_rollout(test_map_env, agent, N_EPISODES_TEST_MAP, ROLLOUT_EPS)
    pred_new = predict(X_new, W, b)
    acc_new = accuracy(y_new, pred_new)
    base_new = majority_baseline(y_tr, y_new)
    print(f"[未経験マップ] サンプル数={len(X_new)}, モニタの精度={acc_new:.4f}, 多数派ベースライン={base_new:.4f}, チャンスレート=0.333")
    print(f"[未経験マップ] クラス分布: エネルギー={np.mean(y_new==0):.3f}, 体温={np.mean(y_new==1):.3f}, 損傷={np.mean(y_new==2):.3f}")

    # --- 可視化 ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["学習マップ\n(held-out)", "未経験マップ"]
    monitor_accs = [acc_te, acc_new]
    baseline_accs = [base_te, base_new]
    x = np.arange(len(labels))
    width = 0.3
    ax.bar(x - width / 2, monitor_accs, width, label="モニタの精度", color="#4472C4")
    ax.bar(x + width / 2, baseline_accs, width, label="多数派ベースライン", color="#BFBFBF")
    ax.axhline(1 / 3, color="#C0504D", linestyle="--", linewidth=1.5, label="チャンスレート(1/3)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("精度(支配的な逸脱の種類を当てられた割合)")
    ax.set_title("要件7プロトタイプ: 行動だけからセンサー状態を推定できるか")
    ax.legend()
    fig.tight_layout()
    fig.savefig("monitor_accuracy.png", dpi=150)
    print("グラフを monitor_accuracy.png に保存しました。")
