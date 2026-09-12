"""
感情AIプロジェクト フェーズ7 プロトタイプ: 要件7 最小限のNN移行(アーキテクチャのみ変更)
==========================================================

これまでの要件7の3つの結論(U字型の非単調性・行動空間依存性・複数マップ+履歴長8手
による汎化改善)はすべて、タブラーQ学習(状態を離散キーとして辞書に保存)の上で
得られたものだった。本プロトタイプは、エージェントの内部実装だけをQテーブルから
MLP(2隠れ層)+経験リプレイ+ターゲットネットワークによる標準的なDQN形式に置き換え、
それ以外(環境・グリッド・マップ、報酬設計、状態表現(相対方向)、モニタの定義
(行動履歴8手・chosen_q・q_gapを特徴量とする線形回帰))は既存のタブラー実験と
完全に同一に保つことで、アーキテクチャの変更そのものが3つの結論にどう影響するかを
検証する。

**変更する点**: エージェントの内部実装のみ。
  - Qテーブル(辞書) → MLP(9入力→隠れ32→隠れ32→5出力、ReLU、Adam最適化)
  - 経験リプレイ(容量5000、バッチサイズ32)
  - ターゲットネットワーク(200ステップごとに同期)
  - 状態のエンコード: HomeostasisEnv.discrete_state()が返す
    (food_dir, shelter_dir, hazard_dir, e_bin, t_bin, d_bin)を、方向成分(-1/0/1)は
    そのまま、ビン(0-5)は5で正規化した9次元の実数ベクトルに変換する。これは
    タブラー版が状態を辞書キーとして厳密照合していたのと「同じ情報」をNNが処理
    できる形式に変換しただけであり、状態表現(相対方向)自体は変更していない。

**変更しない点**: HomeostasisEnv(グリッド・マップ生成・報酬)、eps減衰スケジュール
(instinct_bias_prototype.epsilon_for_episode)、モニタの特徴量設計(選んだ行動の
one-hot・chosen_q・q_gap・直近8手の行動one-hot、方向特徴量なし)と学習方法(リッジ
回帰、monitor_feature_richness_prototype.build_features/monitor_maturity_
prototype.fit_linear_regressionをそのまま再利用)は完全に既存のタブラー実験と同一。

**検証する3点**:
  (1) U字型の再現性: monitor_history8_maturity_prototype.pyと同じ設計(単一マップ
      学習・チェックポイント150/500/1500/3000ep)で、NN版でもheld-out相関のU字型
      (非単調)が現れるか。タブラー版のn=3平均(150ep=0.5410, 500ep=0.4594,
      1500ep=0.3879, 3000ep=0.6195)と比較する。
  (2) 複数マップ+履歴8手の汎化改善: monitor_history_sweep_prototype.pyのhlen=8・
      複数マップ学習と同じ設計で、未経験マップ相関がタブラー版のn=3平均(0.189)と
      比べてどう変わるか。
  (3) grokking的な跳躍: 学習を8000epまで延長し、500ep刻みでチェックポイントを取り、
      複数マップ+履歴8手の未経験マップ相関の推移に、長い停滞の後の急激な改善が
      見られるかを確認する(計算コスト削減のため、この検証だけロールアウトの
      エピソード数を減らす)。

規模: まずn=3(traj_seed=0,11,22)で様子を見て、何らかの効果が見えた場合は
n=15まで拡大する。

45秒のbash呼び出し制限に対応するため、これまでと同じ「時間主導」のチャンク実行
方式(内部で一定エピソード数ずつ学習を進めながら時間予算を監視し、超過前に状態を
保存して呼び出しを終える)を採用する。

使い方:
  python3 homeostasis_nn_prototype.py sanity                        # 動作確認・タイミング計測
  python3 homeostasis_nn_prototype.py partA_chunk <traj_seed>        # (1)U字型、単一マップ学習
  python3 homeostasis_nn_prototype.py partB_chunk <traj_seed>        # (2)複数マップ+履歴8手
  python3 homeostasis_nn_prototype.py partC_chunk <traj_seed>        # (3)grokking探索(8000ep)
  python3 homeostasis_nn_prototype.py aggregate <part>                # partA/partB/partC
"""

import sys, json, pickle, time, random
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from homeostasis_prototype import HomeostasisEnv, ACTIONS, GAMMA as TAB_GAMMA
import instinct_bias_prototype as ib
from monitor_maturity_prototype import fit_linear_regression, predict_linear, mean_correlation, accuracy
import monitor_feature_richness_prototype as mfr

mfr.MAX_HISTORY = 8  # このプロトタイプでは履歴長8手のみを使う(要件7の確立レシピ)

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------
# DQN(MLP+経験リプレイ+ターゲットネットワーク)の実装
# ------------------------------------------------------------
N_ACTIONS = len(ACTIONS)
STATE_DIM = 9
HIDDEN1 = 32
HIDDEN2 = 32
GAMMA = TAB_GAMMA           # タブラー版と同じ割引率(0.95)を使用
LR = 0.001
BUFFER_CAPACITY = 5000
BATCH_SIZE = 32
TARGET_SYNC_STEPS = 200


def encode_state(state):
    food_dir, shelter_dir, hazard_dir, e_bin, t_bin, d_bin = state
    return np.array([
        food_dir[0], food_dir[1], shelter_dir[0], shelter_dir[1],
        hazard_dir[0], hazard_dir[1], e_bin / 5.0, t_bin / 5.0, d_bin / 5.0,
    ], dtype=np.float64)


class MLPParams:
    def __init__(self, rng=None, from_arrays=None):
        if from_arrays is not None:
            (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3) = from_arrays
            return
        self.W1 = rng.randn(STATE_DIM, HIDDEN1) * np.sqrt(2.0 / STATE_DIM)
        self.b1 = np.zeros(HIDDEN1)
        self.W2 = rng.randn(HIDDEN1, HIDDEN2) * np.sqrt(2.0 / HIDDEN1)
        self.b2 = np.zeros(HIDDEN2)
        self.W3 = rng.randn(HIDDEN2, N_ACTIONS) * np.sqrt(2.0 / HIDDEN2)
        self.b3 = np.zeros(N_ACTIONS)

    def copy(self):
        return MLPParams(from_arrays=(
            self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(),
            self.W3.copy(), self.b3.copy(),
        ))


def forward(params, X):
    z1 = X @ params.W1 + params.b1
    h1 = np.maximum(0.0, z1)
    z2 = h1 @ params.W2 + params.b2
    h2 = np.maximum(0.0, z2)
    q = h2 @ params.W3 + params.b3
    return q, (X, z1, h1, z2, h2)


class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim))
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity)
        self.next_states = np.zeros((capacity, state_dim))
        self.dones = np.zeros(capacity)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, sn, d):
        i = self.ptr
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = sn
        self.dones[i] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.randint(0, self.size, batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])


class AdamState:
    def __init__(self, params):
        self.m = {k: np.zeros_like(getattr(params, k)) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        self.v = {k: np.zeros_like(getattr(params, k)) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        self.t = 0


def adam_step(params, grads, adam, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8):
    adam.t += 1
    for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        g = grads[name]
        adam.m[name] = beta1 * adam.m[name] + (1 - beta1) * g
        adam.v[name] = beta2 * adam.v[name] + (1 - beta2) * (g * g)
        mhat = adam.m[name] / (1 - beta1 ** adam.t)
        vhat = adam.v[name] / (1 - beta2 ** adam.t)
        setattr(params, name, getattr(params, name) - lr * mhat / (np.sqrt(vhat) + eps))


class DQNAgent:
    """QLearningAgentと互換のインターフェース(best_action, update)を持つNN版エージェント。"""

    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.params = MLPParams(rng=rng)
        self.target_params = self.params.copy()
        self.adam = AdamState(self.params)
        self.buffer = ReplayBuffer(BUFFER_CAPACITY, STATE_DIM)
        self.np_rng = np.random.RandomState(seed + 777)
        self.step_count = 0

    def q_values(self, state):
        x = encode_state(state)[None, :]
        q, _ = forward(self.params, x)
        return q[0]

    def best_action(self, state):
        q = self.q_values(state)
        return ACTIONS[int(np.argmax(q))]

    def update(self, state, action, reward, next_state, done):
        x = encode_state(state)
        xn = encode_state(next_state)
        a_idx = ACTIONS.index(action)
        self.buffer.add(x, a_idx, reward, xn, 1.0 if done else 0.0)
        self.step_count += 1
        if self.buffer.size >= BATCH_SIZE:
            self._train_step()
        if self.step_count % TARGET_SYNC_STEPS == 0:
            self.target_params = self.params.copy()

    def _train_step(self):
        S, A, R, Sn, D = self.buffer.sample(BATCH_SIZE, self.np_rng)
        q_next, _ = forward(self.target_params, Sn)
        max_q_next = np.max(q_next, axis=1)
        target = R + GAMMA * (1.0 - D) * max_q_next

        q_pred, cache = forward(self.params, S)
        pred_chosen = q_pred[np.arange(BATCH_SIZE), A]
        d_loss = 2.0 * (pred_chosen - target) / BATCH_SIZE

        dQ = np.zeros_like(q_pred)
        dQ[np.arange(BATCH_SIZE), A] = d_loss

        X_in, z1, h1, z2, h2 = cache
        dW3 = h2.T @ dQ
        db3 = dQ.sum(axis=0)
        dh2 = dQ @ self.params.W3.T
        dz2 = dh2 * (z2 > 0)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.params.W2.T
        dz1 = dh1 * (z1 > 0)
        dW1 = X_in.T @ dz1
        db1 = dz1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        adam_step(self.params, grads, self.adam)


class EvalPolicy:
    """特定のチェックポイントの重みだけを使い、学習は一切行わない評価専用ラッパー。"""

    def __init__(self, params):
        self.params = params

    def q_values(self, state):
        x = encode_state(state)[None, :]
        q, _ = forward(self.params, x)
        return q[0]

    def best_action(self, state):
        q = self.q_values(state)
        return ACTIONS[int(np.argmax(q))]


# ------------------------------------------------------------
# 学習ループ・ロールアウト(タブラー版と同じインターフェース)
# ------------------------------------------------------------

def train_nn_with_checkpoints(env, agent, n_episodes, decay_episodes, checkpoint_eps):
    checkpoint_set = set(checkpoint_eps)
    checkpoints = {}
    for ep in range(n_episodes):
        state = env.reset()
        eps = ib.epsilon_for_episode(ep, decay_episodes)
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
            checkpoints[ep + 1] = agent.params.copy()
    return checkpoints


def nn_collect_rollout_raw(env, policy, n_episodes, eps):
    """monitor_feature_richness_prototype.collect_rollout_rawと同じレコード形式を、
    Qテーブルの代わりにNN方策(DQNAgentまたはEvalPolicy)から生成する。"""
    records = []
    for ep in range(n_episodes):
        state = env.reset()
        history = ["stay"] * mfr.MAX_HISTORY
        done = False
        while not done:
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = policy.best_action(state)
            q_values = policy.q_values(state)
            sorted_q = sorted(q_values, reverse=True)
            chosen_q = float(q_values[ACTIONS.index(action)])
            q_gap = float(sorted_q[0] - (sorted_q[1] if len(sorted_q) > 1 else sorted_q[0]))
            food_dir, shelter_dir, hazard_dir = state[0], state[1], state[2]

            next_state, reward, done, deviation = env.step(action)
            dev_vec = mfr.deviations(env.energy, env.temperature, env.damage)

            records.append({
                "action": action, "chosen_q": chosen_q, "q_gap": q_gap,
                "history": list(history),
                "food_dir": food_dir, "shelter_dir": shelter_dir, "hazard_dir": hazard_dir,
                "y_cont": dev_vec,
            })
            history = history[1:] + [action]
            state = next_state
    return records


# ------------------------------------------------------------
# Part A: U字型の再検証(単一マップ学習、チェックポイント150/500/1500/3000ep)
# ------------------------------------------------------------
TRAIN_SEED = 0
CHECKPOINT_EPISODES_A = [150, 500, 1500, 3000]
ROLLOUT_EPS = 0.1
N_EPISODES_TRAIN_MAP = 100
TRAJ_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22]  # n=3->n=15拡大(Part A/B、他実験と同じ拡大慣例)
TRAJ_SEEDS_C = [0, 11, 22]  # Part C(grokking探索)は効果なしのためn=3のまま

# タブラー版(履歴8手モニタ)の既存結果、n=3平均(history8_maturity_seed{0,11,22}.jsonより算出)
BASELINE_A_TABULAR = {150: 0.5410, 500: 0.4594, 1500: 0.3879, 3000: 0.6195}


def partA_multi_chunk(traj_seed, time_budget=36.0):
    t_start = time.time()
    state_file = f"nn_partA_state_seed{traj_seed}.pkl"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        agent = state["agent"]
        ep_done = state["ep_done"]
        checkpoints = state["checkpoints"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        agent = DQNAgent(seed=traj_seed)
        ep_done = 0
        checkpoints = {}
        print(f"[partA seed={traj_seed}] 新規開始")

    env = HomeostasisEnv(random.Random(TRAIN_SEED))
    checkpoint_set = set(CHECKPOINT_EPISODES_A)
    target_end = max(CHECKPOINT_EPISODES_A)

    while ep_done < target_end:
        if (time.time() - t_start) > time_budget:
            with open(state_file, "wb") as f:
                pickle.dump({
                    "agent": agent, "ep_done": ep_done, "checkpoints": checkpoints,
                    "random_state": random.getstate(), "np_random_state": np.random.get_state(),
                }, f)
            print(f"[partA seed={traj_seed}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")
            return
        eps = ib.epsilon_for_episode(ep_done, ib.PARENT_EPS_DECAY_EPISODES)
        s = env.reset()
        done = False
        while not done:
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = agent.best_action(s)
            sn, r, done, dev = env.step(a)
            agent.update(s, a, r, sn, done)
            s = sn
        ep_done += 1
        if ep_done in checkpoint_set:
            checkpoints[ep_done] = agent.params.copy()
            print(f"[partA seed={traj_seed}] {ep_done}epでチェックポイント保存")

    # 学習完了。各チェックポイントでheld-out相関を評価。
    result = {"traj_seed": traj_seed, "checkpoints": {}}
    for n_ep in CHECKPOINT_EPISODES_A:
        random.seed(traj_seed * 1000 + n_ep)
        np.random.seed(traj_seed * 1000 + n_ep)
        policy = EvalPolicy(checkpoints[n_ep])
        map_env = HomeostasisEnv(random.Random(TRAIN_SEED))
        records = nn_collect_rollout_raw(map_env, policy, N_EPISODES_TRAIN_MAP, ROLLOUT_EPS)
        X_all, Y_all = mfr.build_features(records, 8, False)
        n = len(X_all)
        idx = np.random.permutation(n)
        split = int(n * 0.7)
        W = fit_linear_regression(X_all[idx[:split]], Y_all[idx[:split]])
        pred_te = predict_linear(X_all[idx[split:]], W)
        corr_te = mean_correlation(Y_all[idx[split:]], pred_te)
        result["checkpoints"][str(n_ep)] = {"corr_holdout": corr_te}
        print(f"[partA seed={traj_seed}] {n_ep}ep: held-out相関(NN)={corr_te:.4f} "
              f"(タブラー版n=3平均: {BASELINE_A_TABULAR[n_ep]:.4f})")

    with open(f"nn_partA_result_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved nn_partA_result_seed{traj_seed}.json")


def aggregate_partA():
    data = [json.load(open(f"nn_partA_result_seed{s}.json")) for s in TRAJ_SEEDS]
    print("=== (1) U字型の再検証(NN版、n=3平均±標準偏差) ===")
    summary = {}
    for n_ep in CHECKPOINT_EPISODES_A:
        vals = [d["checkpoints"][str(n_ep)]["corr_holdout"] for d in data]
        summary[n_ep] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
        print(f"{n_ep}ep: NN held-out相関={np.mean(vals):.4f}±{np.std(vals):.4f} "
              f"(タブラー版: {BASELINE_A_TABULAR[n_ep]:.4f})")

    with open("nn_partA_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ns = CHECKPOINT_EPISODES_A
    nn_means = [summary[n]["mean"] for n in ns]
    nn_stds = [summary[n]["std"] for n in ns]
    tab_vals = [BASELINE_A_TABULAR[n] for n in ns]
    ax.errorbar(ns, nn_means, yerr=nn_stds, marker="o", label="NN版(DQN)", color="#4472C4", linewidth=2)
    ax.plot(ns, tab_vals, "o--", label="タブラー版(既存結果)", color="#C0504D", linewidth=2)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("エージェントの学習量(episode数)")
    ax.set_ylabel("held-out相関(履歴8手モニタ)")
    ax.set_title("要件7: U字型はNN移行でも現れるか")
    ax.legend()
    fig.tight_layout()
    fig.savefig("homeostasis_nn_partA_comparison.png", dpi=150)
    print("グラフを homeostasis_nn_partA_comparison.png に保存しました。")


# ------------------------------------------------------------
# Part B: 複数マップ+履歴8手の汎化改善(タブラー版0.189との比較)
# ------------------------------------------------------------
MULTI_MAP_SEEDS = [0, 1, 2, 3]
UNSEEN_SEEDS = [5, 6, 7]
N_EPISODES_PER_MAP = 100
N_EPISODES_UNSEEN = 60

BASELINE_B_TABULAR_UNSEEN = 0.189
BASELINE_B_TABULAR_HOLDOUT = 0.4352


def partB_multi_chunk(traj_seed, time_budget=36.0):
    t_start = time.time()
    state_file = f"nn_partB_state_seed{traj_seed}.pkl"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        agent = state["agent"]
        ep_done = state["ep_done"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        agent = DQNAgent(seed=traj_seed)
        ep_done = 0
        print(f"[partB seed={traj_seed}] 新規開始(方策学習)")

    env = HomeostasisEnv(random.Random(TRAIN_SEED))
    target_end = ib.PARENT_EPISODES  # 3000ep、タブラー版と同じ学習量

    while ep_done < target_end:
        if (time.time() - t_start) > time_budget:
            with open(state_file, "wb") as f:
                pickle.dump({
                    "agent": agent, "ep_done": ep_done,
                    "random_state": random.getstate(), "np_random_state": np.random.get_state(),
                }, f)
            print(f"[partB seed={traj_seed}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")
            return
        eps = ib.epsilon_for_episode(ep_done, ib.PARENT_EPS_DECAY_EPISODES)
        s = env.reset()
        done = False
        while not done:
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = agent.best_action(s)
            sn, r, done, dev = env.step(a)
            agent.update(s, a, r, sn, done)
            s = sn
        ep_done += 1

    print(f"[partB seed={traj_seed}] 方策学習完了({ep_done}ep)、複数マップ・未経験マップのロールアウトを実行")
    policy = EvalPolicy(agent.params)

    train_records = []
    for seed in MULTI_MAP_SEEDS:
        random.seed(traj_seed * 1000 + seed)
        np.random.seed(traj_seed * 1000 + seed)
        map_env = HomeostasisEnv(random.Random(seed))
        train_records.extend(nn_collect_rollout_raw(map_env, policy, N_EPISODES_PER_MAP, ROLLOUT_EPS))

    unseen_records = {}
    for seed in UNSEEN_SEEDS:
        random.seed(traj_seed * 5000 + seed)
        np.random.seed(traj_seed * 5000 + seed)
        map_env = HomeostasisEnv(random.Random(seed))
        unseen_records[seed] = nn_collect_rollout_raw(map_env, policy, N_EPISODES_UNSEEN, ROLLOUT_EPS)

    rng = np.random.RandomState(traj_seed * 999)
    X_all, Y_all = mfr.build_features(train_records, 8, False)
    perm = rng.permutation(len(X_all))
    split = int(len(X_all) * 0.7)
    W = fit_linear_regression(X_all[perm[:split]], Y_all[perm[:split]])
    corr_holdout = mean_correlation(Y_all[perm[split:]], predict_linear(X_all[perm[split:]], W))

    unseen_corrs = []
    for seed in UNSEEN_SEEDS:
        X_u, Y_u = mfr.build_features(unseen_records[seed], 8, False)
        unseen_corrs.append(mean_correlation(Y_u, predict_linear(X_u, W)))

    result = {
        "traj_seed": traj_seed,
        "corr_holdout": corr_holdout,
        "corr_unseen_mean": float(np.mean(unseen_corrs)),
        "corr_unseen_std": float(np.std(unseen_corrs)),
        "corr_unseen_list": unseen_corrs,
    }
    with open(f"nn_partB_result_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[partB seed={traj_seed}] held-out相関(NN)={corr_holdout:.4f} "
          f"(タブラー版{BASELINE_B_TABULAR_HOLDOUT:.4f}), "
          f"未経験マップ相関(NN)={np.mean(unseen_corrs):.4f}±{np.std(unseen_corrs):.4f} "
          f"(タブラー版{BASELINE_B_TABULAR_UNSEEN:.4f})")
    print(f"saved nn_partB_result_seed{traj_seed}.json")


def aggregate_partB():
    data = [json.load(open(f"nn_partB_result_seed{s}.json")) for s in TRAJ_SEEDS]
    holdout_vals = [d["corr_holdout"] for d in data]
    unseen_vals = [d["corr_unseen_mean"] for d in data]
    print("=== (2) 複数マップ+履歴8手の汎化改善(NN版、n=3平均±標準偏差) ===")
    print(f"held-out相関: NN={np.mean(holdout_vals):.4f}±{np.std(holdout_vals):.4f} "
          f"(タブラー版{BASELINE_B_TABULAR_HOLDOUT:.4f})")
    print(f"未経験マップ相関: NN={np.mean(unseen_vals):.4f}±{np.std(unseen_vals):.4f} "
          f"(タブラー版{BASELINE_B_TABULAR_UNSEEN:.4f})")

    summary = {
        "holdout_mean": float(np.mean(holdout_vals)), "holdout_std": float(np.std(holdout_vals)),
        "unseen_mean": float(np.mean(unseen_vals)), "unseen_std": float(np.std(unseen_vals)),
        "holdout_values": holdout_vals, "unseen_values": unseen_vals,
    }
    with open("nn_partB_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels = ["held-out\n(学習分布内)", "未経験マップ\n(真の汎化)"]
    nn_means = [np.mean(holdout_vals), np.mean(unseen_vals)]
    nn_stds = [np.std(holdout_vals), np.std(unseen_vals)]
    tab_means = [BASELINE_B_TABULAR_HOLDOUT, BASELINE_B_TABULAR_UNSEEN]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, tab_means, width, label="タブラー版(既存結果)", color="#BFBFBF")
    ax.bar(x + width / 2, nn_means, width, yerr=nn_stds, label="NN版(DQN)", color="#4472C4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("相関係数")
    ax.set_title("要件7: 複数マップ+履歴8手の汎化改善はNN移行でどう変わるか")
    ax.legend()
    fig.tight_layout()
    fig.savefig("homeostasis_nn_partB_comparison.png", dpi=150)
    print("グラフを homeostasis_nn_partB_comparison.png に保存しました。")


# ------------------------------------------------------------
# Part C: grokking的な跳躍の探索(8000epまで延長、500ep刻みでチェックポイント)
# ------------------------------------------------------------
CHECKPOINT_EPISODES_C = list(range(500, 8001, 500))  # 500,1000,...,8000 (16点)
N_EPISODES_PER_MAP_C = 40   # コスト削減のためPart Bより少なめ
N_EPISODES_UNSEEN_C = 25
DECAY_EPISODES_C = 2000     # タブラー版と同じ減衰スケジュールを維持(3000ep以降はEPS_END固定)


def partC_multi_chunk(traj_seed, time_budget=36.0):
    t_start = time.time()
    state_file = f"nn_partC_state_seed{traj_seed}.pkl"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        agent = state["agent"]
        ep_done = state["ep_done"]
        checkpoints_meta = state["checkpoints_done"]  # list of episode numbers already evaluated
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        agent = DQNAgent(seed=traj_seed)
        ep_done = 0
        checkpoints_meta = []
        print(f"[partC seed={traj_seed}] 新規開始")
        with open(f"nn_partC_result_seed{traj_seed}.json", "w") as f:
            json.dump({"traj_seed": traj_seed, "checkpoints": {}}, f)

    env = HomeostasisEnv(random.Random(TRAIN_SEED))
    checkpoint_set = set(CHECKPOINT_EPISODES_C)
    target_end = max(CHECKPOINT_EPISODES_C)

    while ep_done < target_end:
        if (time.time() - t_start) > time_budget:
            with open(state_file, "wb") as f:
                pickle.dump({
                    "agent": agent, "ep_done": ep_done, "checkpoints_done": checkpoints_meta,
                    "random_state": random.getstate(), "np_random_state": np.random.get_state(),
                }, f)
            print(f"[partC seed={traj_seed}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")
            return
        eps = ib.epsilon_for_episode(ep_done, DECAY_EPISODES_C)
        s = env.reset()
        done = False
        while not done:
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = agent.best_action(s)
            sn, r, done, dev = env.step(a)
            agent.update(s, a, r, sn, done)
            s = sn
        ep_done += 1
        if ep_done in checkpoint_set and ep_done not in checkpoints_meta:
            policy = EvalPolicy(agent.params.copy())
            train_records = []
            for seed in MULTI_MAP_SEEDS:
                random.seed(traj_seed * 1000 + seed + ep_done)
                np.random.seed((traj_seed * 1000 + seed + ep_done) % (2**32 - 1))
                map_env = HomeostasisEnv(random.Random(seed))
                train_records.extend(nn_collect_rollout_raw(map_env, policy, N_EPISODES_PER_MAP_C, ROLLOUT_EPS))
            unseen_corrs = []
            for seed in UNSEEN_SEEDS:
                random.seed(traj_seed * 5000 + seed + ep_done)
                np.random.seed((traj_seed * 5000 + seed + ep_done) % (2**32 - 1))
                map_env = HomeostasisEnv(random.Random(seed))
                u_records = nn_collect_rollout_raw(map_env, policy, N_EPISODES_UNSEEN_C, ROLLOUT_EPS)
                X_u, Y_u = mfr.build_features(u_records, 8, False)
                X_tr, Y_tr = mfr.build_features(train_records, 8, False)
                W_c = fit_linear_regression(X_tr, Y_tr)
                unseen_corrs.append(mean_correlation(Y_u, predict_linear(X_u, W_c)))
            unseen_mean = float(np.mean(unseen_corrs))
            checkpoints_meta.append(ep_done)

            with open(f"nn_partC_result_seed{traj_seed}.json", "r") as f:
                result = json.load(f)
            result["checkpoints"][str(ep_done)] = {
                "unseen_mean": unseen_mean, "unseen_std": float(np.std(unseen_corrs)),
            }
            with open(f"nn_partC_result_seed{traj_seed}.json", "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[partC seed={traj_seed}] {ep_done}ep: 未経験マップ相関(NN)={unseen_mean:.4f}")

    print(f"[partC seed={traj_seed}] target_end_ep={target_end}に到達、全チェックポイント評価済み")


def aggregate_partC():
    data = [json.load(open(f"nn_partC_result_seed{s}.json")) for s in TRAJ_SEEDS_C]
    print("=== (3) grokking的な跳躍の探索(NN版、n=3平均±標準偏差) ===")
    summary = {}
    for n_ep in CHECKPOINT_EPISODES_C:
        key = str(n_ep)
        vals = [d["checkpoints"][key]["unseen_mean"] for d in data if key in d["checkpoints"]]
        if len(vals) < len(TRAJ_SEEDS):
            continue
        summary[n_ep] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
        print(f"{n_ep}ep: 未経験マップ相関(NN)={np.mean(vals):.4f}±{np.std(vals):.4f}")

    with open("nn_partC_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    ns = sorted(summary.keys())
    means = [summary[n]["mean"] for n in ns]
    stds = [summary[n]["std"] for n in ns]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(ns, means, yerr=stds, marker="o", color="#4472C4", linewidth=2)
    ax.axhline(BASELINE_B_TABULAR_UNSEEN, color="#C0504D", linestyle="--",
               label=f"タブラー版3000ep時点({BASELINE_B_TABULAR_UNSEEN:.3f})")
    ax.set_xlabel("エージェントの学習量(episode数)")
    ax.set_ylabel("未経験マップ相関(複数マップ+履歴8手)")
    ax.set_title("要件7: grokking的な跳躍は見られるか(NN版、8000epまで)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("homeostasis_nn_partC_grokking.png", dpi=150)
    print("グラフを homeostasis_nn_partC_grokking.png に保存しました。")


# ------------------------------------------------------------
# 動作確認・タイミング計測
# ------------------------------------------------------------
def sanity():
    random.seed(0)
    np.random.seed(0)
    env = HomeostasisEnv(random.Random(TRAIN_SEED))
    agent = DQNAgent(seed=0)
    t0 = time.time()
    for ep in range(50):
        eps = ib.epsilon_for_episode(ep, ib.PARENT_EPS_DECAY_EPISODES)
        s = env.reset()
        done = False
        steps = 0
        while not done:
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = agent.best_action(s)
            sn, r, done, dev = env.step(a)
            agent.update(s, a, r, sn, done)
            s = sn
            steps += 1
    dt = time.time() - t0
    print(f"50ep学習時間: {dt:.3f}s ({dt/50*1000:.2f}ms/ep)")
    print(f"buffer size: {agent.buffer.size}, step_count: {agent.step_count}")

    # ロールアウト+モニタ特徴量抽出の動作確認
    policy = EvalPolicy(agent.params)
    t1 = time.time()
    records = nn_collect_rollout_raw(env, policy, 10, ROLLOUT_EPS)
    X, Y = mfr.build_features(records, 8, False)
    print(f"ロールアウト10ep: {time.time()-t1:.3f}s, X.shape={X.shape}, Y.shape={Y.shape}")
    W = fit_linear_regression(X[:70], Y[:70])
    pred = predict_linear(X[70:], W)
    print("相関(サニティ、意味のある値である必要はない):", mean_correlation(Y[70:], pred))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sanity":
        sanity()
    elif cmd == "partA_chunk":
        partA_multi_chunk(int(sys.argv[2]))
    elif cmd == "partB_chunk":
        partB_multi_chunk(int(sys.argv[2]))
    elif cmd == "partC_chunk":
        partC_multi_chunk(int(sys.argv[2]))
    elif cmd == "aggregate":
        part = sys.argv[2]
        if part == "partA":
            aggregate_partA()
        elif part == "partB":
            aggregate_partB()
        elif part == "partC":
            aggregate_partC()
