"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 最小限のNN移行(アーキテクチャのみ変更)
==========================================================

community_signal_v2_prototype.py(推測ゲームによる直接報酬で信号創発に成功した
タブラー版、n=3平均MI=0.0224bit@3500ep)を基準に、アーキテクチャをNNに変える
だけで結果がどう変わるかを検証する。要件7の最小限NN移行実験と同じ方針で、
環境(4×4グリッド・衝突ペナルティ8.0)・報酬設計(推測ゲームの直接報酬)・
信号チャンネルの定義・状態表現・学習量(3500ep)は一切変更しない。

**変更する点**: 送り手(移動方策、旧QLearningAgent)・受け手(解釈方策、旧GuessAgent)
の両方を、Qテーブル/文脈付きバンディットのテーブルから、MLP(2隠れ層32×32、ReLU)
+経験リプレイ(容量5000、バッチ32)+ターゲットネットワーク(200ステップ同期)による
DQN形式に置き換える。
  - 移動方策(NNMoveAgent): 状態(food_dir, shelter_dir, hazard_dir, e_bin, t_bin,
    d_bin, last_signal(相手の直前signal), partner_dir)を12次元の実数ベクトルに
    エンコードし、6行動(up/down/left/right/stay/signal)のQ値を出力。要件7の
    homeostasis_nn_prototype.pyと同じDQN更新式(2乗誤差のTD誤差を手動逆伝播)。
  - 解釈方策(NNGuessAgent): 相手の直前signal(0/1)のみを入力とする1次元の
    「文脈付きバンディット」をDQN形式に変換。単一ステップの意思決定なので
    done=1.0固定(次状態への割引項は常に0になり、実質的にターゲット=即時報酬
    への回帰になる)。これはタブラー版のGuessAgent.update(TD更新
    q += ALPHA*(reward-q))と等価な設計をNN上で再現したもの。

**変更しない点**: MultiAgentHomeostasisEnv(4×4グリッド・衝突ペナルティ8.0)、
run_episode(推測ゲームの報酬構造)、GUESS_BONUS=1.0、GUESS_EPS=0.2、
N_EPISODES=3500、DECAY_EPISODES=2500、CHECKPOINT_EPISODES=[300,1500,3500]は
すべてcommunity_signal_v2_prototype.pyをそのままインポートして再利用する。

規模: まずn=3(traj_seed=0,11,22)。タブラー版との差が見られればn=15へ拡大する。

45秒のbash呼び出し制限に対応するため、時間主導のチャンク実行方式(内部で
episodeを1つずつ進めながら経過時間を監視し、36秒で状態を保存して終了、
次回呼び出しで自動再開)を採用する。

使い方:
  python3 community_signal_v2_nn_prototype.py chunk <traj_seed>     # 学習(自動再開・自動チェックポイント評価)
  python3 community_signal_v2_nn_prototype.py aggregate [n15]        # 集計・グラフ
"""

import sys, json, pickle, time, random
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import community_signal_v2_prototype as csv2
import instinct_bias_prototype as ib
import homeostasis_nn_prototype as m

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------
# 環境・報酬・学習量はcommunity_signal_v2_prototype.pyから完全にそのまま再利用
# ------------------------------------------------------------
ACTIONS_COMM = csv2.ACTIONS_COMM
GUESS_CLASSES = csv2.GUESS_CLASSES
TRAIN_SEED = csv2.TRAIN_SEED
TRAJ_SEEDS = [0, 11, 22]
TRAJ_SEEDS_15 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22]
N_EPISODES = csv2.N_EPISODES              # 3500
DECAY_EPISODES = csv2.DECAY_EPISODES      # 2500
CHECKPOINT_EPISODES = csv2.CHECKPOINT_EPISODES  # [300, 1500, 3500]
ROLLOUT_EPS = csv2.ROLLOUT_EPS            # 0.1
N_ROLLOUT_EPISODES = csv2.N_ROLLOUT_EPISODES  # 100
GUESS_EPS = csv2.GUESS_EPS                # 0.2

# タブラー版n=3(seed=0,11,22)の実測値(community_v2_train_seed{0,11,22}.jsonより)
TABULAR_BASELINE = {
    300: {"mi_mean": 0.0053, "mi_std": 0.0030, "rate_mean": 0.1433, "rate_std": 0.0168,
          "gacc_mean": 0.4892, "gacc_std": 0.0881},
    1500: {"mi_mean": 0.0199, "mi_std": 0.0188, "rate_mean": 0.2149, "rate_std": 0.0432,
           "gacc_mean": 0.5712, "gacc_std": 0.2839},
    3500: {"mi_mean": 0.0224, "mi_std": 0.0044, "rate_mean": 0.0981, "rate_std": 0.0165,
           "gacc_mean": 0.6976, "gacc_std": 0.1746},
}

# ------------------------------------------------------------
# NN(移動方策)
# ------------------------------------------------------------
MOVE_STATE_DIM = 12
MOVE_N_ACTIONS = len(ACTIONS_COMM)  # 6
GUESS_STATE_DIM = 1
GUESS_N_ACTIONS = len(GUESS_CLASSES)  # 3
HIDDEN1 = m.HIDDEN1
HIDDEN2 = m.HIDDEN2
GAMMA = m.GAMMA
LR = m.LR
BUFFER_CAPACITY = m.BUFFER_CAPACITY
BATCH_SIZE = m.BATCH_SIZE
TARGET_SYNC_STEPS = m.TARGET_SYNC_STEPS


class MLPParamsGen:
    """2つの異なる入出力次元(移動方策12→6、解釈方策1→3)を同時に扱うため、
    homeostasis_nn_prototype.MLPParamsのモジュールグローバル依存版ではなく、
    次元を明示的にコンストラクタ引数で受け取る汎用版を用意した。forward/
    AdamState/adam_stepはhomeostasis_nn_prototype側の実装がパラメータの
    実際の配列形状から動作する次元非依存設計になっているため、そのまま
    再利用できる。"""

    def __init__(self, rng, state_dim, n_actions, hidden1=HIDDEN1, hidden2=HIDDEN2, from_arrays=None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        if from_arrays is not None:
            (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3) = from_arrays
            return
        self.W1 = rng.randn(state_dim, hidden1) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.randn(hidden2, n_actions) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(n_actions)

    def copy(self):
        return MLPParamsGen(
            None, self.state_dim, self.n_actions, self.hidden1, self.hidden2,
            from_arrays=(self.W1.copy(), self.b1.copy(), self.W2.copy(),
                          self.b2.copy(), self.W3.copy(), self.b3.copy()),
        )


def encode_move_state(state):
    food_dir, shelter_dir, hazard_dir, e_bin, t_bin, d_bin, last_signal_j, partner_dir = state
    return np.array([
        food_dir[0], food_dir[1], shelter_dir[0], shelter_dir[1],
        hazard_dir[0], hazard_dir[1], e_bin / 5.0, t_bin / 5.0, d_bin / 5.0,
        float(last_signal_j), partner_dir[0], partner_dir[1],
    ], dtype=np.float64)


def _dqn_train_step(params, target_params, adam, buffer, np_rng, batch_size):
    S, A, R, Sn, D = buffer.sample(batch_size, np_rng)
    q_next, _ = m.forward(target_params, Sn)
    max_q_next = np.max(q_next, axis=1)
    target = R + GAMMA * (1.0 - D) * max_q_next

    q_pred, cache = m.forward(params, S)
    pred_chosen = q_pred[np.arange(batch_size), A]
    d_loss = 2.0 * (pred_chosen - target) / batch_size

    dQ = np.zeros_like(q_pred)
    dQ[np.arange(batch_size), A] = d_loss

    X_in, z1, h1, z2, h2 = cache
    dW3 = h2.T @ dQ
    db3 = dQ.sum(axis=0)
    dh2 = dQ @ params.W3.T
    dz2 = dh2 * (z2 > 0)
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ params.W2.T
    dz1 = dh1 * (z1 > 0)
    dW1 = X_in.T @ dz1
    db1 = dz1.sum(axis=0)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
    m.adam_step(params, grads, adam, lr=LR)


class NNMoveAgent:
    """QLearningAgentと互換のインターフェース(best_action, update)を持つ移動方策。"""

    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.params = MLPParamsGen(rng, MOVE_STATE_DIM, MOVE_N_ACTIONS)
        self.target_params = self.params.copy()
        self.adam = m.AdamState(self.params)
        self.buffer = m.ReplayBuffer(BUFFER_CAPACITY, MOVE_STATE_DIM)
        self.np_rng = np.random.RandomState(seed + 777)
        self.step_count = 0

    def q_values(self, state):
        x = encode_move_state(state)[None, :]
        q, _ = m.forward(self.params, x)
        return q[0]

    def best_action(self, state):
        return ACTIONS_COMM[int(np.argmax(self.q_values(state)))]

    def update(self, state, action, reward, next_state, done):
        x = encode_move_state(state)
        xn = encode_move_state(next_state)
        a_idx = ACTIONS_COMM.index(action)
        self.buffer.add(x, a_idx, reward, xn, 1.0 if done else 0.0)
        self.step_count += 1
        if self.buffer.size >= BATCH_SIZE:
            _dqn_train_step(self.params, self.target_params, self.adam, self.buffer, self.np_rng, BATCH_SIZE)
        if self.step_count % TARGET_SYNC_STEPS == 0:
            self.target_params = self.params.copy()


class NNMoveEvalPolicy:
    def __init__(self, params):
        self.params = params

    def q_values(self, state):
        x = encode_move_state(state)[None, :]
        q, _ = m.forward(self.params, x)
        return q[0]

    def best_action(self, state):
        return ACTIONS_COMM[int(np.argmax(self.q_values(state)))]


class NNGuessAgent:
    """相手の直前signal(0/1)だけを入力とする単一ステップの文脈付きバンディットを、
    done=1.0固定(次状態への割引項が常に0)のDQNとして実装。タブラー版GuessAgentの
    TD更新 q+=ALPHA*(reward-q) と等価な「即時報酬への回帰」をNN上で再現する。"""

    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.params = MLPParamsGen(rng, GUESS_STATE_DIM, GUESS_N_ACTIONS)
        self.target_params = self.params.copy()
        self.adam = m.AdamState(self.params)
        self.buffer = m.ReplayBuffer(BUFFER_CAPACITY, GUESS_STATE_DIM)
        self.np_rng = np.random.RandomState(seed + 777)
        self.step_count = 0

    def _encode(self, sig):
        return np.array([float(sig)], dtype=np.float64)

    def q_values(self, sig):
        x = self._encode(sig)[None, :]
        q, _ = m.forward(self.params, x)
        return q[0]

    def best_guess(self, sig):
        return GUESS_CLASSES[int(np.argmax(self.q_values(sig)))]

    def act(self, sig, eps):
        if random.random() < eps:
            return random.choice(GUESS_CLASSES)
        return self.best_guess(sig)

    def update(self, sig, guess, reward):
        x = self._encode(sig)
        self.buffer.add(x, guess, reward, x, 1.0)  # done=1固定、次状態は不使用
        self.step_count += 1
        if self.buffer.size >= BATCH_SIZE:
            _dqn_train_step(self.params, self.target_params, self.adam, self.buffer, self.np_rng, BATCH_SIZE)
        if self.step_count % TARGET_SYNC_STEPS == 0:
            self.target_params = self.params.copy()


class NNGuessEvalPolicy:
    def __init__(self, params):
        self.params = params

    def q_values(self, sig):
        x = np.array([float(sig)], dtype=np.float64)[None, :]
        q, _ = m.forward(self.params, x)
        return q[0]

    def best_guess(self, sig):
        return GUESS_CLASSES[int(np.argmax(self.q_values(sig)))]


# ------------------------------------------------------------
# 学習・ロールアウト(csv2.run_episode / csv2.act / csv2.mutual_info_signal_vs_dev をそのまま再利用)
# ------------------------------------------------------------

def rollout_for_signal_analysis_nn(env, params0, params1, gparams0, gparams1, n_episodes, eps):
    agent0 = NNMoveEvalPolicy(params0)
    agent1 = NNMoveEvalPolicy(params1)
    guess0 = NNGuessEvalPolicy(gparams0)
    guess1 = NNGuessEvalPolicy(gparams1)
    records = []
    guess_correct = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            dom0 = env.dominant_deviation(0)
            dom1 = env.dominant_deviation(1)
            a0 = csv2.act(agent0, obs[0], eps)
            a1 = csv2.act(agent1, obs[1], eps)
            sig_for_guess0 = obs[0][6]
            sig_for_guess1 = obs[1][6]
            guess0_val = guess0.best_guess(sig_for_guess0)
            guess1_val = guess1.best_guess(sig_for_guess1)
            guess_correct.append(int(guess0_val == dom1))
            guess_correct.append(int(guess1_val == dom0))
            records.append((dom0, 1 if a0 == "signal" else 0))
            records.append((dom1, 1 if a1 == "signal" else 0))
            next_obs, rewards, done, deviations, collided = env.step([a0, a1])
            obs = next_obs
    return records, guess_correct


def chunk(traj_seed, time_budget=36.0):
    state_file = f"nn_comm_state_seed{traj_seed}.pkl"
    t_start = time.time()
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env = state["env"]
        agent0, agent1 = state["agent0"], state["agent1"]
        guess0, guess1 = state["guess0"], state["guess1"]
        avg_dev_hist = state["avg_dev_hist"]
        coll_hist = state["coll_hist"]
        guess_acc_hist = state["guess_acc_hist"]
        checkpoints = state["checkpoints"]
        ep_done = state["ep_done"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        env = csv2.MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
        agent0, agent1 = NNMoveAgent(seed=traj_seed * 4 + 1), NNMoveAgent(seed=traj_seed * 4 + 2)
        guess0, guess1 = NNGuessAgent(seed=traj_seed * 4 + 3), NNGuessAgent(seed=traj_seed * 4 + 4)
        avg_dev_hist, coll_hist, guess_acc_hist = [], [], []
        checkpoints = {}
        ep_done = 0
        print(f"[comm-nn seed={traj_seed}] 新規開始(移動方策{MOVE_STATE_DIM}->{MOVE_N_ACTIONS}、解釈方策{GUESS_STATE_DIM}->{GUESS_N_ACTIONS})")

    checkpoint_set = set(CHECKPOINT_EPISODES)
    while ep_done < N_EPISODES:
        eps = ib.epsilon_for_episode(ep_done, DECAY_EPISODES)
        avg_dev, coll_rate, guess_acc = csv2.run_episode(env, agent0, agent1, guess0, guess1, eps, eps, guess_eps=GUESS_EPS)
        avg_dev_hist.append(avg_dev)
        coll_hist.append(coll_rate)
        guess_acc_hist.append(guess_acc)
        ep_done += 1
        if ep_done in checkpoint_set:
            checkpoints[ep_done] = (agent0.params.copy(), agent1.params.copy(),
                                     guess0.params.copy(), guess1.params.copy())
            print(f"[comm-nn seed={traj_seed}] checkpoint {ep_done}ep 保存")
        if time.time() - t_start > time_budget:
            break

    state = {
        "env": env, "agent0": agent0, "agent1": agent1, "guess0": guess0, "guess1": guess1,
        "avg_dev_hist": avg_dev_hist, "coll_hist": coll_hist, "guess_acc_hist": guess_acc_hist,
        "checkpoints": checkpoints, "ep_done": ep_done,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    if ep_done >= N_EPISODES:
        finalize(traj_seed)
    else:
        print(f"[comm-nn seed={traj_seed}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")


def finalize(traj_seed):
    state_file = f"nn_comm_state_seed{traj_seed}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    avg_dev_hist = state["avg_dev_hist"]
    coll_hist = state["coll_hist"]
    guess_acc_hist = state["guess_acc_hist"]
    checkpoints = state["checkpoints"]

    print(f"[comm-nn seed={traj_seed}] === 土台: 衝突回避タスク自体の改善 ===")
    print(f"[comm-nn seed={traj_seed}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[:500]):.4f}, 推測精度={np.mean(guess_acc_hist[:500]):.4f}")
    print(f"[comm-nn seed={traj_seed}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[-500:]):.4f}, 推測精度={np.mean(guess_acc_hist[-500:]):.4f}")

    mi_by_checkpoint = {}
    for n_ep in CHECKPOINT_EPISODES:
        p0, p1, gp0, gp1 = checkpoints[n_ep]
        random.seed(traj_seed * 7000 + n_ep)
        np.random.seed(traj_seed * 7000 + n_ep)
        rollout_env = csv2.MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
        records, guess_correct = rollout_for_signal_analysis_nn(
            rollout_env, p0, p1, gp0, gp1, N_ROLLOUT_EPISODES, ROLLOUT_EPS
        )
        mi, signal_rate, cond_dist, marg_dist = csv2.mutual_info_signal_vs_class(records)
        guess_acc = float(np.mean(guess_correct))
        mi_by_checkpoint[str(n_ep)] = {
            "mi": mi, "signal_rate": signal_rate, "cond_dist_given_signal": cond_dist,
            "marginal_dist": marg_dist, "guess_accuracy": guess_acc,
        }
        print(f"[comm-nn seed={traj_seed}] {n_ep}ep: I(signal;dominant_dev)={mi:.4f}bit, 信号送信率={signal_rate:.4f}, "
              f"推測精度={guess_acc:.4f}(チャンス=0.333)")

    result = {
        "traj_seed": traj_seed,
        "avg_dev_history": avg_dev_hist,
        "collision_rate_history": coll_hist,
        "guess_acc_history": guess_acc_hist,
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(f"nn_comm_train_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[comm-nn seed={traj_seed}] target_end_ep={N_EPISODES}に到達、全チェックポイント評価済み・保存完了")


def aggregate(use_n15=False):
    seeds = TRAJ_SEEDS_15 if use_n15 else TRAJ_SEEDS
    train_data = []
    for seed in seeds:
        with open(f"nn_comm_train_seed{seed}.json") as f:
            train_data.append(json.load(f))

    n = len(seeds)
    print(f"=== 要件6 NN移行(送り手・受け手ともにDQN化)、n={n} ===")
    coll_early = [np.mean(d["collision_rate_history"][:500]) for d in train_data]
    coll_late = [np.mean(d["collision_rate_history"][-500:]) for d in train_data]
    dev_early = [np.mean(d["avg_dev_history"][:500]) for d in train_data]
    dev_late = [np.mean(d["avg_dev_history"][-500:]) for d in train_data]
    print(f"衝突率: 序盤(最初500ep)={np.mean(coll_early):.4f}±{np.std(coll_early):.4f}, "
          f"終盤(最後500ep)={np.mean(coll_late):.4f}±{np.std(coll_late):.4f}")
    print(f"平均逸脱: 序盤={np.mean(dev_early):.4f}±{np.std(dev_early):.4f}, "
          f"終盤={np.mean(dev_late):.4f}±{np.std(dev_late):.4f}")

    mi_summary = {}
    print("\n=== 信号と内部状態のMI・推測精度(チェックポイント別、タブラー版と比較) ===")
    for n_ep in CHECKPOINT_EPISODES:
        key = str(n_ep)
        mis = [d["mi_by_checkpoint"][key]["mi"] for d in train_data]
        rates = [d["mi_by_checkpoint"][key]["signal_rate"] for d in train_data]
        gaccs = [d["mi_by_checkpoint"][key]["guess_accuracy"] for d in train_data]
        mi_summary[n_ep] = {
            "mi_mean": float(np.mean(mis)), "mi_std": float(np.std(mis)),
            "rate_mean": float(np.mean(rates)), "rate_std": float(np.std(rates)),
            "gacc_mean": float(np.mean(gaccs)), "gacc_std": float(np.std(gaccs)),
        }
        base = TABULAR_BASELINE[n_ep]
        print(f"{n_ep}ep: NN版 MI={np.mean(mis):.4f}±{np.std(mis):.4f}bit, "
              f"信号送信率={np.mean(rates):.4f}±{np.std(rates):.4f}, "
              f"推測精度={np.mean(gaccs):.4f}±{np.std(gaccs):.4f} "
              f"/ タブラー版 MI={base['mi_mean']:.4f}±{base['mi_std']:.4f}bit, "
              f"送信率={base['rate_mean']:.4f}±{base['rate_std']:.4f}, "
              f"推測精度={base['gacc_mean']:.4f}±{base['gacc_std']:.4f}")

    if n >= 15:
        from scipy import stats
        mis_3500 = [d["mi_by_checkpoint"]["3500"]["mi"] for d in train_data]
        t_stat, p_val = stats.ttest_1samp(mis_3500, TABULAR_BASELINE[3500]["mi_mean"])
        print(f"\n3500ep MI(n=15)のタブラー版固定値{TABULAR_BASELINE[3500]['mi_mean']}に対する片側t検定: "
              f"t={t_stat:.4f}, p={p_val:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ep_idx = np.arange(len(train_data[0]["collision_rate_history"]))
    window = 100
    for d in train_data:
        arr = np.array(d["collision_rate_history"])
        smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
        axes[0].plot(smoothed, alpha=0.5, color="#4472C4")
    axes[0].set_xlabel("エピソード")
    axes[0].set_ylabel("衝突率(移動平均100ep)")
    axes[0].set_title(f"衝突率の学習推移(NN版, n={n})")

    ns = CHECKPOINT_EPISODES
    mi_means = [mi_summary[nep]["mi_mean"] for nep in ns]
    mi_stds = [mi_summary[nep]["mi_std"] for nep in ns]
    base_means = [TABULAR_BASELINE[nep]["mi_mean"] for nep in ns]
    base_stds = [TABULAR_BASELINE[nep]["mi_std"] for nep in ns]
    axes[1].errorbar(ns, mi_means, yerr=mi_stds, marker="o", color="#4472C4", label="NN版")
    axes[1].errorbar(ns, base_means, yerr=base_stds, marker="s", color="#C0504D", label="タブラー版(n=3基準値)")
    axes[1].set_xlabel("学習量(episode数)")
    axes[1].set_ylabel("I(signal;dominant_dev)[bit]")
    axes[1].set_title("信号のMI: NN版 vs タブラー版")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"要件6 最小限のNN移行(n={n})")
    fig.tight_layout()
    tag = "n15" if use_n15 else "n3"
    fig.savefig(f"community_signal_v2_nn_comparison_{tag}.png", dpi=150)
    print(f"グラフを community_signal_v2_nn_comparison_{tag}.png に保存しました。")

    with open(f"nn_comm_summary_{tag}.json", "w") as f:
        json.dump({"n": n, "mi_summary": mi_summary}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "chunk":
        tb = float(sys.argv[3]) if len(sys.argv) > 3 else 36.0
        chunk(int(sys.argv[2]), time_budget=tb)
    elif cmd == "aggregate":
        use_n15 = len(sys.argv) > 2 and sys.argv[2] == "n15"
        aggregate(use_n15=use_n15)
