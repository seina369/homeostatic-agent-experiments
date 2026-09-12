"""
感情AIプロジェクト フェーズ12 プロトタイプ: 要件6複雑化 連続値の内部状態への拡張
==========================================================

community_signal_v2_nn_prototype.py(要件6の最小限NN移行、送り手・受け手とも
DQN化、n=15でタブラー版に対し統計的に有意に高いMIを確認済み)を基準に、
内部状態を離散クラス(3クラスのどれが支配的か)から連続値(逸脱の大きさそのもの)
に変え、よりショートカットの効きにくい設定でも「情報の必要性が鍵」という
これまでの結論が成り立つかを検証する。アーキテクチャ・環境・学習量は
要件6NN移行実験のものをそのまま使う。

**変更する点**:
1. 内部状態の表現: `dominant_deviation(i)`(argmaxによる3クラス分類)の代わりに
   `continuous_deviation(env, i)`(3センサーの正規化逸脱量のうち最大値、
   [0,~1]の連続実数)を推測対象にする。
2. 推測ゲームの報酬: 分類の正誤(0/1)の代わりに、
   quality = max(0, 1 - |推定値-真値|) という連続的な品質スコアを使う
   (真値と完全一致でquality=1、誤差1.0以上でquality=0の線形減衰)。
3. 受け手(ContinuousGuessAgent)のアーキテクチャ翻訳: 3クラスのQ値を出力する
   離散版NNGuessAgent(1→32→32→3)を、単一の連続推定値を出力する回帰版
   (1→32→32→1)に変更した。ただし学習則は既存のDQN機構(_dqn_train_step、
   done=1.0固定でTD目標=即時reward)をそのまま再利用できる: rewardとして
   分類の正誤ではなく真の逸脱量そのものを渡すことで、Q値(=唯一の出力)が
   真値へ直接回帰されるという性質を利用した(target = R + GAMMA*(1-D)*next
   がD=1で target=Rに簡約されるため、R=true_valueとすれば単純な2乗誤差
   回帰と数学的に同じ勾配になる)。推測ゲームは真値が環境から直接観測できる
   教師あり回帰問題であるため、離散版にあったGUESS_EPS(探索率)に相当する
   ランダム性は不要と判断し、常に現在の推定値をそのまま使う設計にした。
4. MI推定方法: 離散シャノンMI(2×3分割表)の代わりに、sklearn.feature_selection.
   mutual_info_classif(連続特徴量Xと離散ラベルyの間のMIを、Kraskov/Rossの
   k近傍ベースエントロピー推定で求める関数。ここではX=連続逸脱量、y=信号
   (0/1の離散ラベル)として使う)を採用した。sklearnの実装はnats単位で値を
   返すため、既存実験と単位を揃えるためln(2)で割りbitに変換している
   (rand一様分布からの決定的2値分割で検証: 期待値1.0bitに対し実測1.0004bit
   と正しく較正されていることを確認済み)。

**据え置く点**: 環境(MultiAgentHomeostasisEnv、4×4グリッド、衝突ペナルティ8.0)、
送り手(移動方策)のアーキテクチャ・状態表現・学習則はcommunity_signal_v2_nn_
prototype.pyのNNMoveAgentをそのまま再利用。信号チャンネル自体(action="signal"
による1ビットの二値通知)の形式は変更しない。学習量(N_EPISODES=3500)・
GUESS_BONUS=1.0は変更しない。

規模: まずn=3(traj_seed=0,11,22)。明確な創発の兆候(MIの上昇・推定誤差の
縮小)があればn=15へ拡大する。

使い方:
  python3 community_signal_v2_continuous_prototype.py chunk <traj_seed> [time_budget]
  python3 community_signal_v2_continuous_prototype.py aggregate [n15]
"""

import sys, json, pickle, time, random
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.feature_selection import mutual_info_classif

import homeostasis_prototype as hp
import community_signal_v2_prototype as csv2
import community_signal_v2_nn_prototype as csvnn
import instinct_bias_prototype as ib
import homeostasis_nn_prototype as m

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

ACTIONS_COMM = csv2.ACTIONS_COMM
TRAIN_SEED = csv2.TRAIN_SEED
TRAJ_SEEDS = [0, 11, 22]
TRAJ_SEEDS_15 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22]
N_EPISODES = csv2.N_EPISODES              # 3500
DECAY_EPISODES = csv2.DECAY_EPISODES      # 2500
CHECKPOINT_EPISODES = csv2.CHECKPOINT_EPISODES  # [300, 1500, 3500]
ROLLOUT_EPS = csv2.ROLLOUT_EPS            # 0.1
N_ROLLOUT_EPISODES = csv2.N_ROLLOUT_EPISODES  # 100
GUESS_BONUS = csv2.GUESS_BONUS            # 1.0

MOVE_STATE_DIM = csvnn.MOVE_STATE_DIM     # 12
GUESS_STATE_DIM = csvnn.GUESS_STATE_DIM   # 1
GUESS_N_ACTIONS_CONT = 1                  # 連続推定値の単一出力
BUFFER_CAPACITY = m.BUFFER_CAPACITY
BATCH_SIZE = m.BATCH_SIZE
TARGET_SYNC_STEPS = m.TARGET_SYNC_STEPS

# 参考: 要件6NN移行実験(離散3クラス版)のn=15実測値(nn_comm_summary_n15.jsonより)
DISCRETE_NN_BASELINE = {
    300: {"mi_mean": 0.0678, "mi_std": 0.1145},
    1500: {"mi_mean": 0.0872, "mi_std": 0.1081},
    3500: {"mi_mean": 0.1076, "mi_std": 0.1142},
}


def continuous_deviation(env, i):
    """支配的な逸脱センサーの正規化された逸脱量そのもの(argmaxクラスの代わりに
    max値を返す連続版)。"""
    dev_energy = abs(env.energy[i] - hp.OPTIMAL_ENERGY) / 100.0
    dev_temp = abs(env.temperature[i] - hp.OPTIMAL_TEMP) / 30.0
    dev_damage = abs(env.damage[i] - hp.OPTIMAL_DAMAGE) / 100.0
    return float(max(dev_energy, dev_temp, dev_damage))


def quality(estimate, true_val):
    return max(0.0, 1.0 - abs(estimate - true_val))


class ContinuousGuessAgent:
    """csvnn.NNGuessAgentのアーキテクチャ(1→32→32→N)を出力次元1(回帰)に
    変更した版。csvnn._dqn_train_stepをそのまま再利用し、reward=真の逸脱量
    (分類の正誤ではない)を渡すことで、単一のQ出力を真値へ直接回帰させる。"""

    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.params = csvnn.MLPParamsGen(rng, GUESS_STATE_DIM, GUESS_N_ACTIONS_CONT)
        self.target_params = self.params.copy()
        self.adam = m.AdamState(self.params)
        self.buffer = m.ReplayBuffer(BUFFER_CAPACITY, GUESS_STATE_DIM)
        self.np_rng = np.random.RandomState(seed + 777)
        self.step_count = 0

    def _encode(self, sig):
        return np.array([float(sig)], dtype=np.float64)

    def estimate(self, sig):
        x = self._encode(sig)[None, :]
        q, _ = m.forward(self.params, x)
        return float(q[0, 0])

    def update(self, sig, true_val):
        x = self._encode(sig)
        self.buffer.add(x, 0, true_val, x, 1.0)  # done=1固定、次状態は不使用
        self.step_count += 1
        if self.buffer.size >= BATCH_SIZE:
            csvnn._dqn_train_step(self.params, self.target_params, self.adam, self.buffer, self.np_rng, BATCH_SIZE)
        if self.step_count % TARGET_SYNC_STEPS == 0:
            self.target_params = self.params.copy()


class ContinuousGuessEvalPolicy:
    def __init__(self, params):
        self.params = params

    def estimate(self, sig):
        x = np.array([float(sig)], dtype=np.float64)[None, :]
        q, _ = m.forward(self.params, x)
        return float(q[0, 0])


def run_episode_continuous(env, agent0, agent1, guess0, guess1, eps0, eps1,
                            learn0=True, learn1=True, learn_guess0=True, learn_guess1=True):
    obs = env.reset()
    done = False
    devs, collisions, steps = [], 0, 0
    abs_err_sum, n_guesses = 0.0, 0
    while not done:
        dom0 = continuous_deviation(env, 0)
        dom1 = continuous_deviation(env, 1)

        a0 = csv2.act(agent0, obs[0], eps0)
        a1 = csv2.act(agent1, obs[1], eps1)

        sig_for_guess0 = obs[0][6]  # agent0が見る「相手(agent1)の直前signal」
        sig_for_guess1 = obs[1][6]  # agent1が見る「相手(agent0)の直前signal」
        est0 = guess0.estimate(sig_for_guess0)  # agent0によるagent1の逸脱量の推定
        est1 = guess1.estimate(sig_for_guess1)  # agent1によるagent0の逸脱量の推定
        q0 = quality(est0, dom1)
        q1 = quality(est1, dom0)
        abs_err_sum += abs(est0 - dom1) + abs(est1 - dom0)
        n_guesses += 1

        next_obs, base_rewards, done, deviations, collided = env.step([a0, a1])

        total_r0 = base_rewards[0] + GUESS_BONUS * q0 + GUESS_BONUS * q1
        total_r1 = base_rewards[1] + GUESS_BONUS * q1 + GUESS_BONUS * q0

        if learn0:
            agent0.update(obs[0], a0, total_r0, next_obs[0], done)
        if learn1:
            agent1.update(obs[1], a1, total_r1, next_obs[1], done)
        if learn_guess0:
            guess0.update(sig_for_guess0, dom1)
        if learn_guess1:
            guess1.update(sig_for_guess1, dom0)

        obs = next_obs
        devs.append((deviations[0] + deviations[1]) / 2.0)
        collisions += int(collided)
        steps += 1

    avg_dev = float(np.mean(devs))
    coll_rate = collisions / steps
    mae = abs_err_sum / (2 * n_guesses)
    return avg_dev, coll_rate, mae


def rollout_for_signal_analysis_continuous(env, params0, params1, gparams0, gparams1, n_episodes, eps):
    agent0 = csvnn.NNMoveEvalPolicy(params0)
    agent1 = csvnn.NNMoveEvalPolicy(params1)
    guess0 = ContinuousGuessEvalPolicy(gparams0)
    guess1 = ContinuousGuessEvalPolicy(gparams1)
    records = []       # (true_value, signaled) 送り手側のMI用
    abs_errors = []    # 推定誤差
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            dom0 = continuous_deviation(env, 0)
            dom1 = continuous_deviation(env, 1)
            a0 = csv2.act(agent0, obs[0], eps)
            a1 = csv2.act(agent1, obs[1], eps)
            sig_for_guess0 = obs[0][6]
            sig_for_guess1 = obs[1][6]
            est0 = guess0.estimate(sig_for_guess0)
            est1 = guess1.estimate(sig_for_guess1)
            abs_errors.append(abs(est0 - dom1))
            abs_errors.append(abs(est1 - dom0))
            records.append((dom0, 1 if a0 == "signal" else 0))
            records.append((dom1, 1 if a1 == "signal" else 0))
            next_obs, rewards, done, deviations, collided = env.step([a0, a1])
            obs = next_obs
    return records, abs_errors


def continuous_mutual_info_signal_vs_value(records, seed=0):
    """MI(連続な内部状態値; 離散な信号(0/1))をKSG/Ross法(sklearn実装)で推定し、
    natsからbitへ変換して返す。"""
    values = np.array([r[0] for r in records], dtype=float).reshape(-1, 1)
    signals = np.array([r[1] for r in records], dtype=int)
    signal_rate = float(np.mean(signals))
    if len(set(signals.tolist())) < 2:
        return 0.0, signal_rate
    mi_nats = mutual_info_classif(values, signals, discrete_features=False, n_neighbors=3, random_state=seed)[0]
    mi_bits = float(mi_nats / np.log(2))
    return mi_bits, signal_rate


# ------------------------------------------------------------
# 時間主導チャンク実行(45秒bash制限対応)
# ------------------------------------------------------------

def chunk(traj_seed, time_budget=36.0):
    state_file = f"cont_comm_state_seed{traj_seed}.pkl"
    t_start = time.time()
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env = state["env"]
        agent0, agent1 = state["agent0"], state["agent1"]
        guess0, guess1 = state["guess0"], state["guess1"]
        avg_dev_hist = state["avg_dev_hist"]
        coll_hist = state["coll_hist"]
        mae_hist = state["mae_hist"]
        checkpoints = state["checkpoints"]
        ep_done = state["ep_done"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        env = csv2.MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
        agent0, agent1 = csvnn.NNMoveAgent(seed=traj_seed * 4 + 1), csvnn.NNMoveAgent(seed=traj_seed * 4 + 2)
        guess0, guess1 = ContinuousGuessAgent(seed=traj_seed * 4 + 3), ContinuousGuessAgent(seed=traj_seed * 4 + 4)
        avg_dev_hist, coll_hist, mae_hist = [], [], []
        checkpoints = {}
        ep_done = 0
        print(f"[comm-cont seed={traj_seed}] 新規開始(移動方策{MOVE_STATE_DIM}->6、解釈方策{GUESS_STATE_DIM}->連続1出力)")

    checkpoint_set = set(CHECKPOINT_EPISODES)
    while ep_done < N_EPISODES:
        eps = ib.epsilon_for_episode(ep_done, DECAY_EPISODES)
        avg_dev, coll_rate, mae = run_episode_continuous(env, agent0, agent1, guess0, guess1, eps, eps)
        avg_dev_hist.append(avg_dev)
        coll_hist.append(coll_rate)
        mae_hist.append(mae)
        ep_done += 1
        if ep_done in checkpoint_set:
            checkpoints[ep_done] = (agent0.params.copy(), agent1.params.copy(),
                                     guess0.params.copy(), guess1.params.copy())
            print(f"[comm-cont seed={traj_seed}] checkpoint {ep_done}ep 保存")
        if time.time() - t_start > time_budget:
            break

    state = {
        "env": env, "agent0": agent0, "agent1": agent1, "guess0": guess0, "guess1": guess1,
        "avg_dev_hist": avg_dev_hist, "coll_hist": coll_hist, "mae_hist": mae_hist,
        "checkpoints": checkpoints, "ep_done": ep_done,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    if ep_done >= N_EPISODES:
        finalize(traj_seed)
    else:
        print(f"[comm-cont seed={traj_seed}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")


def finalize(traj_seed):
    state_file = f"cont_comm_state_seed{traj_seed}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    avg_dev_hist = state["avg_dev_hist"]
    coll_hist = state["coll_hist"]
    mae_hist = state["mae_hist"]
    checkpoints = state["checkpoints"]

    print(f"[comm-cont seed={traj_seed}] === 土台: 衝突回避タスク自体の改善 ===")
    print(f"[comm-cont seed={traj_seed}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[:500]):.4f}, 推定MAE={np.mean(mae_hist[:500]):.4f}")
    print(f"[comm-cont seed={traj_seed}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
          f"平均逸脱={np.mean(avg_dev_hist[-500:]):.4f}, 推定MAE={np.mean(mae_hist[-500:]):.4f}")

    mi_by_checkpoint = {}
    for n_ep in CHECKPOINT_EPISODES:
        p0, p1, gp0, gp1 = checkpoints[n_ep]
        random.seed(traj_seed * 7000 + n_ep)
        np.random.seed(traj_seed * 7000 + n_ep)
        rollout_env = csv2.MultiAgentHomeostasisEnv(random.Random(TRAIN_SEED))
        records, abs_errors = rollout_for_signal_analysis_continuous(
            rollout_env, p0, p1, gp0, gp1, N_ROLLOUT_EPISODES, ROLLOUT_EPS
        )
        mi, signal_rate = continuous_mutual_info_signal_vs_value(records, seed=traj_seed * 7000 + n_ep)
        mae = float(np.mean(abs_errors))
        mi_by_checkpoint[str(n_ep)] = {"mi": mi, "signal_rate": signal_rate, "mae": mae}
        print(f"[comm-cont seed={traj_seed}] {n_ep}ep: I(signal;continuous_dev)={mi:.4f}bit(KSG), "
              f"信号送信率={signal_rate:.4f}, 推定MAE(ロールアウト)={mae:.4f}")

    result = {
        "traj_seed": traj_seed,
        "avg_dev_history": avg_dev_hist,
        "collision_rate_history": coll_hist,
        "mae_history": mae_hist,
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(f"cont_comm_train_seed{traj_seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[comm-cont seed={traj_seed}] target_end_ep={N_EPISODES}に到達、全チェックポイント評価済み・保存完了")


def aggregate(use_n15=False):
    seeds = TRAJ_SEEDS_15 if use_n15 else TRAJ_SEEDS
    train_data = []
    for seed in seeds:
        with open(f"cont_comm_train_seed{seed}.json") as f:
            train_data.append(json.load(f))

    n = len(seeds)
    print(f"=== 要件6複雑化: 連続値の内部状態への拡張、n={n} ===")
    coll_early = [np.mean(d["collision_rate_history"][:500]) for d in train_data]
    coll_late = [np.mean(d["collision_rate_history"][-500:]) for d in train_data]
    dev_early = [np.mean(d["avg_dev_history"][:500]) for d in train_data]
    dev_late = [np.mean(d["avg_dev_history"][-500:]) for d in train_data]
    print(f"衝突率: 序盤(最初500ep)={np.mean(coll_early):.4f}±{np.std(coll_early):.4f}, "
          f"終盤(最後500ep)={np.mean(coll_late):.4f}±{np.std(coll_late):.4f}")
    print(f"平均逸脱: 序盤={np.mean(dev_early):.4f}±{np.std(dev_early):.4f}, "
          f"終盤={np.mean(dev_late):.4f}±{np.std(dev_late):.4f}")

    mi_summary = {}
    print("\n=== 信号と連続内部状態のMI(KSG)・推定MAE(チェックポイント別) ===")
    for n_ep in CHECKPOINT_EPISODES:
        key = str(n_ep)
        mis = [d["mi_by_checkpoint"][key]["mi"] for d in train_data]
        rates = [d["mi_by_checkpoint"][key]["signal_rate"] for d in train_data]
        maes = [d["mi_by_checkpoint"][key]["mae"] for d in train_data]
        mi_summary[n_ep] = {
            "mi_mean": float(np.mean(mis)), "mi_std": float(np.std(mis)),
            "rate_mean": float(np.mean(rates)), "rate_std": float(np.std(rates)),
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        }
        cv = (np.std(mis) / np.mean(mis)) if np.mean(mis) != 0 else float("nan")
        print(f"{n_ep}ep: MI={np.mean(mis):.4f}±{np.std(mis):.4f}bit(変動係数={cv:.2f}), "
              f"信号送信率={np.mean(rates):.4f}±{np.std(rates):.4f}, "
              f"推定MAE={np.mean(maes):.4f}±{np.std(maes):.4f}")

    if n >= 15:
        from scipy import stats
        mis_3500 = [d["mi_by_checkpoint"]["3500"]["mi"] for d in train_data]
        t_stat, p_val = stats.ttest_1samp(mis_3500, DISCRETE_NN_BASELINE[3500]["mi_mean"])
        print(f"\n3500ep MI(n=15)の離散NN版平均値{DISCRETE_NN_BASELINE[3500]['mi_mean']}に対する両側t検定: "
              f"t={t_stat:.4f}, p={p_val:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for d in train_data:
        arr = np.array(d["collision_rate_history"])
        window = 100
        smoothed = np.convolve(arr, np.ones(window) / window, mode="valid")
        axes[0].plot(smoothed, alpha=0.5, color="#4472C4")
    axes[0].set_xlabel("エピソード")
    axes[0].set_ylabel("衝突率(移動平均100ep)")
    axes[0].set_title(f"衝突率の学習推移(連続値版, n={n})")

    ns = CHECKPOINT_EPISODES
    mi_means = [mi_summary[nep]["mi_mean"] for nep in ns]
    mi_stds = [mi_summary[nep]["mi_std"] for nep in ns]
    disc_means = [DISCRETE_NN_BASELINE[nep]["mi_mean"] for nep in ns]
    disc_stds = [DISCRETE_NN_BASELINE[nep]["mi_std"] for nep in ns]
    axes[1].errorbar(ns, mi_means, yerr=mi_stds, marker="o", color="#4472C4", label="連続値版(KSG推定)")
    axes[1].errorbar(ns, disc_means, yerr=disc_stds, marker="s", color="#C0504D", label="離散3クラス版(n=15参考)")
    axes[1].set_xlabel("学習量(episode数)")
    axes[1].set_ylabel("I(signal;内部状態)[bit]")
    axes[1].set_title("信号のMI: 連続値版 vs 離散版")
    axes[1].legend(fontsize=9)

    fig.suptitle(f"要件6複雑化: 連続値の内部状態への拡張(n={n})")
    fig.tight_layout()
    tag = "n15" if use_n15 else "n3"
    fig.savefig(f"community_signal_continuous_comparison_{tag}.png", dpi=150)
    print(f"グラフを community_signal_continuous_comparison_{tag}.png に保存しました。")

    with open(f"cont_comm_summary_{tag}.json", "w") as f:
        json.dump({"n": n, "mi_summary": mi_summary}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "chunk":
        tb = float(sys.argv[3]) if len(sys.argv) > 3 else 36.0
        chunk(int(sys.argv[2]), time_budget=tb)
    elif cmd == "aggregate":
        use_n15 = len(sys.argv) > 2 and sys.argv[2] == "n15"
        aggregate(use_n15=use_n15)
