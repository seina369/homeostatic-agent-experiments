"""
感情AIプロジェクト フェーズ7 プロトタイプ: 要件7 状態表現の難化(部分観測化)によるgrokking再検証
==========================================================

前回の難化実験(グリッドサイズ8→16)は、状態表現(相対方向)自体が物理的な
グリッドサイズに依存しないスケール不変な設計だったため、難易度を実質的に
上げられずgrokkingは観測されなかった。本プロトタイプは、その考察を踏まえ、
今度はエージェントが実際に処理する知覚情報自体を難化させる。

**変更する軸は1つだけ**: 食料・シェルターへの相対方向シグナル(-1/0/1)を、
各ステップ・各チャンネル独立に確率p(まずp=0.3)で「観測不可(欠落)」に
置き換える部分観測化を導入する。欠落時は方向値を(0,0)に固定しつつ、
「欠落フラグ」を別次元として追加することで、真に方向がゼロ(既に目的地に
いる)場合と区別できるようにする(=情報の欠落に一本化し、符号反転などの
誤情報ノイズは加えない)。エージェントは、既存の8手の行動履歴を使って
この欠落を補い、隠れた方向情報を推測する必要が生じる。危険地帯への方向
(hazard_dir)は変更せず常に観測可能なまま。

観測のマスキングは「その瞬間にエージェントが実際に見た情報」を表すため、
行動選択に使う観測と、その遷移を経験リプレイに保存する観測は同一のマスク
(同一タイムステップの1回のコイン投げ)を使う。次状態の観測は次のタイム
ステップの独立したマスクを新たに引く(POMDPの観測関数として自然な設計)。

**据え置く点**: グリッドサイズは8×8(前回の拡大が無関係と分かったので基準値
に戻す)、ネットワーク構成(MLP、経験リプレイ+ターゲットネットワーク)・
報酬設計・複数マップ学習・履歴長8手・モニタの定義(build_features、
use_direction=False)は、これまでのNN実験と完全に同一に保つ。監視対象
(モニタが当てるべきセンサー逸脱の真値)にはノイズを加えない。状態行動
多様性の追跡は、エージェントが実際に観測した(欠落込みの)値ではなく、
環境の真の離散状態(discrete_state())を使う(前回のgrid実験と同じ定義)。

**入力次元の拡張**: 9次元(前回のNN実験)→11次元(food_missing・
shelter_missing の2フラグを追加)。MLPParamsクラス自体はhomeostasis_nn_
prototype.pyのものをそのまま再利用し、モジュール変数STATE_DIMを11に
書き換えることで対応する(community_signal_v2_prototype.pyのhp.GRID_SIZE=4
と同じパターン)。ただし観測のマスキングをエージェント内部ではなく学習
ループ側で行う必要があるため、DQNAgent/EvalPolicyは新たにPartialDQNAgent/
PartialEvalPolicy(生の状態ではなく既にエンコード済みの観測ベクトルを
受け取るインターフェース)として書き直した。フォワードパス・経験リプレイ・
Adam最適化のロジック自体はhomeostasis_nn_prototype.pyの実装(次元非依存)を
そのまま再利用している。

使い方:
  python3 homeostasis_nn_partialobs_grokking_prototype.py sanity [p]
  python3 homeostasis_nn_partialobs_grokking_prototype.py chunk <traj_seed> [p]
  python3 homeostasis_nn_partialobs_grokking_prototype.py aggregate [p]
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
hp.GRID_SIZE = 8  # 基準値に復帰(グリッドサイズは今回の難化軸ではない)

import instinct_bias_prototype as ib
import homeostasis_nn_prototype as m  # MLPParams/forward/ReplayBuffer/AdamState/adam_step/mfr等を再利用
import monitor_feature_richness_prototype as mfr

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

ACTIONS = m.ACTIONS
TRAIN_SEED = m.TRAIN_SEED
MULTI_MAP_SEEDS = m.MULTI_MAP_SEEDS
UNSEEN_SEEDS = m.UNSEEN_SEEDS
ROLLOUT_EPS = m.ROLLOUT_EPS
DECAY_EPISODES = ib.PARENT_EPS_DECAY_EPISODES  # 2000、これまでと同じ

STATE_DIM_PARTIAL = 11
m.STATE_DIM = STATE_DIM_PARTIAL  # MLPParams生成時にこの次元を参照させる

N_EPISODES_PER_MAP_CKPT = 40
N_EPISODES_UNSEEN_CKPT = 25

CHECKPOINT_EPISODES = [250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000,
                        5000, 6500, 8000, 10000, 12500, 15000, 17500, 20000]

TRAJ_SEEDS_PRELIM = [0, 11, 22]
TRAJ_SEEDS_15 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22]
DEFAULT_P = 0.3


def encode_partial(state, p_dropout):
    food_dir, shelter_dir, hazard_dir, e_bin, t_bin, d_bin = state
    food_missing = 1.0 if random.random() < p_dropout else 0.0
    shelter_missing = 1.0 if random.random() < p_dropout else 0.0
    fx, fy = (0, 0) if food_missing else food_dir
    sx, sy = (0, 0) if shelter_missing else shelter_dir
    return np.array([
        fx, fy, sx, sy, hazard_dir[0], hazard_dir[1],
        e_bin / 5.0, t_bin / 5.0, d_bin / 5.0, food_missing, shelter_missing,
    ], dtype=np.float64)


class PartialDQNAgent:
    """既にエンコード済みの観測ベクトル(部分観測込み)を受け取るDQNエージェント。
    フォワードパス・リプレイ・Adam更新のロジックはhomeostasis_nn_prototype.pyの
    ものをそのまま再利用する(いずれも次元非依存)。"""

    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.params = m.MLPParams(rng=rng)
        self.target_params = self.params.copy()
        self.adam = m.AdamState(self.params)
        self.buffer = m.ReplayBuffer(m.BUFFER_CAPACITY, STATE_DIM_PARTIAL)
        self.np_rng = np.random.RandomState(seed + 777)
        self.step_count = 0

    def q_values_obs(self, obs):
        q, _ = m.forward(self.params, obs[None, :])
        return q[0]

    def best_action_obs(self, obs):
        return ACTIONS[int(np.argmax(self.q_values_obs(obs)))]

    def update_obs(self, obs, action, reward, next_obs, done):
        a_idx = ACTIONS.index(action)
        self.buffer.add(obs, a_idx, reward, next_obs, 1.0 if done else 0.0)
        self.step_count += 1
        if self.buffer.size >= m.BATCH_SIZE:
            self._train_step()
        if self.step_count % m.TARGET_SYNC_STEPS == 0:
            self.target_params = self.params.copy()

    def _train_step(self):
        S, A, R, Sn, D = self.buffer.sample(m.BATCH_SIZE, self.np_rng)
        q_next, _ = m.forward(self.target_params, Sn)
        max_q_next = np.max(q_next, axis=1)
        target = R + m.GAMMA * (1.0 - D) * max_q_next

        q_pred, cache = m.forward(self.params, S)
        pred_chosen = q_pred[np.arange(m.BATCH_SIZE), A]
        d_loss = 2.0 * (pred_chosen - target) / m.BATCH_SIZE
        dQ = np.zeros_like(q_pred)
        dQ[np.arange(m.BATCH_SIZE), A] = d_loss

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
        m.adam_step(self.params, grads, self.adam)


class PartialEvalPolicy:
    def __init__(self, params):
        self.params = params

    def q_values_obs(self, obs):
        q, _ = m.forward(self.params, obs[None, :])
        return q[0]

    def best_action_obs(self, obs):
        return ACTIONS[int(np.argmax(self.q_values_obs(obs)))]


def action_entropy_bits(action_counter):
    total = sum(action_counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for a in ACTIONS:
        c = action_counter.get(a, 0)
        if c > 0:
            p = c / total
            ent -= p * np.log2(p)
    return float(ent)


def collect_rollout_full(env, policy, n_episodes, eps, p_dropout):
    """monitor_feature_richness_prototype.collect_rollout_rawと同じレコード
    形式に加え、行動エントロピー・(真の状態,行動)多様性を計算するための
    補助データも同時に集める。chosen_q/q_gapは、その時点で方策が実際に見た
    (部分観測込みの)観測ベクトルから計算する。"""
    records = []
    action_counter = {}
    state_action_set = set()
    n_steps = 0
    for ep in range(n_episodes):
        state = env.reset()
        history = ["stay"] * mfr.MAX_HISTORY
        done = False
        while not done:
            obs = encode_partial(state, p_dropout)
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = policy.best_action_obs(obs)
            q_values = policy.q_values_obs(obs)
            sorted_q = sorted(q_values, reverse=True)
            chosen_q = float(q_values[ACTIONS.index(action)])
            q_gap = float(sorted_q[0] - (sorted_q[1] if len(sorted_q) > 1 else sorted_q[0]))
            food_dir, shelter_dir, hazard_dir = state[0], state[1], state[2]

            action_counter[action] = action_counter.get(action, 0) + 1
            state_action_set.add((state, action))
            n_steps += 1

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
    return records, action_counter, len(state_action_set), n_steps


def evaluate_checkpoint(traj_seed, ep_done, params, p_dropout):
    policy = PartialEvalPolicy(params)
    train_records = []
    action_counter_total = {}
    n_unique_total = 0
    n_steps_total = 0
    for seed in MULTI_MAP_SEEDS:
        random.seed(traj_seed * 1000 + seed + ep_done)
        np.random.seed((traj_seed * 1000 + seed + ep_done) % (2**32 - 1))
        map_env = hp.HomeostasisEnv(random.Random(seed))
        recs, ac, n_unique, n_steps = collect_rollout_full(map_env, policy, N_EPISODES_PER_MAP_CKPT, ROLLOUT_EPS, p_dropout)
        train_records.extend(recs)
        for a, c in ac.items():
            action_counter_total[a] = action_counter_total.get(a, 0) + c
        n_unique_total += n_unique
        n_steps_total += n_steps

    unseen_records = {}
    for seed in UNSEEN_SEEDS:
        random.seed(traj_seed * 5000 + seed + ep_done)
        np.random.seed((traj_seed * 5000 + seed + ep_done) % (2**32 - 1))
        map_env = hp.HomeostasisEnv(random.Random(seed))
        recs, _, _, _ = collect_rollout_full(map_env, policy, N_EPISODES_UNSEEN_CKPT, ROLLOUT_EPS, p_dropout)
        unseen_records[seed] = recs

    rng = np.random.RandomState(traj_seed * 999 + ep_done)
    X_all, Y_all = mfr.build_features(train_records, 8, False)
    perm = rng.permutation(len(X_all))
    split = int(len(X_all) * 0.7)
    W = m.fit_linear_regression(X_all[perm[:split]], Y_all[perm[:split]])
    corr_holdout = m.mean_correlation(Y_all[perm[split:]], m.predict_linear(X_all[perm[split:]], W))

    unseen_corrs = []
    for seed in UNSEEN_SEEDS:
        X_u, Y_u = mfr.build_features(unseen_records[seed], 8, False)
        unseen_corrs.append(m.mean_correlation(Y_u, m.predict_linear(X_u, W)))

    entropy = action_entropy_bits(action_counter_total)
    diversity_frac = n_unique_total / n_steps_total if n_steps_total > 0 else 0.0

    return {
        "corr_holdout": corr_holdout,
        "corr_unseen_mean": float(np.mean(unseen_corrs)),
        "corr_unseen_std": float(np.std(unseen_corrs)),
        "action_entropy_bits": entropy,
        "state_action_unique": n_unique_total,
        "state_action_diversity_frac": diversity_frac,
    }


def chunk(traj_seed, p_dropout=DEFAULT_P, time_budget=36.0):
    t_start = time.time()
    ptag = f"p{int(p_dropout*100)}"
    state_file = f"nn_pobs_state_{ptag}_seed{traj_seed}.pkl"
    result_file = f"nn_pobs_result_{ptag}_seed{traj_seed}.json"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        agent = state["agent"]
        ep_done = state["ep_done"]
        checkpoints_done = state["checkpoints_done"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
    except FileNotFoundError:
        random.seed(traj_seed)
        np.random.seed(traj_seed)
        agent = PartialDQNAgent(seed=traj_seed)
        ep_done = 0
        checkpoints_done = []
        print(f"[pobs p={p_dropout} seed={traj_seed}] 新規開始(GRID_SIZE={hp.GRID_SIZE}, STATE_DIM={STATE_DIM_PARTIAL})")
        with open(result_file, "w") as f:
            json.dump({"traj_seed": traj_seed, "p_dropout": p_dropout, "checkpoints": {}}, f)

    env = hp.HomeostasisEnv(random.Random(TRAIN_SEED))
    checkpoint_set = set(CHECKPOINT_EPISODES)
    target_end = max(CHECKPOINT_EPISODES)

    while ep_done < target_end:
        if (time.time() - t_start) > time_budget:
            with open(state_file, "wb") as f:
                pickle.dump({
                    "agent": agent, "ep_done": ep_done, "checkpoints_done": checkpoints_done,
                    "random_state": random.getstate(), "np_random_state": np.random.get_state(),
                }, f)
            print(f"[pobs p={p_dropout} seed={traj_seed}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")
            return
        eps = ib.epsilon_for_episode(ep_done, DECAY_EPISODES)
        state = env.reset()
        done = False
        while not done:
            obs = encode_partial(state, p_dropout)
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = agent.best_action_obs(obs)
            next_state, r, done, dev = env.step(a)
            next_obs = encode_partial(next_state, p_dropout)
            agent.update_obs(obs, a, r, next_obs, done)
            state = next_state
        ep_done += 1
        if ep_done in checkpoint_set and ep_done not in checkpoints_done:
            metrics = evaluate_checkpoint(traj_seed, ep_done, agent.params.copy(), p_dropout)
            checkpoints_done.append(ep_done)
            with open(result_file, "r") as f:
                result = json.load(f)
            result["checkpoints"][str(ep_done)] = metrics
            with open(result_file, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[pobs p={p_dropout} seed={traj_seed}] {ep_done}ep: held-out={metrics['corr_holdout']:.4f}, "
                  f"未経験={metrics['corr_unseen_mean']:.4f}, 行動エントロピー={metrics['action_entropy_bits']:.4f}bit, "
                  f"状態行動多様性={metrics['state_action_diversity_frac']:.4f}")

    print(f"[pobs p={p_dropout} seed={traj_seed}] target_end_ep={target_end}に到達、全チェックポイント評価済み")


def aggregate(p_dropout=DEFAULT_P, seeds=None):
    seeds = seeds or TRAJ_SEEDS_PRELIM
    ptag = f"p{int(p_dropout*100)}"
    data = [json.load(open(f"nn_pobs_result_{ptag}_seed{s}.json")) for s in seeds]
    print(f"=== NN版 部分観測化によるgrokking再検証(p={p_dropout}, n={len(seeds)}平均±標準偏差) ===")
    summary = {}
    for n_ep in CHECKPOINT_EPISODES:
        key = str(n_ep)
        rows = [d["checkpoints"][key] for d in data if key in d["checkpoints"]]
        if len(rows) < len(seeds):
            continue
        holdout = [r["corr_holdout"] for r in rows]
        unseen = [r["corr_unseen_mean"] for r in rows]
        entropy = [r["action_entropy_bits"] for r in rows]
        diversity = [r["state_action_diversity_frac"] for r in rows]
        summary[n_ep] = {
            "holdout_mean": float(np.mean(holdout)), "holdout_std": float(np.std(holdout)),
            "unseen_mean": float(np.mean(unseen)), "unseen_std": float(np.std(unseen)),
            "entropy_mean": float(np.mean(entropy)), "entropy_std": float(np.std(entropy)),
            "diversity_mean": float(np.mean(diversity)), "diversity_std": float(np.std(diversity)),
        }
        print(f"{n_ep}ep: held-out={np.mean(holdout):.4f}±{np.std(holdout):.4f}, "
              f"未経験={np.mean(unseen):.4f}±{np.std(unseen):.4f}, "
              f"行動エントロピー={np.mean(entropy):.4f}±{np.std(entropy):.4f}bit, "
              f"状態行動多様性={np.mean(diversity):.4f}±{np.std(diversity):.4f}")

    with open(f"nn_pobs_summary_{ptag}.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    ns = sorted(summary.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].errorbar(ns, [summary[n]["holdout_mean"] for n in ns], yerr=[summary[n]["holdout_std"] for n in ns],
                      marker="o", label="held-out(学習分布内)", color="#4472C4")
    axes[0].errorbar(ns, [summary[n]["unseen_mean"] for n in ns], yerr=[summary[n]["unseen_std"] for n in ns],
                      marker="o", label="未経験マップ", color="#C0504D")
    axes[0].set_xlabel("エージェントの学習量(episode数)")
    axes[0].set_ylabel("相関係数")
    axes[0].set_title(f"held-out vs 未経験マップ精度の推移(p={p_dropout})")
    axes[0].legend()

    ax2 = axes[1]
    ax2.plot(ns, [summary[n]["entropy_mean"] for n in ns], "o-", color="#2E7D32", label="行動エントロピー(bit)")
    ax2.set_xlabel("エージェントの学習量(episode数)")
    ax2.set_ylabel("行動エントロピー(bit)", color="#2E7D32")
    ax3 = ax2.twinx()
    ax3.plot(ns, [summary[n]["diversity_mean"] for n in ns], "s--", color="#7030A0", label="状態行動多様性")
    ax3.set_ylabel("状態行動多様性(ユニーク率)", color="#7030A0")
    axes[1].set_title("行動複雑性の推移")

    fig.suptitle(f"要件7: NN版 部分観測化によるgrokking再検証(p={p_dropout})")
    fig.tight_layout()
    fig.savefig(f"homeostasis_nn_partialobs_grokking_{ptag}_comparison.png", dpi=150)
    print(f"グラフを homeostasis_nn_partialobs_grokking_{ptag}_comparison.png に保存しました。")


def sanity(p_dropout=DEFAULT_P):
    random.seed(0)
    np.random.seed(0)
    env = hp.HomeostasisEnv(random.Random(TRAIN_SEED))
    agent = PartialDQNAgent(seed=0)
    t0 = time.time()
    for ep in range(50):
        eps = ib.epsilon_for_episode(ep, DECAY_EPISODES)
        state = env.reset()
        done = False
        steps = 0
        while not done:
            obs = encode_partial(state, p_dropout)
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = agent.best_action_obs(obs)
            next_state, r, done, dev = env.step(a)
            next_obs = encode_partial(next_state, p_dropout)
            agent.update_obs(obs, a, r, next_obs, done)
            state = next_state
            steps += 1
    dt = time.time() - t0
    print(f"p={p_dropout}, GRID_SIZE={hp.GRID_SIZE}, 50ep学習時間: {dt:.3f}s ({dt/50*1000:.2f}ms/ep)")

    t1 = time.time()
    metrics = evaluate_checkpoint(999, 50, agent.params, p_dropout)
    print(f"チェックポイント評価: {time.time()-t1:.3f}s")
    print(metrics)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sanity":
        p = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_P
        sanity(p)
    elif cmd == "chunk":
        traj_seed = int(sys.argv[2])
        p = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_P
        chunk(traj_seed, p)
    elif cmd == "aggregate":
        p = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_P
        seeds = TRAJ_SEEDS_15 if (len(sys.argv) > 3 and sys.argv[3] == "n15") else TRAJ_SEEDS_PRELIM
        aggregate(p, seeds)
