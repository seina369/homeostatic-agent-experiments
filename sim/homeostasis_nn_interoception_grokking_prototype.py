"""
感情AIプロジェクト フェーズ1・2・7合同プロトタイプ: 内受容感覚の部分観測化によるgrokking検証
==========================================================

前回2つの難化実験(グリッドサイズ拡大・食料/シェルター方向の部分観測化)は
いずれもgrokkingを誘発できなかった。その考察は「モニタが予測する対象
(dominant_deviation)を直接規定するエネルギー・体温・損傷のビンが常に完全
観測のままだったため、モニタの予測課題自体の難易度が変わらなかった」という
ものだった。本プロトタイプは、この本丸である内受容感覚(エネルギー・体温・
損傷のビン)自体を部分観測化する。

**変更する軸**: エージェントが受け取るセンサー観測値(e_bin・t_bin・d_bin)を、
各ステップ・各チャンネル独立に確率pで「観測不可(欠落)」に置き換える。
欠落時は値を0.0に固定し、欠落フラグ(e_missing・t_missing・d_missing)を
別次元として追加する(食料/シェルター方向の部分観測化実験と同じ設計
パターン)。**内部的な真のセンサー値の推移(エネルギー減衰・体温ドリフト・
損傷)と、それに基づく報酬・エピソード強制終了(エネルギー枯渇)の判定は
一切変えない**。変わるのは「エージェントが今その値を見られるかどうか」だけ
で、要件1・2が検証した恒常性システムの環境動態そのものには触れない。
食料・シェルター・危険地帯への方向情報は今回は完全観測に戻す(前回の
部分観測化実験との違いを明確にするため、難化軸を1つに絞る)。

**入力次元**: 6(方向、完全観測)+3(e/t/dの値)+3(e/t/d欠落フラグ)=12次元。

**第1段階(検証ゲート、要件1・2の前提保護)**: まずp=0.3程度で、恒常性維持の
学習自体がなお成立するか(死亡率・センサー逸脱の抑制が破綻しないか)を短い
学習量(3000ep)・n=3で確認する。ここで学習が崩壊するようならpを下げる
(0.15など)。この段階は要件7の主目的ではなく、あくまで土台が壊れていない
ことの確認。

**第2段階(本実験)**: 検証済みのpを使い、これまでのNN実験と同じ設定
(グリッド8×8、複数マップ学習、履歴8手、方向情報は完全観測)で、20000epまで
モニタの学習マップ内相関・未経験マップ相関を別々に追跡し、行動エントロピー・
状態行動多様性も並行記録してgrokkingの有無を検証する。

**規模**: 第1段階・第2段階ともにまずn=3。第2段階で明確な兆候があればn=15へ
拡大する。

使い方:
  python3 homeostasis_nn_interoception_grokking_prototype.py stage1_chunk <traj_seed> [p]
  python3 homeostasis_nn_interoception_grokking_prototype.py stage1_aggregate [p]
  python3 homeostasis_nn_interoception_grokking_prototype.py chunk <traj_seed> [p]
  python3 homeostasis_nn_interoception_grokking_prototype.py aggregate [p] [n15]
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
hp.GRID_SIZE = 8  # 基準値(難化軸は内受容感覚の部分観測化のみに絞る)

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

STATE_DIM_INTERO = 12
m.STATE_DIM = STATE_DIM_INTERO  # MLPParams生成時にこの次元を参照させる

N_EPISODES_PER_MAP_CKPT = 40
N_EPISODES_UNSEEN_CKPT = 25

CHECKPOINT_EPISODES = [250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000,
                        5000, 6500, 8000, 10000, 12500, 15000, 17500, 20000]
STAGE1_CHECKPOINTS = [250, 500, 1000, 1500, 2000, 2500, 3000]

TRAJ_SEEDS_PRELIM = [0, 11, 22]
TRAJ_SEEDS_15 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22]
DEFAULT_P = 0.3


def encode_interoception_partial(state, p_dropout):
    food_dir, shelter_dir, hazard_dir, e_bin, t_bin, d_bin = state
    e_missing = 1.0 if random.random() < p_dropout else 0.0
    t_missing = 1.0 if random.random() < p_dropout else 0.0
    d_missing = 1.0 if random.random() < p_dropout else 0.0
    e_val = 0.0 if e_missing else e_bin / 5.0
    t_val = 0.0 if t_missing else t_bin / 5.0
    d_val = 0.0 if d_missing else d_bin / 5.0
    return np.array([
        food_dir[0], food_dir[1], shelter_dir[0], shelter_dir[1],
        hazard_dir[0], hazard_dir[1], e_val, t_val, d_val,
        e_missing, t_missing, d_missing,
    ], dtype=np.float64)


class PartialInteroDQNAgent:
    """内受容感覚(e/t/dビン)を部分観測化した観測ベクトルを受け取るDQN。
    フォワードパス・リプレイ・Adam更新はhomeostasis_nn_prototype.pyの
    次元非依存の実装をそのまま再利用する。"""

    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.params = m.MLPParams(rng=rng)
        self.target_params = self.params.copy()
        self.adam = m.AdamState(self.params)
        self.buffer = m.ReplayBuffer(m.BUFFER_CAPACITY, STATE_DIM_INTERO)
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


class PartialInteroEvalPolicy:
    def __init__(self, params):
        self.params = params

    def q_values_obs(self, obs):
        q, _ = m.forward(self.params, obs[None, :])
        return q[0]

    def best_action_obs(self, obs):
        return ACTIONS[int(np.argmax(self.q_values_obs(obs)))]


# ------------------------------------------------------------
# 第1段階: 検証ゲート(恒常性維持学習が破綻しないかの短期確認)
# ------------------------------------------------------------

def stage1_chunk(traj_seed, p_dropout=DEFAULT_P, time_budget=36.0):
    t_start = time.time()
    ptag = f"p{int(p_dropout*100)}"
    state_file = f"nn_intero_stage1_state_{ptag}_seed{traj_seed}.pkl"
    result_file = f"nn_intero_stage1_result_{ptag}_seed{traj_seed}.json"
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
        agent = PartialInteroDQNAgent(seed=traj_seed)
        ep_done = 0
        checkpoints_done = []
        print(f"[stage1 p={p_dropout} seed={traj_seed}] 新規開始(検証ゲート)")
        with open(result_file, "w") as f:
            json.dump({"traj_seed": traj_seed, "p_dropout": p_dropout, "checkpoints": {}}, f)

    env = hp.HomeostasisEnv(random.Random(TRAIN_SEED))
    checkpoint_set = set(STAGE1_CHECKPOINTS)
    target_end = max(STAGE1_CHECKPOINTS)

    dev_window = []
    death_window = []

    while ep_done < target_end:
        if (time.time() - t_start) > time_budget:
            with open(state_file, "wb") as f:
                pickle.dump({
                    "agent": agent, "ep_done": ep_done, "checkpoints_done": checkpoints_done,
                    "random_state": random.getstate(), "np_random_state": np.random.get_state(),
                }, f)
            print(f"[stage1 p={p_dropout} seed={traj_seed}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")
            return
        eps = ib.epsilon_for_episode(ep_done, DECAY_EPISODES)
        state = env.reset()
        done = False
        devs = []
        while not done:
            obs = encode_interoception_partial(state, p_dropout)
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = agent.best_action_obs(obs)
            next_state, r, done, dev = env.step(a)
            next_obs = encode_interoception_partial(next_state, p_dropout)
            agent.update_obs(obs, a, r, next_obs, done)
            devs.append(dev)
            state = next_state
        died = env.energy <= 0.0
        dev_window.append(float(np.mean(devs)))
        death_window.append(int(died))
        ep_done += 1
        if ep_done in checkpoint_set and ep_done not in checkpoints_done:
            avg_dev = float(np.mean(dev_window[-250:]))
            death_rate = float(np.mean(death_window[-250:]))
            checkpoints_done.append(ep_done)
            with open(result_file, "r") as f:
                result = json.load(f)
            result["checkpoints"][str(ep_done)] = {"avg_deviation": avg_dev, "death_rate": death_rate}
            with open(result_file, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[stage1 p={p_dropout} seed={traj_seed}] {ep_done}ep: 直近250ep平均逸脱={avg_dev:.4f}, "
                  f"死亡率={death_rate:.4f}")

    print(f"[stage1 p={p_dropout} seed={traj_seed}] target_end_ep={target_end}に到達")


def stage1_aggregate(p_dropout=DEFAULT_P, seeds=None):
    seeds = seeds or TRAJ_SEEDS_PRELIM
    ptag = f"p{int(p_dropout*100)}"
    data = [json.load(open(f"nn_intero_stage1_result_{ptag}_seed{s}.json")) for s in seeds]
    print(f"=== 第1段階: 検証ゲート(p={p_dropout}, n={len(seeds)}平均±標準偏差) ===")
    for n_ep in STAGE1_CHECKPOINTS:
        key = str(n_ep)
        rows = [d["checkpoints"][key] for d in data if key in d["checkpoints"]]
        if len(rows) < len(seeds):
            continue
        devs = [r["avg_deviation"] for r in rows]
        deaths = [r["death_rate"] for r in rows]
        print(f"{n_ep}ep: 平均逸脱={np.mean(devs):.4f}±{np.std(devs):.4f}, "
              f"死亡率={np.mean(deaths):.4f}±{np.std(deaths):.4f}")


# ------------------------------------------------------------
# 第2段階: 本実験(grokking判定)
# ------------------------------------------------------------

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
    records = []
    action_counter = {}
    state_action_set = set()
    n_steps = 0
    for ep in range(n_episodes):
        state = env.reset()
        history = ["stay"] * mfr.MAX_HISTORY
        done = False
        while not done:
            obs = encode_interoception_partial(state, p_dropout)
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
    policy = PartialInteroEvalPolicy(params)
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
    state_file = f"nn_intero_state_{ptag}_seed{traj_seed}.pkl"
    result_file = f"nn_intero_result_{ptag}_seed{traj_seed}.json"
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
        agent = PartialInteroDQNAgent(seed=traj_seed)
        ep_done = 0
        checkpoints_done = []
        print(f"[intero p={p_dropout} seed={traj_seed}] 新規開始(GRID_SIZE={hp.GRID_SIZE}, STATE_DIM={STATE_DIM_INTERO})")
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
            print(f"[intero p={p_dropout} seed={traj_seed}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")
            return
        eps = ib.epsilon_for_episode(ep_done, DECAY_EPISODES)
        state = env.reset()
        done = False
        while not done:
            obs = encode_interoception_partial(state, p_dropout)
            if random.random() < eps:
                a = random.choice(ACTIONS)
            else:
                a = agent.best_action_obs(obs)
            next_state, r, done, dev = env.step(a)
            next_obs = encode_interoception_partial(next_state, p_dropout)
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
            print(f"[intero p={p_dropout} seed={traj_seed}] {ep_done}ep: held-out={metrics['corr_holdout']:.4f}, "
                  f"未経験={metrics['corr_unseen_mean']:.4f}, 行動エントロピー={metrics['action_entropy_bits']:.4f}bit, "
                  f"状態行動多様性={metrics['state_action_diversity_frac']:.4f}")

    print(f"[intero p={p_dropout} seed={traj_seed}] target_end_ep={target_end}に到達、全チェックポイント評価済み")


def aggregate(p_dropout=DEFAULT_P, seeds=None):
    seeds = seeds or TRAJ_SEEDS_PRELIM
    ptag = f"p{int(p_dropout*100)}"
    data = [json.load(open(f"nn_intero_result_{ptag}_seed{s}.json")) for s in seeds]
    print(f"=== 内受容感覚の部分観測化によるgrokking検証(p={p_dropout}, n={len(seeds)}平均±標準偏差) ===")
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

    with open(f"nn_intero_summary_{ptag}.json", "w") as f:
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

    fig.suptitle(f"要件1・2/7: 内受容感覚の部分観測化によるgrokking検証(p={p_dropout})")
    fig.tight_layout()
    fig.savefig(f"homeostasis_nn_interoception_grokking_{ptag}_comparison.png", dpi=150)
    print(f"グラフを homeostasis_nn_interoception_grokking_{ptag}_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stage1_chunk":
        traj_seed = int(sys.argv[2])
        p = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_P
        stage1_chunk(traj_seed, p)
    elif cmd == "stage1_aggregate":
        p = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_P
        stage1_aggregate(p)
    elif cmd == "chunk":
        traj_seed = int(sys.argv[2])
        p = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_P
        chunk(traj_seed, p)
    elif cmd == "aggregate":
        p = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_P
        seeds = TRAJ_SEEDS_15 if (len(sys.argv) > 3 and sys.argv[3] == "n15") else TRAJ_SEEDS_PRELIM
        aggregate(p, seeds)
