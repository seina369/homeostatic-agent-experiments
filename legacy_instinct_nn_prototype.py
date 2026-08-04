"""
感情AIプロジェクト フェーズ4 プロトタイプ: 要件4後半 レガシー本能の最小限NN移行
==========================================================

legacy_instinct_prototype.py(教示行動"teach"への内発的報酬(レガシー報酬)により、
エルダーが自分の恒常性維持を一部犠牲にしてでも教える行動を学習し、その転写が
サクセサーの学習を助けるという用量反応をlegacy_bonus=0/1/3で確認したタブラー版)
を基準に、要件6・7と同じ方針でエージェントの内部実装だけをMLP+経験リプレイ+
ターゲットネットワークのDQNに置き換え、環境・報酬設計(内発的レガシー報酬)・
状態表現・学習量・転写手順の「精神」は完全に同一に保つ。

**タブラー版からの必然的な設計変更点(Qテーブルの離散構造に固有の仕組みを
NNの連続関数近似に翻訳する必要があったため)**:

1. 行動空間の拡張: タブラー版はhp.ACTIONSに"teach"を後から追加するだけで
   Qテーブルが自動的に新しい(state,"teach")エントリを0初期化で受け付けるが、
   NNは出力層の幅(行動数)をネットワーク構築時に固定する必要がある。そこで、
   まず5行動(up/down/left/right/stay)のネットワークでエルダーの基礎学習
   (ib.PARENT_EPISODES=3000ep)を行い、その後「隠れ層の重み(W1,b1,W2,b2)は
   そのままコピーし、出力層(W3,b3)は5行動分をコピーした上で6列目("teach")の
   重みを"stay"列と完全に同じ値で初期化する」ことで6行動ネットワークを構築する。
   これにより、教示フェーズ開始時点でどんな状態についてもQ(state,"teach")=
   Q(state,"stay")が厳密に成り立ち、タブラー版のseed_teach_baseline()
   (teachの初期値をstayの値に揃えて楽観的初期化バイアスを避ける)と数学的に
   同じ効果をNNの重み空間で再現している。

2. 転写(teach時のQテーブルエントリ混合コピー)のNN化: タブラー版は教示のたびに
   エルダーのQテーブルから(state,action)エントリをTRANSFER_COUNT個ランダムに
   サンプルし、各エントリをsuccessor.q[k] = old + BLEND*(elder.q[k]-old)で
   個別に混合コピーする。NNは1回の順伝播で状態1つにつき全行動のQ値を同時に
   出力する構造のため、個々の(state,action)エントリではなく「状態」を
   TRANSFER_COUNT個サンプルし、その状態についてエルダーの全行動Qベクトルと
   サクセサーの現在の全行動Qベクトルをブレンド(target = succ_q + BLEND×
   (elder_q - succ_q))し、その目標値へサクセサーのネットワークを1回転の
   回帰(勾配降下)で近づける。これはタブラー版の「エントリ単位の混合コピー」を
   「状態単位のQベクトル全体の混合コピー」に一般化したものであり、環境との
   相互作用や報酬は一切介さない点(サクセサーは教示フェーズ中、受動的に
   知識を受け取るだけ)はタブラー版と同一。

3. カバー率のNN化: タブラー版はcoverage=len(successor.q)/len(elder.q)
   (転写された(state,action)エントリ数の割合)で測る。NNには「エントリ」が
   存在しないため、エルダーが基礎学習+教示フェーズを通じて実際に訪れた
   状態の集合(elder_visited_states)と、転写に使われた状態の集合
   (successor_touched_states)を追跡し、coverage=len(successor_touched_states)/
   len(elder_visited_states)として、タブラー版と同じ「知識空間のうちどれだけが
   伝わったか」という意味を保つ。

変更しない点: 損傷閾値・強制終了条件は本実験には無関係、内発的レガシー報酬
(legacy_bonus∈{0,1,3})・状態表現(9次元、homeostasis_nn_prototype.encode_state
と同一)・学習量(ELDER_EPISODES=500、EVAL_EPISODES=300)・評価時の探索率補正
(eps_start=1-0.7×coverage)は完全にタブラー版と同一に保つ。

規模: n=3(RUN_SEEDS=[100,200,300]、タブラー版と同一)。タブラー版と大きく
違う結果が出た場合のみn=15へ拡大する。

使い方:
  python3 legacy_instinct_nn_prototype.py base_chunk [time_budget]   # エルダー基礎学習(3000ep、1回だけ)
  python3 legacy_instinct_nn_prototype.py run_chunk <legacy_bonus> <seed> [time_budget]
  python3 legacy_instinct_nn_prototype.py aggregate
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
from homeostasis_prototype import HomeostasisEnv
import instinct_bias_prototype as ib
import homeostasis_nn_prototype as m

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

ELDER_SEED = 0
EVAL_SEED = 2
STATE_DIM = 9
HIDDEN1 = m.HIDDEN1
HIDDEN2 = m.HIDDEN2

ELDER_EPISODES = 500
ELDER_EPS_DECAY_EPISODES = 300
EVAL_EPISODES = 300
EVAL_EPS_DECAY_EPISODES = 200

LEGACY_BONUSES = [0.0, 1.0, 3.0]
TRANSFER_COUNT = 3
BLEND = 0.3
RUN_SEEDS = [100, 200, 300]
RUN_SEEDS_15 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

ACTIONS = hp.ACTIONS  # 共有される可変リスト。teach追加後は全モジュールに反映される。


class MLPParamsGen:
    """次元をコンストラクタで明示的に指定する汎用MLPパラメータ。
    5行動の基礎ネットワークと6行動の教示フェーズネットワークを同時に
    扱うため、homeostasis_nn_prototype.MLPParams(モジュールグローバル
    STATE_DIM/N_ACTIONS依存)ではなくこちらを使う。"""

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


def _dqn_train_step(params, target_params, adam, buffer, np_rng, batch_size):
    S, A, R, Sn, D = buffer.sample(batch_size, np_rng)
    q_next, _ = m.forward(target_params, Sn)
    max_q_next = np.max(q_next, axis=1)
    target = R + m.GAMMA * (1.0 - D) * max_q_next

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
    m.adam_step(params, grads, adam, lr=m.LR)


class NNAgentGeneric:
    """状態次元・行動数を明示指定できる汎用DQNエージェント。init_paramsを渡せば
    その重みを初期値として学習を継続できる(レガシー知識の継承に相当)。
    interfaceはタブラー版QLearningAgentと同一(best_action, update)。"""

    def __init__(self, state_dim, n_actions, seed=0, init_params=None):
        rng = np.random.RandomState(seed)
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.params = init_params.copy() if init_params is not None else MLPParamsGen(rng, state_dim, n_actions)
        self.target_params = self.params.copy()
        self.adam = m.AdamState(self.params)
        self.buffer = m.ReplayBuffer(m.BUFFER_CAPACITY, state_dim)
        self.np_rng = np.random.RandomState(seed + 777)
        self.step_count = 0

    def q_values(self, state):
        x = m.encode_state(state)[None, :]
        q, _ = m.forward(self.params, x)
        return q[0]

    def best_action(self, state):
        q = self.q_values(state)
        return ACTIONS[int(np.argmax(q))]

    def update(self, state, action, reward, next_state, done):
        x = m.encode_state(state)
        xn = m.encode_state(next_state)
        a_idx = ACTIONS.index(action)
        self.buffer.add(x, a_idx, reward, xn, 1.0 if done else 0.0)
        self.step_count += 1
        if self.buffer.size >= m.BATCH_SIZE:
            _dqn_train_step(self.params, self.target_params, self.adam, self.buffer, self.np_rng, m.BATCH_SIZE)
        if self.step_count % m.TARGET_SYNC_STEPS == 0:
            self.target_params = self.params.copy()


class SuccessorNet:
    """教示フェーズ中は環境と一切相互作用せず、distill_stepによる回帰更新
    だけを受け取る受動的なネットワーク(タブラー版のsuccessor=QLearningAgent()
    白紙スタートに相当)。"""

    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.params = MLPParamsGen(rng, STATE_DIM, 6)
        self.adam = m.AdamState(self.params)


def distill_step(elder_params, successor_net, states, blend):
    """タブラー版のsuccessor.q[k] = old + BLEND*(elder.q[k]-old)という
    (state,action)エントリ単位の混合コピーを、状態単位のQベクトル全体の
    混合コピーへ一般化。サンプルされた各状態についてエルダーの全行動Qベクトルと
    サクセサーの現在の全行動Qベクトルをブレンドした目標値へ、1回の勾配降下で
    近づける。"""
    X = np.stack([m.encode_state(s) for s in states])
    elder_q, _ = m.forward(elder_params, X)
    succ_q, cache = m.forward(successor_net.params, X)
    target = succ_q + blend * (elder_q - succ_q)
    n = len(states)
    d_loss = 2.0 * (succ_q - target) / n
    X_in, z1, h1, z2, h2 = cache
    dW3 = h2.T @ d_loss
    db3 = d_loss.sum(axis=0)
    dh2 = d_loss @ successor_net.params.W3.T
    dz2 = dh2 * (z2 > 0)
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ successor_net.params.W2.T
    dz1 = dh1 * (z1 > 0)
    dW1 = X_in.T @ dz1
    db1 = dz1.sum(axis=0)
    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
    m.adam_step(successor_net.params, grads, successor_net.adam, lr=m.LR)


def epsilon_for_episode(ep, decay_episodes, eps_start=1.0, eps_end=0.05):
    frac = min(1.0, ep / decay_episodes)
    return eps_start + (eps_end - eps_start) * frac


def train_and_collect_states(env, agent, n_episodes, decay_episodes, eps_start=1.0, visited=None):
    """ib.train相当だが、訪れた状態集合も追跡する(coverage算出用)。"""
    visited = visited if visited is not None else set()
    avg_dev = []
    for ep in range(n_episodes):
        state = env.reset()
        eps = epsilon_for_episode(ep, decay_episodes, eps_start)
        done = False
        devs = []
        while not done:
            visited.add(state)
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = agent.best_action(state)
            next_state, reward, done, deviation = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            devs.append(deviation)
        avg_dev.append(float(np.mean(devs)))
    return avg_dev, visited


def build_teach_seeded_params(base_params):
    """5行動の基礎ネットワークから6行動ネットワークを構築する。隠れ層はそのまま
    コピーし、出力層は5行動分をコピー、6列目("teach")は"stay"列と厳密に
    同じ重みにする(タブラー版seed_teach_baselineの数学的に等価なNN版)。"""
    stay_idx = hp.ACTIONS.index("stay") if "stay" in hp.ACTIONS[:5] else 4
    new_params = MLPParamsGen(np.random.RandomState(0), STATE_DIM, 6)
    new_params.W1 = base_params.W1.copy()
    new_params.b1 = base_params.b1.copy()
    new_params.W2 = base_params.W2.copy()
    new_params.b2 = base_params.b2.copy()
    new_W3 = np.zeros((HIDDEN2, 6))
    new_b3 = np.zeros(6)
    new_W3[:, :5] = base_params.W3
    new_b3[:5] = base_params.b3
    new_W3[:, 5] = base_params.W3[:, stay_idx]
    new_b3[5] = base_params.b3[stay_idx]
    new_params.W3 = new_W3
    new_params.b3 = new_b3
    return new_params


def train_elder_with_teaching_nn(base_params, elder_seed, legacy_bonus, n_episodes, decay_episodes,
                                  transfer_count=TRANSFER_COUNT, blend=BLEND, base_visited=None):
    teach_env = HomeostasisEnv(random.Random(ELDER_SEED))
    elder_params0 = build_teach_seeded_params(base_params)
    elder = NNAgentGeneric(STATE_DIM, 6, seed=elder_seed, init_params=elder_params0)
    successor = SuccessorNet(seed=elder_seed + 500)

    elder_visited = set(base_visited) if base_visited else set()
    successor_touched = set()
    teach_counts, avg_dev = [], []

    for ep in range(n_episodes):
        state = teach_env.reset()
        eps = epsilon_for_episode(ep, decay_episodes)
        done = False
        devs, teach_count = [], 0
        while not done:
            elder_visited.add(state)
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = elder.best_action(state)

            next_state, reward, done, deviation = teach_env.step(action)

            if action == "teach":
                teach_count += 1
                reward = reward + legacy_bonus
                pool = list(elder_visited)
                sample_n = min(transfer_count, len(pool))
                sampled_states = random.sample(pool, sample_n)
                distill_step(elder.params, successor, sampled_states, blend)
                successor_touched.update(sampled_states)

            elder.update(state, action, reward, next_state, done)
            state = next_state
            devs.append(deviation)

        avg_dev.append(float(np.mean(devs)))
        teach_counts.append(teach_count)

    coverage = (len(successor_touched) / len(elder_visited)) if elder_visited else 0.0
    return elder, successor, avg_dev, teach_counts, coverage, elder_visited


def evaluate_successor_nn(successor_params, coverage, eval_seed):
    eps_start = 1.0 - 0.7 * coverage
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    env = HomeostasisEnv(random.Random(EVAL_SEED))
    agent = NNAgentGeneric(STATE_DIM, 6, seed=eval_seed, init_params=successor_params)
    avg_dev, _ = train_and_collect_states(env, agent, EVAL_EPISODES, EVAL_EPS_DECAY_EPISODES, eps_start)
    return avg_dev


# ------------------------------------------------------------
# 時間主導チャンク実行(45秒bash制限対応)
# ------------------------------------------------------------

def base_chunk(time_budget=40.0):
    """エルダーの基礎学習(3000ep、5行動、全run共通の土台)。1回だけ実行すればよい。"""
    state_file = "nn_legacy_base_state.pkl"
    t_start = time.time()
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env, agent = state["env"], state["agent"]
        avg_dev, visited = state["avg_dev"], state["visited"]
        ep_done = state["ep_done"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
    except FileNotFoundError:
        random.seed(ELDER_SEED)
        np.random.seed(ELDER_SEED)
        env = HomeostasisEnv(random.Random(ELDER_SEED))
        agent = NNAgentGeneric(STATE_DIM, 5, seed=ELDER_SEED)
        avg_dev, visited = [], set()
        ep_done = 0
        print("[legacy-nn base] 新規開始(エルダー基礎学習、3000ep、5行動)")

    n_episodes = ib.PARENT_EPISODES
    decay = ib.PARENT_EPS_DECAY_EPISODES
    while ep_done < n_episodes:
        state_s = env.reset()
        eps = epsilon_for_episode(ep_done, decay)
        done = False
        devs = []
        while not done:
            visited.add(state_s)
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = agent.best_action(state_s)
            next_state, reward, done, deviation = env.step(action)
            agent.update(state_s, action, reward, next_state, done)
            state_s = next_state
            devs.append(deviation)
        avg_dev.append(float(np.mean(devs)))
        ep_done += 1
        if time.time() - t_start > time_budget:
            break

    state = {
        "env": env, "agent": agent, "avg_dev": avg_dev, "visited": visited, "ep_done": ep_done,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    if ep_done >= n_episodes:
        with open("nn_legacy_base_params.pkl", "wb") as f:
            pickle.dump({"params": agent.params, "visited": visited}, f)
        print(f"[legacy-nn base] {ep_done}epで基礎学習完了(直近50ep平均逸脱={np.mean(avg_dev[-50:]):.4f}, "
              f"訪問状態数={len(visited)})。nn_legacy_base_params.pklに保存。")
    else:
        print(f"[legacy-nn base] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")


def run_chunk(legacy_bonus, seed, time_budget=40.0):
    tag = f"b{int(legacy_bonus)}_s{seed}"
    state_file = f"nn_legacy_run_result_{tag}.json"
    if os.path.exists(state_file):
        print(f"[legacy-nn {tag}] 既に完了済み(スキップ)")
        return

    with open("nn_legacy_base_params.pkl", "rb") as f:
        base_data = pickle.load(f)
    base_params = base_data["params"]
    base_visited = base_data["visited"]

    random.seed(seed)
    np.random.seed(seed)
    elder, successor, avg_dev, teach_counts, coverage, elder_visited = train_elder_with_teaching_nn(
        base_params, elder_seed=seed, legacy_bonus=legacy_bonus,
        n_episodes=ELDER_EPISODES, decay_episodes=ELDER_EPS_DECAY_EPISODES,
        base_visited=base_visited,
    )
    teach_rate = float(np.mean(teach_counts[-100:]) / hp.MAX_STEPS)

    succ_avg_dev = evaluate_successor_nn(successor.params, coverage, eval_seed=seed + 999)
    succ_first50 = float(np.mean(succ_avg_dev[:50]))

    print(f"[legacy-nn {tag}] teach頻度(終盤100ep)={teach_rate:.4f}, カバー率={coverage:.4f}, "
          f"サクセサー最初50ep平均逸脱={succ_first50:.4f}")

    result = {
        "legacy_bonus": legacy_bonus, "seed": seed,
        "teach_rate": teach_rate, "coverage": coverage, "succ_first50_dev": succ_first50,
    }
    with open(state_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[legacy-nn {tag}] 保存完了({state_file})")


def aggregate(use_n15=False):
    seeds = RUN_SEEDS_15 if use_n15 else RUN_SEEDS
    results = {}
    for bonus in LEGACY_BONUSES:
        teach_rates, coverages, succ_devs = [], [], []
        for seed in seeds:
            tag = f"b{int(bonus)}_s{seed}"
            with open(f"nn_legacy_run_result_{tag}.json") as f:
                d = json.load(f)
            teach_rates.append(d["teach_rate"])
            coverages.append(d["coverage"])
            succ_devs.append(d["succ_first50_dev"])
        results[bonus] = {
            "teach_rate_mean": float(np.mean(teach_rates)), "teach_rate_std": float(np.std(teach_rates)),
            "coverage_mean": float(np.mean(coverages)), "coverage_std": float(np.std(coverages)),
            "succ_first50_mean": float(np.mean(succ_devs)), "succ_first50_std": float(np.std(succ_devs)),
        }
        print(f"legacy_bonus={bonus}: teach頻度={results[bonus]['teach_rate_mean']:.4f}±{results[bonus]['teach_rate_std']:.4f}, "
              f"カバー率={results[bonus]['coverage_mean']:.4f}±{results[bonus]['coverage_std']:.4f}, "
              f"サクセサー最初50ep平均逸脱={results[bonus]['succ_first50_mean']:.4f}±{results[bonus]['succ_first50_std']:.4f}")

    tag = "n15" if use_n15 else "n3"
    out_json = f"nn_legacy_instinct_{tag}_results.json"
    with open(out_json, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"saved {out_json}")

    if use_n15:
        from scipy import stats
        b1_devs = []
        b3_devs = []
        b0_devs = []
        for seed in seeds:
            with open(f"nn_legacy_run_result_b0_s{seed}.json") as f:
                b0_devs.append(json.load(f)["succ_first50_dev"])
            with open(f"nn_legacy_run_result_b1_s{seed}.json") as f:
                b1_devs.append(json.load(f)["succ_first50_dev"])
            with open(f"nn_legacy_run_result_b3_s{seed}.json") as f:
                b3_devs.append(json.load(f)["succ_first50_dev"])
        t_stat, p_val = stats.ttest_ind(b3_devs, b1_devs)
        print(f"サクセサー最初50ep平均逸脱: bonus=3 vs bonus=1 の対応なしt検定: t={t_stat:.4f}, p={p_val:.4e}")
        t_stat2, p_val2 = stats.ttest_ind(b1_devs, b0_devs)
        print(f"サクセサー最初50ep平均逸脱: bonus=1 vs bonus=0 の対応なしt検定: t={t_stat2:.4f}, p={p_val2:.4e}")
        t_stat3, p_val3 = stats.ttest_ind(b3_devs, b0_devs)
        print(f"サクセサー最初50ep平均逸脱: bonus=3 vs bonus=0 の対応なしt検定: t={t_stat3:.4f}, p={p_val3:.4e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    bonuses = LEGACY_BONUSES
    teach_means = [results[b]["teach_rate_mean"] for b in bonuses]
    teach_stds = [results[b]["teach_rate_std"] for b in bonuses]
    cov_means = [results[b]["coverage_mean"] for b in bonuses]
    cov_stds = [results[b]["coverage_std"] for b in bonuses]
    succ_means = [results[b]["succ_first50_mean"] for b in bonuses]
    succ_stds = [results[b]["succ_first50_std"] for b in bonuses]

    axes[0].bar([str(b) for b in bonuses], teach_means, yerr=teach_stds, color="#4472C4")
    axes[0].set_xlabel("レガシー報酬(legacy_bonus)")
    axes[0].set_ylabel("teach行動の頻度(終盤100episode)")
    axes[0].set_title("レガシー報酬が強いほど教える頻度は上がるか(NN版)")

    axes[1].bar([str(b) for b in bonuses], cov_means, yerr=cov_stds, color="#9BBB59")
    axes[1].set_xlabel("レガシー報酬(legacy_bonus)")
    axes[1].set_ylabel("サクセサーのカバー率")
    axes[1].set_title("転写されたQ知識のカバー率(NN版)")

    axes[2].bar([str(b) for b in bonuses], succ_means, yerr=succ_stds, color="#C0504D")
    axes[2].set_xlabel("レガシー報酬(legacy_bonus)")
    axes[2].set_ylabel("サクセサー最初50ep平均逸脱(小さいほど良い)")
    axes[2].set_title("エルダーが教えた結果、サクセサーは早く恒常性を保てるか(NN版)")

    fig.suptitle(f"要件4後半 最小限NN移行: レガシー本能の用量反応(n={len(seeds)})")
    fig.tight_layout()
    out_png = f"legacy_instinct_nn_comparison_{tag}.png"
    fig.savefig(out_png, dpi=150)
    print(f"グラフを {out_png} に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "base_chunk":
        tb = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
        base_chunk(time_budget=tb)
    elif cmd == "run_chunk":
        legacy_bonus = float(sys.argv[2])
        seed = int(sys.argv[3])
        tb = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
        if "teach" not in hp.ACTIONS:
            hp.ACTIONS.append("teach")
        run_chunk(legacy_bonus, seed, time_budget=tb)
    elif cmd == "aggregate":
        if "teach" not in hp.ACTIONS:
            hp.ACTIONS.append("teach")
        use_n15 = len(sys.argv) > 2 and sys.argv[2] == "n15"
        aggregate(use_n15=use_n15)
