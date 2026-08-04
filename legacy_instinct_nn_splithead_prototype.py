"""
感情AIプロジェクト フェーズ8 プロトタイプ: 要件4 破滅的干渉仮説の検証(教示ヘッド分離実験)
==========================================================

legacy_instinct_nn_prototype.py(要件4後半の最小限NN移行)で見つかった
「legacy_bonusを上げるほどサクセサーの独り立ち後の性能が悪化する」という
タブラー版と正反対の用量反応が、"移動行動と教示行動が同一の隠れ層を共有する
ことによる破滅的干渉(教示行動優位の経験リプレイが移動行動のQ値学習を
阻害する)"によるものかを直接検証する。

**変更する軸(アーキテクチャのみ)**:
移動行動(up/down/left/right/stay)のQ値を出力する既存のMLP(隠れ層32×32、
状態9次元→32→32→5)はそのまま(=legacy_instinct_nn_prototype.pyの基礎エルダー
nn_legacy_base_params.pklをそのまま再利用、再学習不要)にしつつ、教示行動
("teach")のQ値だけを、隠れ層を一切共有しない完全に独立した小さな専用
サブネットワーク(状態9次元→8→8→1、TeachNet)から出力する。行動選択は
両ネットワークの出力を連結した6次元Qベクトル(move_q[0:5], teach_q)への
通常のε-greedyで、環境・報酬・状態表現・学習量(ELDER_EPISODES=500など)は
legacy_instinct_nn_prototype.pyと完全に同一。

**必然的な設計変更点(ヘッド分離により生じた追加のNN翻訳)**:

1. 教示行動の初期値バイアス: 共有ヘッド版はteach列の重みをstay列と文字通り
   同じ値にコピーすることで、あらゆる状態についてQ(s,teach)=Q(s,stay)を
   厳密に(数学的に)成立させていた(seed_teach_baselineの等価な再現)。しかし
   TeachNetは移動ネットとは異なる独立した重み空間を持つため、重みコピーでは
   この等価性を再現できない。代わりに、基礎エルダーが訪れた状態のプールから
   サンプルした状態についてTeachNetをQ(s,stay)へ回帰的に事前学習(200ステップ、
   warm_start_teach_net)することで、近似的に同じ初期条件(楽観的初期化バイアス
   の回避)を再現した。これは分離アーキテクチャで可能な最も忠実な代替である。

2. 転写(distillation)のヘッド分離: 元のdistill_stepは1つの6次元Qベクトルを
   一括で回帰していたが、分離版はmove部分(5次元)とteach部分(1次元)を、
   それぞれ対応するサブネットワークに独立して回帰する(distill_step_split)。
   これにより、転写フェーズ自体も破滅的干渉の再導入経路にならないよう設計した
   (teachの転写誤差が移動ネットの重みを更新することは一切ない)。

3. 経験リプレイの分離: 移動行動が選ばれた遷移はmove_buffer、教示行動が
   選ばれた遷移はteach_bufferにそれぞれ格納し、勾配更新もバッファ単位で
   完全に独立して行う(train_move_step/train_teach_stepが別々のネットワークの
   重みのみを更新)。TDターゲット計算時のmax_q_next(次状態での最大Q値)だけは、
   両ネットワークのtarget重みを連結した6次元ベクトルから計算する(環境の
   意思決定は依然として6行動全体のgreedy方策に基づくため、これは
   アーキテクチャに依らず必要)。

**据え置く点**: 環境(HomeostasisEnv)・報酬設計(legacy_bonus∈{0,1,3})・
状態表現(9次元)・学習量(ELDER_EPISODES=500、EVAL_EPISODES=300)・転写個数
(TRANSFER_COUNT=3)・混合率(BLEND=0.3)・評価時の探索率補正は
legacy_instinct_nn_prototype.pyと完全に同一。

**判定基準**: ヘッド分離後、用量反応がタブラー版と同じ正の方向(bonusを
上げるほどサクセサーの独り立ち後の逸脱が下がる=改善する)に戻れば、
共有隠れ層による破滅的干渉が原因であったことがほぼ確定する。逆転が
解消されなければ、別の原因(NNの学習率・報酬スケールの違いなど)を疑う。

規模: まずn=3(RUN_SEEDS、共有ヘッド版と同一の100/200/300)。方向が戻れば
n=15まで拡大して統計的に確認する。

使い方:
  python3 legacy_instinct_nn_splithead_prototype.py run_chunk <legacy_bonus> <seed> [time_budget]
  python3 legacy_instinct_nn_splithead_prototype.py aggregate [n15]
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
import homeostasis_nn_prototype as m
from legacy_instinct_nn_prototype import (
    MLPParamsGen, epsilon_for_episode, train_and_collect_states,
    ELDER_SEED, EVAL_SEED, STATE_DIM, ELDER_EPISODES, ELDER_EPS_DECAY_EPISODES,
    EVAL_EPISODES, EVAL_EPS_DECAY_EPISODES, LEGACY_BONUSES, TRANSFER_COUNT, BLEND,
    RUN_SEEDS, RUN_SEEDS_15,
)

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

ACTIONS = hp.ACTIONS  # 共有される可変リスト。teach追加後は全モジュールに反映される。
TEACH_HIDDEN = 8
WARM_START_STEPS = 200
WARM_START_BATCH = 32
WARM_START_POOL = 300


def _regress_backprop(params, adam, cache, pred, target):
    """distill_step/warm_start_teach_netの両方で使う共通の回帰勾配更新。"""
    n = pred.shape[0]
    d_loss = 2.0 * (pred - target) / n
    X_in, z1, h1, z2, h2 = cache
    dW3 = h2.T @ d_loss
    db3 = d_loss.sum(axis=0)
    dh2 = d_loss @ params.W3.T
    dz2 = dh2 * (z2 > 0)
    dW2 = h1.T @ dz2
    db2 = dz2.sum(axis=0)
    dh1 = dz2 @ params.W2.T
    dz1 = dh1 * (z1 > 0)
    dW1 = X_in.T @ dz1
    db1 = dz1.sum(axis=0)
    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
    m.adam_step(params, grads, adam, lr=m.LR)


def _dqn_train_step_net(params, adam, S, A, R, max_q_next, D, batch_size):
    """1つのサブネットワークだけを更新する汎用DQN勾配ステップ。
    max_q_nextは呼び出し側が(move+teach連結ベクトルの)最大値として渡す。"""
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


def warm_start_teach_net(move_params, teach_params, teach_adam, states, seed=0,
                          n_steps=WARM_START_STEPS, batch_size=WARM_START_BATCH):
    """TeachNetをQ(s,"stay")へ回帰的に事前学習する。共有ヘッド版のseed_teach_baseline
    (Q(s,teach)=Q(s,stay)の厳密な重みコピー)を、独立ネットワークで可能な範囲で
    近似的に再現する代替(設計変更点1)。"""
    if not states:
        return
    rng = np.random.RandomState(seed)
    stay_idx = ACTIONS.index("stay")
    X_all = np.stack([m.encode_state(s) for s in states])
    move_q_all, _ = m.forward(move_params, X_all)
    target_all = move_q_all[:, stay_idx:stay_idx + 1]
    n = len(states)
    for _ in range(n_steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        X = X_all[idx]
        target = target_all[idx]
        q_pred, cache = m.forward(teach_params, X)
        _regress_backprop(teach_params, teach_adam, cache, q_pred, target)


class NNAgentSplit:
    """移動5行動を出力するmove_paramsと、教示1行動だけを出力する独立した
    小さなteach_paramsを両方保持し、6次元連結Qベクトルでε-greedy行動選択する
    エージェント。interfaceはNNAgentGenericと同一(best_action, update)。"""

    def __init__(self, state_dim, seed=0, init_move_params=None, init_teach_params=None,
                 teach_hidden=TEACH_HIDDEN):
        rng = np.random.RandomState(seed)
        self.state_dim = state_dim
        self.move_params = init_move_params.copy() if init_move_params is not None else MLPParamsGen(rng, state_dim, 5)
        self.move_target = self.move_params.copy()
        self.move_adam = m.AdamState(self.move_params)
        self.move_buffer = m.ReplayBuffer(m.BUFFER_CAPACITY, state_dim)

        self.teach_params = init_teach_params.copy() if init_teach_params is not None else MLPParamsGen(
            rng, state_dim, 1, hidden1=teach_hidden, hidden2=teach_hidden)
        self.teach_target = self.teach_params.copy()
        self.teach_adam = m.AdamState(self.teach_params)
        self.teach_buffer = m.ReplayBuffer(m.BUFFER_CAPACITY, state_dim)

        self.np_rng = np.random.RandomState(seed + 777)
        self.step_count = 0

    def q_values(self, state):
        x = m.encode_state(state)[None, :]
        q_move, _ = m.forward(self.move_params, x)
        q_teach, _ = m.forward(self.teach_params, x)
        return np.concatenate([q_move[0], q_teach[0]])

    def best_action(self, state):
        q = self.q_values(state)
        return ACTIONS[int(np.argmax(q))]

    def _combined_max_q_next(self, Sn):
        q_move, _ = m.forward(self.move_target, Sn)
        q_teach, _ = m.forward(self.teach_target, Sn)
        combined = np.concatenate([q_move, q_teach], axis=1)
        return np.max(combined, axis=1)

    def update(self, state, action, reward, next_state, done):
        x = m.encode_state(state)
        xn = m.encode_state(next_state)
        a_idx = ACTIONS.index(action)
        self.step_count += 1
        if a_idx < 5:
            self.move_buffer.add(x, a_idx, reward, xn, 1.0 if done else 0.0)
        else:
            self.teach_buffer.add(x, 0, reward, xn, 1.0 if done else 0.0)

        if self.move_buffer.size >= m.BATCH_SIZE:
            S, A, R, Sn, D = self.move_buffer.sample(m.BATCH_SIZE, self.np_rng)
            max_q_next = self._combined_max_q_next(Sn)
            _dqn_train_step_net(self.move_params, self.move_adam, S, A, R, max_q_next, D, m.BATCH_SIZE)
        if self.teach_buffer.size >= m.BATCH_SIZE:
            S, A, R, Sn, D = self.teach_buffer.sample(m.BATCH_SIZE, self.np_rng)
            max_q_next = self._combined_max_q_next(Sn)
            _dqn_train_step_net(self.teach_params, self.teach_adam, S, A, R, max_q_next, D, m.BATCH_SIZE)

        if self.step_count % m.TARGET_SYNC_STEPS == 0:
            self.move_target = self.move_params.copy()
            self.teach_target = self.teach_params.copy()


class SuccessorNetSplit:
    """教示フェーズ中は環境と一切相互作用せず、distill_step_splitによる回帰更新
    だけを受け取る受動的な分離ネットワーク(タブラー版successor=白紙QLearningAgent
    に相当)。move/teachとも完全にランダム初期化からスタート(warm startなし)。"""

    def __init__(self, seed=0, teach_hidden=TEACH_HIDDEN):
        rng = np.random.RandomState(seed)
        self.move_params = MLPParamsGen(rng, STATE_DIM, 5)
        self.move_adam = m.AdamState(self.move_params)
        self.teach_params = MLPParamsGen(rng, STATE_DIM, 1, hidden1=teach_hidden, hidden2=teach_hidden)
        self.teach_adam = m.AdamState(self.teach_params)


def distill_step_split(elder, successor, states, blend):
    """move部分とteach部分を独立に回帰(設計変更点2)。"""
    X = np.stack([m.encode_state(s) for s in states])

    elder_move_q, _ = m.forward(elder.move_params, X)
    succ_move_q, cache_move = m.forward(successor.move_params, X)
    target_move = succ_move_q + blend * (elder_move_q - succ_move_q)
    _regress_backprop(successor.move_params, successor.move_adam, cache_move, succ_move_q, target_move)

    elder_teach_q, _ = m.forward(elder.teach_params, X)
    succ_teach_q, cache_teach = m.forward(successor.teach_params, X)
    target_teach = succ_teach_q + blend * (elder_teach_q - succ_teach_q)
    _regress_backprop(successor.teach_params, successor.teach_adam, cache_teach, succ_teach_q, target_teach)


def train_elder_with_teaching_split_init(base_params, elder_seed, base_visited=None):
    """teachフェーズの初期化(エージェント構築・warm start・successor構築)だけを行う。
    時間主導チャンク実行(run_chunk)から呼ばれ、以後はエピソードループを
    チャンク単位で進める。"""
    elder = NNAgentSplit(STATE_DIM, seed=elder_seed, init_move_params=base_params)
    pool0 = list(base_visited) if base_visited else []
    warm_states = random.sample(pool0, min(len(pool0), WARM_START_POOL)) if pool0 else []
    warm_start_teach_net(elder.move_params, elder.teach_params, elder.teach_adam, warm_states, seed=elder_seed)
    elder.teach_target = elder.teach_params.copy()
    successor = SuccessorNetSplit(seed=elder_seed + 500)
    elder_visited = set(base_visited) if base_visited else set()
    successor_touched = set()
    return elder, successor, elder_visited, successor_touched


def train_elder_teaching_episode(teach_env, elder, successor, elder_visited, successor_touched,
                                  eps, legacy_bonus, transfer_count, blend):
    """teachフェーズの1エピソード分だけ進める(チャンク実行用に切り出し)。"""
    state = teach_env.reset()
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
            distill_step_split(elder, successor, sampled_states, blend)
            successor_touched.update(sampled_states)

        elder.update(state, action, reward, next_state, done)
        state = next_state
        devs.append(deviation)

    return float(np.mean(devs)), teach_count


def eval_one_episode(env, agent, eps):
    """train_and_collect_states相当の1エピソード分(チャンク実行用に切り出し)。"""
    state = env.reset()
    done = False
    devs = []
    while not done:
        if random.random() < eps:
            action = random.choice(ACTIONS)
        else:
            action = agent.best_action(state)
        next_state, reward, done, deviation = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        devs.append(deviation)
    return float(np.mean(devs))


def run_chunk(legacy_bonus, seed, time_budget=35.0):
    """時間主導チャンク実行(bashの45秒制限対応)。phase="teach"→"eval"の順に
    エピソード単位でピクル保存・自動再開する(legacy_multigen_nn_prototype.pyの
    gen_chunkと同じ設計)。"""
    tag = f"b{int(legacy_bonus)}_s{seed}"
    result_file = f"nn_legacy_split_run_result_{tag}.json"
    if os.path.exists(result_file):
        print(f"[legacy-split {tag}] 既に完了済み(スキップ)")
        return
    state_file = f"nn_legacy_split_state_{tag}.pkl"

    t_start = time.time()
    try:
        with open(state_file, "rb") as f:
            st = pickle.load(f)
        random.setstate(st["random_state"])
        np.random.set_state(st["np_random_state"])
        print(f"[legacy-split {tag}] 再開(phase={st['phase']}, teach_ep={st['teach_ep']}/{ELDER_EPISODES}, "
              f"eval_ep={st['eval_ep']}/{EVAL_EPISODES})")
    except FileNotFoundError:
        with open("nn_legacy_base_params.pkl", "rb") as f:
            base_data = pickle.load(f)
        base_params = base_data["params"]
        base_visited = base_data["visited"]
        random.seed(seed)
        np.random.seed(seed)
        elder, successor, elder_visited, successor_touched = train_elder_with_teaching_split_init(
            base_params, elder_seed=seed, base_visited=base_visited)
        st = {
            "phase": "teach", "teach_ep": 0, "eval_ep": 0,
            "teach_env": HomeostasisEnv(random.Random(ELDER_SEED)),
            "elder": elder, "successor": successor,
            "elder_visited": elder_visited, "successor_touched": successor_touched,
            "avg_dev": [], "teach_counts": [], "succ_avg_dev": [],
            "coverage": 0.0, "eval_agent": None,
        }
        print(f"[legacy-split {tag}] 新規開始")

    while st["phase"] == "teach" and st["teach_ep"] < ELDER_EPISODES:
        eps = epsilon_for_episode(st["teach_ep"], ELDER_EPS_DECAY_EPISODES)
        dev, teach_count = train_elder_teaching_episode(
            st["teach_env"], st["elder"], st["successor"], st["elder_visited"], st["successor_touched"],
            eps, legacy_bonus, TRANSFER_COUNT, BLEND)
        st["avg_dev"].append(dev)
        st["teach_counts"].append(teach_count)
        st["teach_ep"] += 1
        if time.time() - t_start > time_budget:
            break

    if st["phase"] == "teach" and st["teach_ep"] >= ELDER_EPISODES:
        st["coverage"] = (len(st["successor_touched"]) / len(st["elder_visited"])) if st["elder_visited"] else 0.0
        eps_start = 1.0 - 0.7 * st["coverage"]
        random.seed(EVAL_SEED)
        np.random.seed(EVAL_SEED)
        st["eval_env"] = HomeostasisEnv(random.Random(EVAL_SEED))
        st["eval_agent"] = NNAgentSplit(STATE_DIM, seed=seed + 999,
                                         init_move_params=st["successor"].move_params,
                                         init_teach_params=st["successor"].teach_params)
        st["eval_eps_start"] = eps_start
        st["phase"] = "eval"
        print(f"[legacy-split {tag}] 教示フェーズ完了(カバー率={st['coverage']:.4f})、評価フェーズへ")

    if st["phase"] == "eval":
        while st["eval_ep"] < EVAL_EPISODES:
            eps = epsilon_for_episode(st["eval_ep"], EVAL_EPS_DECAY_EPISODES, eps_start=st["eval_eps_start"])
            dev = eval_one_episode(st["eval_env"], st["eval_agent"], eps)
            st["succ_avg_dev"].append(dev)
            st["eval_ep"] += 1
            if time.time() - t_start > time_budget:
                break

    if st["phase"] == "eval" and st["eval_ep"] >= EVAL_EPISODES:
        teach_rate = float(np.mean(st["teach_counts"][-100:]) / hp.MAX_STEPS)
        succ_first50 = float(np.mean(st["succ_avg_dev"][:50]))
        print(f"[legacy-split {tag}] teach頻度(終盤100ep)={teach_rate:.4f}, カバー率={st['coverage']:.4f}, "
              f"サクセサー最初50ep平均逸脱={succ_first50:.4f}")
        result = {
            "legacy_bonus": legacy_bonus, "seed": seed,
            "teach_rate": teach_rate, "coverage": st["coverage"], "succ_first50_dev": succ_first50,
        }
        with open(result_file, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        try:
            if os.path.exists(state_file):
                os.remove(state_file)
        except OSError:
            pass
        print(f"[legacy-split {tag}] 保存完了({result_file})")
        return

    st["random_state"] = random.getstate()
    st["np_random_state"] = np.random.get_state()
    with open(state_file, "wb") as f:
        pickle.dump(st, f)
    print(f"[legacy-split {tag}] 時間予算({time_budget}s)到達、phase={st['phase']}, "
          f"teach_ep={st['teach_ep']}/{ELDER_EPISODES}, eval_ep={st['eval_ep']}/{EVAL_EPISODES}")


def aggregate(use_n15=False):
    seeds = RUN_SEEDS_15 if use_n15 else RUN_SEEDS
    results = {}
    for bonus in LEGACY_BONUSES:
        teach_rates, coverages, succ_devs = [], [], []
        for seed in seeds:
            tag = f"b{int(bonus)}_s{seed}"
            with open(f"nn_legacy_split_run_result_{tag}.json") as f:
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
    out_json = f"nn_legacy_instinct_split_{tag}_results.json"
    with open(out_json, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"saved {out_json}")

    if use_n15:
        from scipy import stats
        b0_devs, b1_devs, b3_devs = [], [], []
        for seed in seeds:
            with open(f"nn_legacy_split_run_result_b0_s{seed}.json") as f:
                b0_devs.append(json.load(f)["succ_first50_dev"])
            with open(f"nn_legacy_split_run_result_b1_s{seed}.json") as f:
                b1_devs.append(json.load(f)["succ_first50_dev"])
            with open(f"nn_legacy_split_run_result_b3_s{seed}.json") as f:
                b3_devs.append(json.load(f)["succ_first50_dev"])
        t01, p01 = stats.ttest_ind(b1_devs, b0_devs)
        t13, p13 = stats.ttest_ind(b3_devs, b1_devs)
        t03, p03 = stats.ttest_ind(b3_devs, b0_devs)
        print(f"bonus=1 vs bonus=0: t={t01:.4f}, p={p01:.4e}")
        print(f"bonus=3 vs bonus=1: t={t13:.4f}, p={p13:.4e}")
        print(f"bonus=3 vs bonus=0: t={t03:.4f}, p={p03:.4e}")

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
    axes[0].set_title("レガシー報酬が強いほど教える頻度は上がるか(ヘッド分離版)")

    axes[1].bar([str(b) for b in bonuses], cov_means, yerr=cov_stds, color="#9BBB59")
    axes[1].set_xlabel("レガシー報酬(legacy_bonus)")
    axes[1].set_ylabel("サクセサーのカバー率")
    axes[1].set_title("転写されたQ知識のカバー率(ヘッド分離版)")

    axes[2].bar([str(b) for b in bonuses], succ_means, yerr=succ_stds, color="#C0504D")
    axes[2].set_xlabel("レガシー報酬(legacy_bonus)")
    axes[2].set_ylabel("サクセサー最初50ep平均逸脱(小さいほど良い)")
    axes[2].set_title("エルダーが教えた結果、サクセサーは早く恒常性を保てるか(ヘッド分離版)")

    fig.suptitle(f"要件4 教示ヘッド分離実験: 破滅的干渉仮説の検証(n={len(seeds)})")
    fig.tight_layout()
    out_png = f"legacy_instinct_nn_splithead_comparison_{tag}.png"
    fig.savefig(out_png, dpi=150)
    print(f"グラフを {out_png} に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if "teach" not in hp.ACTIONS:
        hp.ACTIONS.append("teach")
    if cmd == "run_chunk":
        legacy_bonus = float(sys.argv[2])
        seed = int(sys.argv[3])
        tb = float(sys.argv[4]) if len(sys.argv) > 4 else 35.0
        run_chunk(legacy_bonus, seed, time_budget=tb)
    elif cmd == "aggregate":
        use_n15 = len(sys.argv) > 2 and sys.argv[2] == "n15"
        aggregate(use_n15=use_n15)
