"""
感情AIプロジェクト フェーズ7 プロトタイプ: 要件7 NN版でgrokkingを誘発する難化実験
==========================================================

前回のNN移行実験(homeostasis_nn_prototype.py)は、grokking(長い停滞の後の
急激な汎化改善)を探索したが観測されなかった。ただし前回の実験はhomeostasis_
prototype.pyのデフォルト値(GRID_SIZE=8)をそのまま使っており、序盤(500ep
程度)から既に高い性能に達していた。ユーザーとの確認の結果、「4×4/5×5」と
いう当初の前提は誤りで実際は8×8だったこと、および難化のためグリッドサイズを
16×16まで拡大することを合意した。

**変更する軸は1つだけ**: `homeostasis_prototype.GRID_SIZE`を8→16に変更する
(community_signal_v2_prototype.pyがhp.GRID_SIZE=4で行っているのと同じ、
モジュール変数の書き換えパターン)。これにより探索すべき物理空間が4倍に
広がり、序盤の高epsilon探索フェーズでは資源(食料・シェルター)を偶然
見つけにくくなる。ただし状態表現自体(食料・シェルター・危険地帯への相対
方向、-1/0/1)はグリッドサイズに依存しないため、方策が一度「資源の方向へ
進む」という一般化された振る舞いを学習すれば、グリッドサイズに関わらず
同程度に機能しうる、という点は事前の理論的な注意点として記録しておく
(=もしgrokkingが起きなかった場合、この相対方向表現によるスケール不変性が
原因である可能性がある)。

**据え置く点**: ネットワーク構成(MLP 9→32→32→5、経験リプレイ+ターゲット
ネットワーク)、報酬設計、状態表現(相対方向・行動履歴8手)、複数マップ学習
(MULTI_MAP_SEEDS=[0,1,2,3]、UNSEEN_SEEDS=[5,6,7])、モニタの定義
(monitor_feature_richness_prototype.build_features、履歴8手)は
homeostasis_nn_prototype.pyと完全に同一に保つ。

**grokkingの判定基準**:
  1. 学習マップ(訓練分布内、held-out)のモニタ精度(相関)と、未経験マップの
     モニタ精度を、同じ学習曲線上で別々に追跡する。held-outが早期に頭打ちに
     なる一方、未経験マップは長く低いまま停滞し、その後急に追いつく、という
     形が見られるかを確認する。
  2. 精度の推移と並行して、方策の行動エントロピー(bit、多様なマップの
     ロールアウトから算出)と、訪問した(状態,行動)組の多様性(ユニーク数/
     総ステップ数)を記録し、精度の急上昇が方策の質的な変化(単純な丸暗記
     パターンからより一般化された振る舞いへの切り替わり)と同時に起きて
     いるかを確認する。

**学習量**: 20000epまで。45秒のbash制限に対応するため、前回と同じ時間主導の
チャンク実行方式を採用する。

**規模**: まずn=3(traj_seed=0,11,22)で明確な停滞→急上昇の兆候があるかを見る。
兆候があればn=15まで拡大する。

使い方:
  python3 homeostasis_nn_grokking_prototype.py sanity
  python3 homeostasis_nn_grokking_prototype.py chunk <traj_seed>
  python3 homeostasis_nn_grokking_prototype.py aggregate
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
hp.GRID_SIZE = 16  # 難化: 8(前回NN実験のデフォルト値)→16へ拡大(ユーザー確認済み)

import instinct_bias_prototype as ib
import homeostasis_nn_prototype as m  # DQNAgent, EvalPolicy, encode_state, forward, mfr等を再利用
import monitor_feature_richness_prototype as mfr

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAIN_SEED = m.TRAIN_SEED
MULTI_MAP_SEEDS = m.MULTI_MAP_SEEDS
UNSEEN_SEEDS = m.UNSEEN_SEEDS
ROLLOUT_EPS = m.ROLLOUT_EPS
ACTIONS = m.ACTIONS
DECAY_EPISODES = ib.PARENT_EPS_DECAY_EPISODES  # 2000、前回と同じ減衰スケジュールを維持

N_EPISODES_PER_MAP_CKPT = 40   # 前回のgrokking探索(Part C)と同じ、コスト削減
N_EPISODES_UNSEEN_CKPT = 25

CHECKPOINT_EPISODES = [250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000,
                        5000, 6500, 8000, 10000, 12500, 15000, 17500, 20000]

TRAJ_SEEDS_PRELIM = [0, 11, 22]
TRAJ_SEEDS_15 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22]


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


def collect_rollout_full(env, policy, n_episodes, eps):
    """monitor_feature_richness_prototype.collect_rollout_rawと同じレコードに加え、
    行動エントロピー・(状態,行動)多様性を計算するための補助データも同時に集める。"""
    records = []
    action_counter = {}
    state_action_set = set()
    n_steps = 0
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


def evaluate_checkpoint(traj_seed, ep_done, params):
    policy = m.EvalPolicy(params)
    train_records = []
    action_counter_total = {}
    n_unique_total = 0
    n_steps_total = 0
    for seed in MULTI_MAP_SEEDS:
        random.seed(traj_seed * 1000 + seed + ep_done)
        np.random.seed((traj_seed * 1000 + seed + ep_done) % (2**32 - 1))
        map_env = hp.HomeostasisEnv(random.Random(seed))
        recs, ac, n_unique, n_steps = collect_rollout_full(map_env, policy, N_EPISODES_PER_MAP_CKPT, ROLLOUT_EPS)
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
        recs, _, _, _ = collect_rollout_full(map_env, policy, N_EPISODES_UNSEEN_CKPT, ROLLOUT_EPS)
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


def chunk(traj_seed, time_budget=36.0):
    t_start = time.time()
    state_file = f"nn_grok_state_seed{traj_seed}.pkl"
    result_file = f"nn_grok_result_seed{traj_seed}.json"
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
        agent = m.DQNAgent(seed=traj_seed)
        ep_done = 0
        checkpoints_done = []
        print(f"[grok seed={traj_seed}] 新規開始(GRID_SIZE={hp.GRID_SIZE})")
        with open(result_file, "w") as f:
            json.dump({"traj_seed": traj_seed, "grid_size": hp.GRID_SIZE, "checkpoints": {}}, f)

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
            print(f"[grok seed={traj_seed}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")
            return
        eps = ib.epsilon_for_episode(ep_done, DECAY_EPISODES)
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
        if ep_done in checkpoint_set and ep_done not in checkpoints_done:
            metrics = evaluate_checkpoint(traj_seed, ep_done, agent.params.copy())
            checkpoints_done.append(ep_done)
            with open(result_file, "r") as f:
                result = json.load(f)
            result["checkpoints"][str(ep_done)] = metrics
            with open(result_file, "w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[grok seed={traj_seed}] {ep_done}ep: held-out={metrics['corr_holdout']:.4f}, "
                  f"未経験={metrics['corr_unseen_mean']:.4f}, 行動エントロピー={metrics['action_entropy_bits']:.4f}bit, "
                  f"状態行動多様性={metrics['state_action_diversity_frac']:.4f}")

    print(f"[grok seed={traj_seed}] target_end_ep={target_end}に到達、全チェックポイント評価済み")


def aggregate(seeds=None):
    seeds = seeds or TRAJ_SEEDS_PRELIM
    data = [json.load(open(f"nn_grok_result_seed{s}.json")) for s in seeds]
    print(f"=== NN版grokking誘発難化実験(GRID_SIZE=16、n={len(seeds)}平均±標準偏差) ===")
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

    with open("nn_grok_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    ns = sorted(summary.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].errorbar(ns, [summary[n]["holdout_mean"] for n in ns], yerr=[summary[n]["holdout_std"] for n in ns],
                      marker="o", label="held-out(学習分布内)", color="#4472C4")
    axes[0].errorbar(ns, [summary[n]["unseen_mean"] for n in ns], yerr=[summary[n]["unseen_std"] for n in ns],
                      marker="o", label="未経験マップ", color="#C0504D")
    axes[0].set_xlabel("エージェントの学習量(episode数)")
    axes[0].set_ylabel("相関係数")
    axes[0].set_title("held-out vs 未経験マップ精度の推移")
    axes[0].legend()

    ax2 = axes[1]
    ax2.plot(ns, [summary[n]["entropy_mean"] for n in ns], "o-", color="#2E7D32", label="行動エントロピー(bit)")
    ax2.set_xlabel("エージェントの学習量(episode数)")
    ax2.set_ylabel("行動エントロピー(bit)", color="#2E7D32")
    ax3 = ax2.twinx()
    ax3.plot(ns, [summary[n]["diversity_mean"] for n in ns], "s--", color="#7030A0", label="状態行動多様性")
    ax3.set_ylabel("状態行動多様性(ユニーク率)", color="#7030A0")
    axes[1].set_title("行動複雑性の推移")

    fig.suptitle("要件7: NN版grokking誘発難化実験(GRID_SIZE=16)")
    fig.tight_layout()
    fig.savefig("homeostasis_nn_grokking_comparison.png", dpi=150)
    print("グラフを homeostasis_nn_grokking_comparison.png に保存しました。")


def sanity():
    random.seed(0)
    np.random.seed(0)
    env = hp.HomeostasisEnv(random.Random(TRAIN_SEED))
    agent = m.DQNAgent(seed=0)
    t0 = time.time()
    for ep in range(50):
        eps = ib.epsilon_for_episode(ep, DECAY_EPISODES)
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
    print(f"GRID_SIZE={hp.GRID_SIZE}, 50ep学習時間: {dt:.3f}s ({dt/50*1000:.2f}ms/ep)")

    policy = m.EvalPolicy(agent.params)
    t1 = time.time()
    metrics = evaluate_checkpoint(999, 50, agent.params)
    print(f"チェックポイント評価(4マップ×{N_EPISODES_PER_MAP_CKPT}ep+3マップ×{N_EPISODES_UNSEEN_CKPT}ep): {time.time()-t1:.3f}s")
    print(metrics)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sanity":
        sanity()
    elif cmd == "chunk":
        chunk(int(sys.argv[2]))
    elif cmd == "aggregate":
        if len(sys.argv) > 2 and sys.argv[2] == "n15":
            aggregate(TRAJ_SEEDS_15)
        else:
            aggregate(TRAJ_SEEDS_PRELIM)
