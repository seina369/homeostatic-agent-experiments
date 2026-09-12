"""
感情AIプロジェクト フェーズ4 プロトタイプ: 要件4後半 レガシー本能 複数世代連鎖の最小限NN移行
==========================================================

legacy_multigen_prototype.py(エルダー→サクセサーの引き継ぎを5世代連鎖させ、
3系統×5世代=15世代で完全収束13/15(86.7%)という頑健性を確認したタブラー版)を
基準に、legacy_instinct_nn_prototype.pyで構築した「教示のNN移行」設計
(6行動ネットワーク・状態単位のQベクトル混合コピー・訪問状態集合によるカバー率)を
そのまま再利用し、世代を跨いだ連鎖でも同程度の頑健性が保たれるかを検証する。

**構造(タブラー版run_generationと対応)**:
  1. 教示フェーズ(TEACH_EPISODES=500): 前世代から引き継いだエルダーのネットワークの
     "teach"列を"stay"列で上書き(タブラー版seed_teach_baselineが毎世代実行される
     のと同じく、前世代の教示フェーズで多少変化したteach列も含めて無条件にリセットする)
     した上で、legacy_bonus=3.0(タブラー版で最も恩恵が大きかった値)固定で教示学習し、
     サクセサーへ状態単位のQベクトル混合コピーで転写する。
  2. 育成フェーズ(GROW_EPISODES=3000、要件3プロトタイプ(instinct_bias)の収束実績に
     合わせた学習量): 転写を受けたサクセサーを、カバー率に応じた探索率補正
     (eps_start=1-0.7*coverage)付きで別マップ(GROW_MAP_SEED)にて独り立ちさせる。
     育ちきったネットワークが次世代のエルダーになる。
  3. これをN_GENERATIONS=5世代、LINEAGES=[0,1,2]の3系統で繰り返す。

訪問状態の集合(elder_visited)は世代を跨いで累積する(タブラー版のelder.qが
世代を跨いで同じ辞書オブジェクトとして引き継がれ、キーが増え続けるのと同じ設計)。

規模: 3系統×5世代(タブラー版と同一規模)。タブラー版と大きく違う結果が出た
場合のみ、系統数の追加拡大を検討する。

使い方:
  python3 legacy_multigen_nn_prototype.py init <lineage> [time_budget]
  python3 legacy_multigen_nn_prototype.py gen_chunk <lineage> <gen> [time_budget]
  python3 legacy_multigen_nn_prototype.py aggregate_all <lineage...>
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
from legacy_instinct_nn_prototype import (
    MLPParamsGen, NNAgentGeneric, SuccessorNet, distill_step,
    epsilon_for_episode, STATE_DIM, HIDDEN1, HIDDEN2,
)

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

ACTIONS = hp.ACTIONS

BASE_MAP_SEED = 0
GROW_MAP_SEED = 2
LEGACY_BONUS = 3.0
TEACH_EPISODES = 500
TEACH_DECAY = 300
GROW_EPISODES = 3000
GROW_DECAY = 2000
N_GENERATIONS = 5
TRANSFER_COUNT = 3
BLEND = 0.3

LINEAGES = [0, 1, 2]


def reseed_teach_column(params):
    """タブラー版seed_teach_baselineのNN版。隠れ層はそのまま、出力層の
    "teach"列だけを"stay"列の値で無条件に上書きする(5行動→6行動への
    拡張が必要な場合はそれも行う)。前世代の教示フェーズでteach列が
    変化していても、このリセットにより毎世代同じ楽観的初期化バイアス
    回避が適用される。"""
    stay_idx = ACTIONS.index("stay")
    if params.n_actions == 5:
        new_params = MLPParamsGen(np.random.RandomState(0), STATE_DIM, 6)
        new_params.W1 = params.W1.copy()
        new_params.b1 = params.b1.copy()
        new_params.W2 = params.W2.copy()
        new_params.b2 = params.b2.copy()
        new_W3 = np.zeros((HIDDEN2, 6))
        new_b3 = np.zeros(6)
        new_W3[:, :5] = params.W3
        new_b3[:5] = params.b3
        new_W3[:, 5] = params.W3[:, stay_idx]
        new_b3[5] = params.b3[stay_idx]
        new_params.W3 = new_W3
        new_params.b3 = new_b3
        return new_params
    else:
        new_params = params.copy()
        teach_idx = ACTIONS.index("teach")
        new_params.W3[:, teach_idx] = params.W3[:, stay_idx]
        new_params.b3[teach_idx] = params.b3[stay_idx]
        return new_params


def init_chunk(lineage, time_budget=40.0):
    """創始エルダー(世代0)の基礎学習(3000ep、5行動)。lineageごとに1回だけ実行。"""
    seed_base = lineage * 100000
    state_file = f"nn_multigen_init_state_L{lineage}.pkl"
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
        random.seed(seed_base)
        np.random.seed(seed_base)
        env = HomeostasisEnv(random.Random(BASE_MAP_SEED))
        agent = NNAgentGeneric(STATE_DIM, 5, seed=seed_base)
        avg_dev, visited = [], set()
        ep_done = 0
        print(f"[multigen-nn L{lineage}] 創始エルダー新規開始")

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
                action = random.choice(hp.ACTIONS[:5])
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
        with open(f"nn_multigen_elder_L{lineage}_gen0.pkl", "wb") as f:
            pickle.dump({"params": agent.params, "visited": visited}, f)
        print(f"[multigen-nn L{lineage}] 創始エルダー学習完了(直近50ep平均逸脱={np.mean(avg_dev[-50:]):.4f}, "
              f"訪問状態数={len(visited)})")
    else:
        print(f"[multigen-nn L{lineage}] 時間予算({time_budget}s)到達、{ep_done}epで一旦終了")


def gen_chunk(lineage, gen, time_budget=40.0):
    seed_base = lineage * 100000 + gen * 100
    state_file = f"nn_multigen_gen_state_L{lineage}_gen{gen}.pkl"
    result_file = f"nn_multigen_record_L{lineage}_gen{gen}.json"
    if os.path.exists(result_file):
        print(f"[multigen-nn L{lineage} gen{gen}] 既に完了済み(スキップ)")
        return

    t_start = time.time()
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        phase = state["phase"]
        elder = state["elder"]
        successor = state["successor"]
        elder_visited = state["elder_visited"]
        successor_touched = state["successor_touched"]
        teach_ep_done = state["teach_ep_done"]
        teach_avg_dev = state["teach_avg_dev"]
        teach_counts = state["teach_counts"]
        teach_env = state["teach_env"]
        grow_agent = state.get("grow_agent")
        grow_env = state.get("grow_env")
        grow_ep_done = state.get("grow_ep_done", 0)
        grow_avg_dev = state.get("grow_avg_dev", [])
        coverage = state.get("coverage")
        eps_start_grow = state.get("eps_start_grow")
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
    except FileNotFoundError:
        with open(f"nn_multigen_elder_L{lineage}_gen{gen - 1}.pkl", "rb") as f:
            prev = pickle.load(f)
        random.seed(seed_base)
        np.random.seed(seed_base)
        elder_params0 = reseed_teach_column(prev["params"])
        elder = NNAgentGeneric(STATE_DIM, 6, seed=seed_base + 1, init_params=elder_params0)
        successor = SuccessorNet(seed=seed_base + 2)
        elder_visited = set(prev["visited"])
        successor_touched = set()
        teach_env = HomeostasisEnv(random.Random(BASE_MAP_SEED))
        teach_ep_done = 0
        teach_avg_dev, teach_counts = [], []
        grow_agent, grow_env, grow_ep_done, grow_avg_dev, coverage, eps_start_grow = None, None, 0, [], None, None
        phase = "teach"
        print(f"[multigen-nn L{lineage} gen{gen}] 新規開始(教示フェーズ)")

    if phase == "teach":
        while teach_ep_done < TEACH_EPISODES:
            state_s = teach_env.reset()
            eps = epsilon_for_episode(teach_ep_done, TEACH_DECAY)
            done = False
            devs, teach_count = [], 0
            while not done:
                elder_visited.add(state_s)
                if random.random() < eps:
                    action = random.choice(ACTIONS)
                else:
                    action = elder.best_action(state_s)
                next_state, reward, done, deviation = teach_env.step(action)
                if action == "teach":
                    teach_count += 1
                    reward = reward + LEGACY_BONUS
                    pool = list(elder_visited)
                    sample_n = min(TRANSFER_COUNT, len(pool))
                    sampled_states = random.sample(pool, sample_n)
                    distill_step(elder.params, successor, sampled_states, BLEND)
                    successor_touched.update(sampled_states)
                elder.update(state_s, action, reward, next_state, done)
                state_s = next_state
                devs.append(deviation)
            teach_avg_dev.append(float(np.mean(devs)))
            teach_counts.append(teach_count)
            teach_ep_done += 1
            if time.time() - t_start > time_budget:
                break

        if teach_ep_done >= TEACH_EPISODES:
            coverage = (len(successor_touched) / len(elder_visited)) if elder_visited else 0.0
            eps_start_grow = 1.0 - 0.7 * coverage
            random.seed(seed_base + 3)
            np.random.seed(seed_base + 3)
            grow_env = HomeostasisEnv(random.Random(GROW_MAP_SEED))
            grow_agent = NNAgentGeneric(STATE_DIM, 6, seed=seed_base + 4, init_params=successor.params)
            phase = "grow"
            print(f"[multigen-nn L{lineage} gen{gen}] 教示フェーズ完了(teach頻度(終盤100ep)="
                  f"{np.mean(teach_counts[-100:]) / hp.MAX_STEPS:.4f}, カバー率={coverage:.4f})、育成フェーズへ")

    if phase == "grow":
        while grow_ep_done < GROW_EPISODES:
            state_s = grow_env.reset()
            eps = epsilon_for_episode(grow_ep_done, GROW_DECAY, eps_start=eps_start_grow)
            done = False
            devs = []
            while not done:
                elder_visited.add(state_s)
                if random.random() < eps:
                    action = random.choice(ACTIONS)
                else:
                    action = grow_agent.best_action(state_s)
                next_state, reward, done, deviation = grow_env.step(action)
                grow_agent.update(state_s, action, reward, next_state, done)
                state_s = next_state
                devs.append(deviation)
            grow_avg_dev.append(float(np.mean(devs)))
            grow_ep_done += 1
            if time.time() - t_start > time_budget:
                break

    state = {
        "phase": phase, "elder": elder, "successor": successor,
        "elder_visited": elder_visited, "successor_touched": successor_touched,
        "teach_ep_done": teach_ep_done, "teach_avg_dev": teach_avg_dev, "teach_counts": teach_counts,
        "teach_env": teach_env,
        "grow_agent": grow_agent, "grow_env": grow_env, "grow_ep_done": grow_ep_done,
        "grow_avg_dev": grow_avg_dev, "coverage": coverage, "eps_start_grow": eps_start_grow,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    if phase == "grow" and grow_ep_done >= GROW_EPISODES:
        teach_rate = float(np.mean(teach_counts[-100:]) / hp.MAX_STEPS)
        head_start_first50 = float(np.mean(grow_avg_dev[:50]))
        grown_last50 = float(np.mean(grow_avg_dev[-50:]))
        record = {
            "lineage": lineage, "generation": gen,
            "coverage": coverage, "teach_rate": teach_rate,
            "head_start_first50_dev": head_start_first50,
            "grown_last50_dev": grown_last50,
            "n_visited_states": len(elder_visited),
        }
        print(f"[multigen-nn L{lineage}] 世代{gen}: {json.dumps(record, ensure_ascii=False)}")
        with open(f"nn_multigen_elder_L{lineage}_gen{gen}.pkl", "wb") as f:
            pickle.dump({"params": grow_agent.params, "visited": elder_visited}, f)
        with open(result_file, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        print(f"[multigen-nn L{lineage}] 世代{gen}完了・保存({result_file})")
    else:
        print(f"[multigen-nn L{lineage} gen{gen}] 時間予算({time_budget}s)到達、"
              f"phase={phase}, teach_ep={teach_ep_done}/{TEACH_EPISODES}, grow_ep={grow_ep_done}/{GROW_EPISODES}")


def load_records(lineage):
    records = []
    for gen in range(1, N_GENERATIONS + 1):
        with open(f"nn_multigen_record_L{lineage}_gen{gen}.json") as f:
            records.append(json.load(f))
    return records


def aggregate_all(lineages):
    all_records = []
    for lineage in lineages:
        all_records.extend(load_records(lineage))

    print("=== 要件4後半 NN移行: 複数世代連鎖 全lineage集計 ===")
    converged = [r for r in all_records if r["grown_last50_dev"] < 0.2]
    stuck = [r for r in all_records if r["grown_last50_dev"] >= 0.2]
    print(f"総世代数={len(all_records)}, 完全収束(逸脱<0.2)={len(converged)}件, "
          f"不完全収束(逸脱>=0.2)={len(stuck)}件, 収束率={len(converged)/len(all_records):.1%}")

    head_start_vals = [r["head_start_first50_dev"] for r in all_records]
    grown_vals = [r["grown_last50_dev"] for r in all_records]
    coverage_vals = [r["coverage"] for r in all_records]
    print(f"頭出し逸脱: 平均={np.mean(head_start_vals):.4f}, 標準偏差={np.std(head_start_vals):.4f}")
    print(f"独り立ち後逸脱: 平均={np.mean(grown_vals):.4f}, 標準偏差={np.std(grown_vals):.4f}")
    print(f"カバー率: 平均={np.mean(coverage_vals):.4f}, 標準偏差={np.std(coverage_vals):.4f}")

    print("\n=== 世代番号ごとの平均(全lineageをまたいで) ===")
    for gen in range(1, N_GENERATIONS + 1):
        gen_records = [r for r in all_records if r["generation"] == gen]
        hs = np.mean([r["head_start_first50_dev"] for r in gen_records])
        gf = np.mean([r["grown_last50_dev"] for r in gen_records])
        print(f"世代{gen}(n={len(gen_records)}): 頭出し逸脱平均={hs:.4f}, 独り立ち後逸脱平均={gf:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#C0504D", "#4472C4", "#9BBB59", "#4BACC6", "#7F6000"]
    for i, lineage in enumerate(lineages):
        records = load_records(lineage)
        gens = [r["generation"] for r in records]
        grown_final = [r["grown_last50_dev"] for r in records]
        axes[0].plot(gens, grown_final, "o-", label=f"系統{lineage}", color=colors[i % len(colors)])
    axes[0].axhline(0.2, color="gray", linestyle="--", linewidth=1, label="収束の目安(0.2)")
    axes[0].set_xlabel("世代")
    axes[0].set_ylabel("独り立ち後(最後50ep)平均逸脱")
    axes[0].set_title("系統ごとの独り立ち後の到達点(NN版)")
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(range(1, N_GENERATIONS + 1))

    axes[1].hist(grown_vals, bins=10, color="#4472C4", alpha=0.75)
    axes[1].axvline(0.2, color="gray", linestyle="--", linewidth=1, label="収束の目安(0.2)")
    axes[1].set_xlabel("独り立ち後(最後50ep)平均逸脱")
    axes[1].set_ylabel("度数(全lineage×全世代)")
    axes[1].set_title(f"到達点の分布(全{len(all_records)}件、NN版)")
    axes[1].legend()

    fig.suptitle(f"要件4後半 最小限NN移行: 複数世代検証({len(lineages)}系統×{N_GENERATIONS}世代)")
    fig.tight_layout()
    fig.savefig("legacy_multigen_nn_all_lineages.png", dpi=150)
    print("グラフを legacy_multigen_nn_all_lineages.png に保存しました。")

    by_gen = {}
    for gen in range(1, N_GENERATIONS + 1):
        gen_records = [r for r in all_records if r["generation"] == gen]
        by_gen[str(gen)] = {
            "n": len(gen_records),
            "head_start_first50_dev_mean": float(np.mean([r["head_start_first50_dev"] for r in gen_records])),
            "grown_last50_dev_mean": float(np.mean([r["grown_last50_dev"] for r in gen_records])),
        }
    summary = {
        "lineages": lineages, "n_generations": N_GENERATIONS,
        "total_records": len(all_records), "converged": len(converged), "stuck": len(stuck),
        "convergence_rate": len(converged) / len(all_records),
        "head_start_first50_dev_mean": float(np.mean(head_start_vals)),
        "head_start_first50_dev_std": float(np.std(head_start_vals)),
        "grown_last50_dev_mean": float(np.mean(grown_vals)),
        "grown_last50_dev_std": float(np.std(grown_vals)),
        "coverage_mean": float(np.mean(coverage_vals)), "coverage_std": float(np.std(coverage_vals)),
        "by_generation": by_gen,
        "records": all_records,
    }
    with open("nn_multigen_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("数値サマリを nn_multigen_summary.json に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if "teach" not in hp.ACTIONS:
        hp.ACTIONS.append("teach")
    if cmd == "init":
        lineage = int(sys.argv[2])
        tb = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
        init_chunk(lineage, time_budget=tb)
    elif cmd == "gen_chunk":
        lineage = int(sys.argv[2])
        gen = int(sys.argv[3])
        tb = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0
        gen_chunk(lineage, gen, time_budget=tb)
    elif cmd == "aggregate_all":
        aggregate_all([int(a) for a in sys.argv[2:]])
