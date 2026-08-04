"""
感情AIプロジェクト フェーズ10 プロトタイプ: 要件4 転写回数操作による最終診断実験
==========================================================

破滅的干渉仮説(教示ヘッド分離実験、legacy_instinct_nn_splithead_prototype.py)・
教示タイミング仮説(既存ログ再解析、legacy_teach_timing_reanalysis.py)がいずれも
否定された後の3つ目の診断。legacy_bonusとは独立に転写(distillation)の実行回数
だけを操作し、サクセサーの成績を左右しているのが「回数」そのものかを直接確認する。

**条件設計(legacy_instinct_nn_splithead_prototype.pyの
train_elder_teaching_episodeをベースに、教示イベントごとの転写実行を
distill_probability・distill_repeatsという2つの独立変数で制御)**:

- 条件A(thin): legacy_bonus=3.0のまま(報酬・行動選択・teach頻度は通常通り)、
  各teachイベントで実際に転写(distill_step_split)を実行するかどうかを確率
  THIN_KEEP_PROBでベルヌーイ判定し、間引く。THIN_KEEP_PROB=0.4522は、
  既存ログ(nn_legacy_split_run_result_b3_s*.json由来のteach総数平均31563)を
  bonus=0水準(同b0_s*.json由来のteach総数平均14273)まで下げる期待値になるよう
  校正した。
- 条件B(inflate): legacy_bonus=0.0のまま(報酬・行動選択・teach頻度は通常通り)、
  各teachイベントで転写を1回ではなくINFLATE_REPEATS=2回連続実行する(独立に
  新しい状態サンプルを取り直して2回分の回帰更新)。期待される転写回数は
  14273×2≈28546で、bonus=1水準(既存ログ平均29032)にほぼ一致する。

**据え置く点**: 環境(HomeostasisEnv)・報酬設計(teach時のreward+=legacy_bonus)・
状態表現・エルダーの学習過程(action選択・探索率減衰・elder.update)・評価手順
(evaluate_successor_split相当)は直前の実験群と完全に同一。転写操作
(distill_prob/distill_repeats)は転写の実行有無・回数だけに介入し、
elder自身のQネットワークやteach行動の選択確率には一切影響しない
(distill_step_splitはsuccessorの重みしか更新しないため)。

**判定基準**: 条件Aでサクセサーの成績(独り立ち後逸脱)がbonus=0本来の水準
(0.3762±0.0831)並みに改善すれば、条件Bでbonus=0本来の良好な成績が
bonus>=1水準(0.42前後)まで悪化すれば、原因は転写の絶対回数そのものに
あることがほぼ確定する。どちらも変化がなければ、回数仮説も否定する。

規模: n=3(seed=100,200,300、既存実験と同一)からまず様子を見る。

使い方:
  python3 legacy_instinct_nn_transfercount_prototype.py run_chunk <A|B> <seed> [time_budget]
  python3 legacy_instinct_nn_transfercount_prototype.py aggregate
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
    EVAL_EPISODES, EVAL_EPS_DECAY_EPISODES, TRANSFER_COUNT, BLEND, RUN_SEEDS,
)
from legacy_instinct_nn_splithead_prototype import (
    NNAgentSplit, SuccessorNetSplit, distill_step_split, warm_start_teach_net,
    train_elder_with_teaching_split_init, eval_one_episode, WARM_START_POOL,
)

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

ACTIONS = hp.ACTIONS

THIN_KEEP_PROB = 0.4522    # 条件A: bonus=3の転写をbonus=0水準まで間引く保持確率
INFLATE_REPEATS = 2        # 条件B: bonus=0の転写をbonus=1水準まで水増しする倍率

CONDITIONS = {
    "A": {"legacy_bonus": 3.0, "distill_prob": THIN_KEEP_PROB, "distill_repeats": 1},
    "B": {"legacy_bonus": 0.0, "distill_prob": 1.0, "distill_repeats": INFLATE_REPEATS},
}


def train_elder_teaching_episode_ctrl(teach_env, elder, successor, elder_visited, successor_touched,
                                       eps, legacy_bonus, transfer_count, blend,
                                       distill_prob=1.0, distill_repeats=1):
    """distill_prob・distill_repeatsでteachイベントあたりの転写実行有無・回数を
    legacy_bonusとは独立に制御する。報酬・行動選択・環境は一切変えない。"""
    state = teach_env.reset()
    done = False
    devs, teach_count, transfer_count_actual = [], 0, 0
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
            if random.random() < distill_prob:
                for _ in range(distill_repeats):
                    pool = list(elder_visited)
                    sample_n = min(transfer_count, len(pool))
                    sampled_states = random.sample(pool, sample_n)
                    distill_step_split(elder, successor, sampled_states, blend)
                    successor_touched.update(sampled_states)
                    transfer_count_actual += 1

        elder.update(state, action, reward, next_state, done)
        state = next_state
        devs.append(deviation)

    return float(np.mean(devs)), teach_count, transfer_count_actual


def run_chunk(condition, seed, time_budget=33.0):
    """時間主導チャンク実行(legacy_instinct_nn_splithead_prototype.pyのrun_chunkと
    同じphase="teach"→"eval"設計)。"""
    cfg = CONDITIONS[condition]
    tag = f"{condition}_s{seed}"
    result_file = f"nn_legacy_tc_run_result_{tag}.json"
    if os.path.exists(result_file):
        print(f"[legacy-tc {tag}] 既に完了済み(スキップ)")
        return
    state_file = f"nn_legacy_tc_state_{tag}.pkl"

    t_start = time.time()
    try:
        with open(state_file, "rb") as f:
            st = pickle.load(f)
        random.setstate(st["random_state"])
        np.random.set_state(st["np_random_state"])
        print(f"[legacy-tc {tag}] 再開(phase={st['phase']}, teach_ep={st['teach_ep']}/{ELDER_EPISODES}, "
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
            "avg_dev": [], "teach_counts": [], "transfer_counts": [], "succ_avg_dev": [],
            "coverage": 0.0, "eval_agent": None,
        }
        print(f"[legacy-tc {tag}] 新規開始(condition={condition}, legacy_bonus={cfg['legacy_bonus']}, "
              f"distill_prob={cfg['distill_prob']:.4f}, distill_repeats={cfg['distill_repeats']})")

    while st["phase"] == "teach" and st["teach_ep"] < ELDER_EPISODES:
        eps = epsilon_for_episode(st["teach_ep"], ELDER_EPS_DECAY_EPISODES)
        dev, teach_count, transfer_count_actual = train_elder_teaching_episode_ctrl(
            st["teach_env"], st["elder"], st["successor"], st["elder_visited"], st["successor_touched"],
            eps, cfg["legacy_bonus"], TRANSFER_COUNT, BLEND,
            distill_prob=cfg["distill_prob"], distill_repeats=cfg["distill_repeats"])
        st["avg_dev"].append(dev)
        st["teach_counts"].append(teach_count)
        st["transfer_counts"].append(transfer_count_actual)
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
        total_transfer = sum(st["transfer_counts"])
        print(f"[legacy-tc {tag}] 教示フェーズ完了(カバー率={st['coverage']:.4f}, "
              f"転写実行回数合計={total_transfer})、評価フェーズへ")

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
        total_transfer = int(sum(st["transfer_counts"]))
        print(f"[legacy-tc {tag}] teach頻度(終盤100ep)={teach_rate:.4f}, カバー率={st['coverage']:.4f}, "
              f"転写実行回数合計={total_transfer}, サクセサー最初50ep平均逸脱={succ_first50:.4f}")
        result = {
            "condition": condition, "legacy_bonus": cfg["legacy_bonus"], "seed": seed,
            "distill_prob": cfg["distill_prob"], "distill_repeats": cfg["distill_repeats"],
            "teach_rate": teach_rate, "coverage": st["coverage"],
            "total_transfer_count": total_transfer, "succ_first50_dev": succ_first50,
        }
        with open(result_file, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        try:
            if os.path.exists(state_file):
                os.remove(state_file)
        except OSError:
            pass
        print(f"[legacy-tc {tag}] 保存完了({result_file})")
        return

    st["random_state"] = random.getstate()
    st["np_random_state"] = np.random.get_state()
    with open(state_file, "wb") as f:
        pickle.dump(st, f)
    print(f"[legacy-tc {tag}] 時間予算({time_budget}s)到達、phase={st['phase']}, "
          f"teach_ep={st['teach_ep']}/{ELDER_EPISODES}, eval_ep={st['eval_ep']}/{EVAL_EPISODES}")


def _load_baseline(bonus_int, seeds=RUN_SEEDS):
    """既存のsplithead実験(bonus=0/1/3の通常条件)の結果を参考値として再利用する。"""
    vals = []
    for seed in seeds:
        path = f"nn_legacy_split_run_result_b{bonus_int}_s{seed}.json"
        if os.path.exists(path):
            with open(path) as f:
                vals.append(json.load(f)["succ_first50_dev"])
    return vals


def aggregate():
    results = {}
    for cond in ["A", "B"]:
        devs, transfers, teach_rates = [], [], []
        for seed in RUN_SEEDS:
            with open(f"nn_legacy_tc_run_result_{cond}_s{seed}.json") as f:
                d = json.load(f)
            devs.append(d["succ_first50_dev"])
            transfers.append(d["total_transfer_count"])
            teach_rates.append(d["teach_rate"])
        results[cond] = {
            "succ_first50_mean": float(np.mean(devs)), "succ_first50_std": float(np.std(devs)),
            "total_transfer_mean": float(np.mean(transfers)), "total_transfer_std": float(np.std(transfers)),
            "teach_rate_mean": float(np.mean(teach_rates)), "teach_rate_std": float(np.std(teach_rates)),
        }
        print(f"条件{cond}: サクセサー最初50ep平均逸脱={results[cond]['succ_first50_mean']:.4f}±{results[cond]['succ_first50_std']:.4f}, "
              f"転写回数={results[cond]['total_transfer_mean']:.0f}±{results[cond]['total_transfer_std']:.0f}, "
              f"teach頻度={results[cond]['teach_rate_mean']:.4f}")

    baseline_b0 = _load_baseline(0)
    baseline_b1 = _load_baseline(1)
    baseline_b3 = _load_baseline(3)
    ref = {
        "bonus0_normal_mean": float(np.mean(baseline_b0)), "bonus0_normal_std": float(np.std(baseline_b0)),
        "bonus1_normal_mean": float(np.mean(baseline_b1)), "bonus1_normal_std": float(np.std(baseline_b1)),
        "bonus3_normal_mean": float(np.mean(baseline_b3)), "bonus3_normal_std": float(np.std(baseline_b3)),
    }
    print(f"\n参考(既存の通常条件): bonus0={ref['bonus0_normal_mean']:.4f}±{ref['bonus0_normal_std']:.4f}, "
          f"bonus1={ref['bonus1_normal_mean']:.4f}±{ref['bonus1_normal_std']:.4f}, "
          f"bonus3={ref['bonus3_normal_mean']:.4f}±{ref['bonus3_normal_std']:.4f}")
    print(f"条件A(bonus3+転写間引き→bonus0水準)={results['A']['succ_first50_mean']:.4f} "
          f"(bonus0本来={ref['bonus0_normal_mean']:.4f}, bonus3本来={ref['bonus3_normal_mean']:.4f})")
    print(f"条件B(bonus0+転写水増し→bonus1水準)={results['B']['succ_first50_mean']:.4f} "
          f"(bonus0本来={ref['bonus0_normal_mean']:.4f}, bonus1本来={ref['bonus1_normal_mean']:.4f})")

    out = {"conditions": results, "reference": ref}
    with open("nn_legacy_tc_results.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved nn_legacy_tc_results.json")

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = ["bonus0\n本来", "条件A\n(bonus3+間引き)", "bonus1\n本来", "条件B\n(bonus0+水増し)", "bonus3\n本来"]
    means = [ref["bonus0_normal_mean"], results["A"]["succ_first50_mean"], ref["bonus1_normal_mean"],
             results["B"]["succ_first50_mean"], ref["bonus3_normal_mean"]]
    stds = [ref["bonus0_normal_std"], results["A"]["succ_first50_std"], ref["bonus1_normal_std"],
            results["B"]["succ_first50_std"], ref["bonus3_normal_std"]]
    colors = ["#9BBB59", "#4472C4", "#BFBFBF", "#C0504D", "#BFBFBF"]
    ax.bar(xs, means, yerr=stds, color=colors)
    ax.set_ylabel("サクセサー最初50ep平均逸脱(小さいほど良い)")
    ax.set_title("転写回数操作による最終診断実験(n=3)")
    fig.tight_layout()
    fig.savefig("legacy_instinct_nn_transfercount_comparison.png", dpi=150)
    print("グラフを legacy_instinct_nn_transfercount_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if "teach" not in hp.ACTIONS:
        hp.ACTIONS.append("teach")
    if cmd == "run_chunk":
        condition = sys.argv[2]
        seed = int(sys.argv[3])
        tb = float(sys.argv[4]) if len(sys.argv) > 4 else 33.0
        run_chunk(condition, seed, time_budget=tb)
    elif cmd == "aggregate":
        aggregate()
