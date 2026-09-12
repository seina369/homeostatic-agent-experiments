"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 世代交代ボトルネック(反復学習)v2 送り手側も教師あり初期化
==========================================================

前回のボトルネック実験(community_signal_iterated_prototype.py)は、受け手
(GuessAgent)だけをボトルネックサンプルから教師あり初期化し、送り手
(QLearningAgent、移動+信号選択方策)は毎世代完全に新規リセットしていた。
結果、収束後のMIは世代を追って向上せず(傾き-0.00267bit/世代、p=0.4174)、
要件4の複数世代連鎖と同様の横ばい傾向になった。考察として、「送り手側が
前世代の規約と無関係にゼロから探索するため、受け手側の初期バイアスと
噛み合う保証がない」ことを一因として挙げた。

本プロトタイプは、この仮説「送り手側もボトルネックサンプルから初期化すれば
改善するか」を直接検証する。前回との唯一の違い:

  - ボトルネックロールアウト(N_BOTTLENECK=200ep、前世代の凍結エージェント)
    で、GuessAgent用の(dominant_deviation, signal)ペアに加え、QLearningAgent
    用の(observation, action)ペアも新たに収集する。
  - 新しい世代のQLearningAgent(送り手側)を、この(obs, action)の経験的
    頻度分布から教師あり初期化する: Q(obs, action) = SENDER_BONUS ×
    (サンプル内でobsを観測したときにactionを取った割合)。GUESS_BONUSと
    同じスケール(1.0)を採用し、学習が進むにつれて通常のQ学習更新
    (ALPHA=0.2)によって上書きされていく「弱い事前分布」として機能させる。
    サンプルに一度も現れなかったobsは0のまま(情報なし、RLのみで学習)。
  - それ以外(受け手側の教師あり初期化、5世代×3系統、N_EPISODES=3500、
    衝突ペナルティ8.0等)は前回と完全に同一。

これにより、送り手・受け手の両方が前世代の規約の「痕跡」を初期状態として
受け継いだ状態から新しい世代の学習が始まる。この設計変更だけで、収束後の
MIが世代を追って向上する(反復学習の予測どおりの体系化)ようになるかを
確認する。

**2026-07-29 追記(n=3→n=15への拡大)**: n=3では傾き+0.00656bit/世代(R²=0.622)
だったが、p=0.1126で統計的有意性には届かなかった。要件6の新参者効果の
n拡大と同様、traj_seedを追加しn=15(既存のcommunity_v2_qtables_seed{s}.pkl
が既に存在する0,1,2,...,13,22の15系統)まで拡大し、世代に対するMIの傾きが
統計的に有意になるかを再検証する。bashの45秒制限に対応するため、
gen_multi_chunkという新しいコマンドを追加した: 900ep相当を目安としつつ、
実際の経過時間を300ep刻みで監視し、安全余裕(38秒)に達したら自動的に
その時点までの進捗を保存して終了する「時間主導」のチャンク実行方式。
これにより、サンドボックスの実測速度に応じて安全な範囲で最大限まとめて
学習を進められ、固定エピソード数チャンクより少ない呼び出し回数で済む。

使い方:
  python3 community_signal_iterated_v2_prototype.py gen_chunk <traj_seed> <generation> <end_ep>
  python3 community_signal_iterated_v2_prototype.py gen_multi_chunk <traj_seed> <generation> <target_end_ep>
  python3 community_signal_iterated_v2_prototype.py gen_finalize <traj_seed> <generation>
  python3 community_signal_iterated_v2_prototype.py aggregate
"""

import sys, json, pickle, random, time
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import homeostasis_prototype as hp
from homeostasis_prototype import QLearningAgent
import instinct_bias_prototype as ib
import community_signal_v2_prototype as m  # 環境・GuessAgent・run_episode・MI計算等を再利用

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAJ_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22]  # n=3→n=15への拡大(新参者効果のn拡大と同じ既存15系統)
N_GENERATIONS = 5
N_EPISODES = m.N_EPISODES
DECAY_EPISODES = m.DECAY_EPISODES
CHECKPOINT_EPISODES = m.CHECKPOINT_EPISODES
N_BOTTLENECK_EPISODES = 200
BOTTLENECK_EPS = 0.1
GUESS_BONUS = m.GUESS_BONUS
SENDER_BONUS = m.GUESS_BONUS  # 送り手側の教師あり初期化もGUESS_BONUSと同じスケール(1.0)を採用
ACTIONS_COMM = m.ACTIONS_COMM


def rollout_bottleneck_records(env, q0, q1, gq0, gq1, n_episodes, eps):
    """凍結した前世代のエージェントから、限られたエピソード数だけ
    (dominant_deviation, signal)ペア(受け手用)と(observation, action)ペア
    (送り手用、今回新規追加)の両方を収集する。"""
    agent0 = QLearningAgent(); agent0.q = dict(q0)
    agent1 = QLearningAgent(); agent1.q = dict(q1)
    guess0 = m.GuessAgent(); guess0.q = dict(gq0)
    guess1 = m.GuessAgent(); guess1.q = dict(gq1)
    records_guess = []
    records_agent = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            dom0 = env.dominant_deviation(0)
            dom1 = env.dominant_deviation(1)
            a0 = m.act(agent0, obs[0], eps)
            a1 = m.act(agent1, obs[1], eps)
            records_guess.append((dom0, 1 if a0 == "signal" else 0))
            records_guess.append((dom1, 1 if a1 == "signal" else 0))
            records_agent.append((obs[0], a0))
            records_agent.append((obs[1], a1))
            next_obs, rewards, done, deviations, collided = env.step([a0, a1])
            obs = next_obs
    return records_guess, records_agent


def supervised_init_guess_q(records, guess_classes=(0, 1, 2)):
    counts = {0: [0, 0, 0], 1: [0, 0, 0]}
    for dom, sig in records:
        counts[sig][dom] += 1
    q = {}
    for sig in (0, 1):
        total = sum(counts[sig])
        for g in guess_classes:
            q[(sig, g)] = (GUESS_BONUS * counts[sig][g] / total) if total > 0 else 0.0
    return q, counts


def supervised_init_agent_q(records_agent, actions=ACTIONS_COMM, bonus=SENDER_BONUS):
    """送り手側(QLearningAgent)を、ボトルネックサンプルで観測された
    (obs, action)の経験的頻度分布から教師あり初期化する。
    Q(obs, action) = bonus × (サンプル内でobsを観測した際にactionを取った割合)。
    サンプルに一度も現れなかったobsはQ値0のまま(情報なし、通常のRLのみで学習)。"""
    counts = {}
    for obs, a in records_agent:
        counts.setdefault(obs, {act: 0 for act in actions})
        counts[obs][a] += 1
    q = {}
    for obs, act_counts in counts.items():
        total = sum(act_counts.values())
        for a in actions:
            q[(obs, a)] = bonus * act_counts[a] / total
    return q, len(counts)


def load_generation1(traj_seed):
    with open(f"community_v2_qtables_seed{traj_seed}.pkl", "rb") as f:
        d = pickle.load(f)
    return d["agent0_q"], d["agent1_q"], d["guess0_q"], d["guess1_q"]


def load_prev_generation(traj_seed, generation):
    if generation == 2:
        return load_generation1(traj_seed)
    with open(f"iterated_v2_gen_qtables_seed{traj_seed}_gen{generation - 1}.pkl", "rb") as f:
        d = pickle.load(f)
    return d["agent0_q"], d["agent1_q"], d["guess0_q"], d["guess1_q"]


def gen_chunk(traj_seed, generation, end_ep):
    state_file = f"iterated_v2_state_seed{traj_seed}_gen{generation}.pkl"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env = state["env"]
        agent0, agent1 = state["agent0"], state["agent1"]
        guess0, guess1 = state["guess0"], state["guess1"]
        avg_dev_hist, coll_hist, guess_acc_hist = state["avg_dev_hist"], state["coll_hist"], state["guess_acc_hist"]
        all_checkpoints = state["checkpoints"]
        start_ep = state["last_ep"]
        bottleneck_info = state["bottleneck_info"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
        print(f"[v2世代{generation} seed={traj_seed}] 再開: {start_ep}ep目から{end_ep}epまで学習")
    except FileNotFoundError:
        random.seed(traj_seed * 10000 + generation)
        np.random.seed(traj_seed * 10000 + generation)

        prev_q0, prev_q1, prev_gq0, prev_gq1 = load_prev_generation(traj_seed, generation)
        bottleneck_env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        records_guess, records_agent = rollout_bottleneck_records(
            bottleneck_env, prev_q0, prev_q1, prev_gq0, prev_gq1, N_BOTTLENECK_EPISODES, BOTTLENECK_EPS
        )
        seeded_guess_q, counts = supervised_init_guess_q(records_guess)
        seeded_agent_q, n_unique_obs = supervised_init_agent_q(records_agent)
        bottleneck_info = {
            "n_records": len(records_guess), "sig_counts": {str(k): v for k, v in counts.items()},
            "n_unique_obs_for_sender": n_unique_obs,
        }
        print(f"[v2世代{generation} seed={traj_seed}] ボトルネック({N_BOTTLENECK_EPISODES}ep, "
              f"{len(records_guess)}レコード)から教師あり初期化: sig=0の分布={counts[0]}, sig=1の分布={counts[1]}, "
              f"送り手側の初期化obs種類数={n_unique_obs}")

        env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        agent0, agent1 = QLearningAgent(), QLearningAgent()
        agent0.q = dict(seeded_agent_q)  # 送り手側も今回はボトルネックサンプルから教師あり初期化
        agent1.q = dict(seeded_agent_q)
        guess0, guess1 = m.GuessAgent(), m.GuessAgent()
        guess0.q = dict(seeded_guess_q)
        guess1.q = dict(seeded_guess_q)
        avg_dev_hist, coll_hist, guess_acc_hist = [], [], []
        all_checkpoints = {}
        start_ep = 0
        print(f"[v2世代{generation} seed={traj_seed}] 新規開始: 0ep目から{end_ep}epまで学習")

    dev_h, coll_h, gacc_h, checkpoints = m.train_pair_range(
        env, agent0, agent1, guess0, guess1, start_ep, end_ep, DECAY_EPISODES, checkpoint_eps=CHECKPOINT_EPISODES
    )
    avg_dev_hist.extend(dev_h); coll_hist.extend(coll_h); guess_acc_hist.extend(gacc_h)
    all_checkpoints.update(checkpoints)

    state = {
        "env": env, "agent0": agent0, "agent1": agent1, "guess0": guess0, "guess1": guess1,
        "avg_dev_hist": avg_dev_hist, "coll_hist": coll_hist, "guess_acc_hist": guess_acc_hist,
        "checkpoints": all_checkpoints, "last_ep": end_ep, "bottleneck_info": bottleneck_info,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)
    print(f"[v2世代{generation} seed={traj_seed}] {end_ep}epまで完了・保存 "
          f"(直近100ep衝突率={np.mean(coll_hist[-100:]):.4f}, 推測精度={np.mean(guess_acc_hist[-100:]):.4f})")


def gen_multi_chunk(traj_seed, generation, target_end_ep=N_EPISODES, time_budget=38.0, sub_step=300):
    """900ep程度を目安にしつつ、実際の経過時間をsub_step(300ep)刻みで監視し、
    time_budget(38秒)に達したらその時点までの進捗を保存して終了する「時間主導」の
    チャンク実行。bashの45秒制限に対して、サンドボックスの実測速度に応じて安全な
    範囲で最大限まとめて学習を進め、固定エピソード数チャンクより呼び出し回数を
    減らすための効率化。target_end_epまで到達したら自動的にgen_finalizeも実行する。"""
    t_start = time.time()
    state_file = f"iterated_v2_state_seed{traj_seed}_gen{generation}.pkl"
    n_sub_chunks = 0
    while True:
        try:
            with open(state_file, "rb") as f:
                cur_last_ep = pickle.load(f)["last_ep"]
        except FileNotFoundError:
            cur_last_ep = 0
        if cur_last_ep >= target_end_ep:
            break
        if (time.time() - t_start) > time_budget:
            print(f"[v2世代{generation} seed={traj_seed}] 時間予算({time_budget}s)到達、{cur_last_ep}epで一旦終了")
            return
        next_ep = min(cur_last_ep + sub_step, target_end_ep)
        gen_chunk(traj_seed, generation, next_ep)
        n_sub_chunks += 1

    print(f"[v2世代{generation} seed={traj_seed}] target_end_ep={target_end_ep}に到達、{n_sub_chunks}個のサブチャンクで完了、"
          f"gen_finalizeを実行")
    gen_finalize(traj_seed, generation)


def gen_finalize(traj_seed, generation):
    state_file = f"iterated_v2_state_seed{traj_seed}_gen{generation}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    agent0, agent1 = state["agent0"], state["agent1"]
    guess0, guess1 = state["guess0"], state["guess1"]
    avg_dev_hist, coll_hist, guess_acc_hist = state["avg_dev_hist"], state["coll_hist"], state["guess_acc_hist"]
    checkpoints = state["checkpoints"]
    bottleneck_info = state["bottleneck_info"]

    print(f"[v2世代{generation} seed={traj_seed}] === 土台: 衝突回避タスク自体の改善 ===")
    print(f"[v2世代{generation} seed={traj_seed}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"推測精度={np.mean(guess_acc_hist[:500]):.4f}")
    print(f"[v2世代{generation} seed={traj_seed}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
          f"推測精度={np.mean(guess_acc_hist[-500:]):.4f}")

    mi_by_checkpoint = {}
    for n_ep in CHECKPOINT_EPISODES:
        q0, q1, gq0, gq1 = checkpoints[n_ep]
        random.seed(traj_seed * 70000 + generation * 1000 + n_ep)
        np.random.seed(traj_seed * 70000 + generation * 1000 + n_ep)
        rollout_env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        records, guess_correct = m.rollout_for_signal_analysis(
            rollout_env, dict(q0), dict(q1), dict(gq0), dict(gq1), m.N_ROLLOUT_EPISODES, m.ROLLOUT_EPS
        )
        mi, signal_rate, cond_dist, marg_dist = m.mutual_info_signal_vs_class(records)
        guess_acc = float(np.mean(guess_correct))
        mi_by_checkpoint[str(n_ep)] = {
            "mi": mi, "signal_rate": signal_rate, "cond_dist_given_signal": cond_dist,
            "marginal_dist": marg_dist, "guess_accuracy": guess_acc,
        }
        print(f"[v2世代{generation} seed={traj_seed}] {n_ep}ep: I(signal;dominant_dev)={mi:.4f}bit, "
              f"信号送信率={signal_rate:.4f}, 推測精度={guess_acc:.4f}(チャンス=0.333)")

    final_q0, final_q1, final_gq0, final_gq1 = checkpoints[N_EPISODES]
    with open(f"iterated_v2_gen_qtables_seed{traj_seed}_gen{generation}.pkl", "wb") as f:
        pickle.dump({"agent0_q": final_q0, "agent1_q": final_q1, "guess0_q": final_gq0, "guess1_q": final_gq1}, f)

    results_file = f"iterated_v2_results_seed{traj_seed}.json"
    try:
        with open(results_file) as f:
            all_gens = json.load(f)
    except FileNotFoundError:
        all_gens = {}
    all_gens[str(generation)] = {
        "generation": generation, "traj_seed": traj_seed,
        "bottleneck_info": bottleneck_info,
        "collision_early": float(np.mean(coll_hist[:500])), "collision_late": float(np.mean(coll_hist[-500:])),
        "mi_by_checkpoint": mi_by_checkpoint,
    }
    with open(results_file, "w") as f:
        json.dump(all_gens, f, ensure_ascii=False, indent=2)
    print(f"saved {results_file}(世代{generation}まで) と iterated_v2_gen_qtables_seed{traj_seed}_gen{generation}.pkl")


def aggregate():
    print("=== 要件6: 世代交代のボトルネック v2(送り手側も教師あり初期化)による信号創発の検証 ===")
    gens = list(range(1, N_GENERATIONS + 1))
    mi_by_gen = {g: [] for g in gens}
    guessacc_by_gen = {g: [] for g in gens}
    rate_by_gen = {g: [] for g in gens}

    for s in TRAJ_SEEDS:
        gen1_data = json.load(open(f"community_v2_train_seed{s}.json"))
        mi_by_gen[1].append(gen1_data["mi_by_checkpoint"]["3500"]["mi"])
        guessacc_by_gen[1].append(gen1_data["mi_by_checkpoint"]["3500"].get("guess_accuracy", None))
        rate_by_gen[1].append(gen1_data["mi_by_checkpoint"]["3500"]["signal_rate"])

        iter_data = json.load(open(f"iterated_v2_results_seed{s}.json"))
        for g in gens[1:]:
            entry = iter_data[str(g)]["mi_by_checkpoint"]["3500"]
            mi_by_gen[g].append(entry["mi"])
            guessacc_by_gen[g].append(entry["guess_accuracy"])
            rate_by_gen[g].append(entry["signal_rate"])

    print("\n=== 世代ごとの収束後(3500ep)MI(n=3系統の平均±標準偏差) ===")
    summary = {}
    for g in gens:
        mis = mi_by_gen[g]
        gaccs = [x for x in guessacc_by_gen[g] if x is not None]
        rates = rate_by_gen[g]
        summary[g] = {
            "mi_mean": float(np.mean(mis)), "mi_std": float(np.std(mis)), "mi_values": mis,
            "guess_acc_mean": float(np.mean(gaccs)) if gaccs else None,
            "signal_rate_mean": float(np.mean(rates)),
        }
        print(f"世代{g}: MI={np.mean(mis):.4f}±{np.std(mis):.4f}bit(系統別: {['%.4f' % x for x in mis]}), "
              f"推測精度={summary[g]['guess_acc_mean']:.4f}, 信号送信率={np.mean(rates):.4f}")

    mi_means = [summary[g]["mi_mean"] for g in gens]
    from scipy import stats as sstats
    slope, intercept, r, p, se = sstats.linregress(gens, mi_means)
    print(f"\n世代とMIの線形回帰: 傾き={slope:.5f}bit/世代, R^2={r**2:.3f}, p値={p:.4f}")

    # 前回(受け手側のみ初期化)の結果との比較
    prev_summary = None
    try:
        prev_summary = json.load(open("iterated_summary.json"))
    except FileNotFoundError:
        pass

    with open("iterated_v2_summary.json", "w") as f:
        json.dump({"by_generation": {str(k): v for k, v in summary.items()},
                    "trend_slope": slope, "trend_r2": r ** 2, "trend_p": p}, f, ensure_ascii=False, indent=2)
    print("saved iterated_v2_summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for s_idx, s in enumerate(TRAJ_SEEDS):
        series = [mi_by_gen[g][s_idx] for g in gens]
        axes[0].plot(gens, series, "o--", alpha=0.4, color="#2E7D32", label=f"v2系統(seed={s})" if s_idx == 0 else None)
        axes[0].plot(gens, series, "o--", alpha=0.4, color="#2E7D32")
    mi_stds = [summary[g]["mi_std"] for g in gens]
    axes[0].errorbar(gens, mi_means, yerr=mi_stds, marker="o", color="#2E7D32", linewidth=2.5,
                      label="v2(送り手側も初期化)平均±標準偏差")
    if prev_summary is not None:
        prev_means = [prev_summary["by_generation"][str(g)]["mi_mean"] for g in gens]
        prev_stds = [prev_summary["by_generation"][str(g)]["mi_std"] for g in gens]
        axes[0].errorbar(gens, prev_means, yerr=prev_stds, marker="s", color="#C0504D", linewidth=2.0,
                          linestyle="--", label="前回(受け手側のみ初期化)平均±標準偏差")
    axes[0].set_xlabel("世代")
    axes[0].set_ylabel("収束後(3500ep)のI(signal;dominant_dev)[bit]")
    axes[0].set_title(f"送り手側も教師あり初期化した場合のMI推移(傾き={slope:.5f}/世代, p={p:.4f})")
    axes[0].set_xticks(gens)
    axes[0].legend(fontsize=7)

    gacc_means = [summary[g]["guess_acc_mean"] for g in gens]
    rate_means = [summary[g]["signal_rate_mean"] for g in gens]
    axes[1].plot(gens, gacc_means, "s-", color="#4472C4", label="推測精度(チャンス=0.333)")
    axes[1].axhline(1 / 3, color="gray", linestyle=":", alpha=0.6)
    axes[1].plot(gens, rate_means, "^-", color="#C0504D", label="信号送信率")
    axes[1].set_xlabel("世代")
    axes[1].set_ylabel("値")
    axes[1].set_title("推測精度・信号送信率の世代推移(v2)")
    axes[1].set_xticks(gens)
    axes[1].legend(fontsize=8)

    fig.suptitle("要件6: 世代交代ボトルネックv2(送り手側も教師あり初期化)による信号創発の検証")
    fig.tight_layout()
    fig.savefig("community_signal_iterated_v2_comparison.png", dpi=150)
    print("グラフを community_signal_iterated_v2_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen_chunk":
        gen_chunk(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "gen_multi_chunk":
        gen_multi_chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "gen_finalize":
        gen_finalize(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "aggregate":
        aggregate()
