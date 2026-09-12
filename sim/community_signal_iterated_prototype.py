"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 世代交代のボトルネック(反復学習)による信号創発の検証
==========================================================

集団規模(3体)拡大がむしろ規約を弱めた(受け手の対応表が複数の送り手の符号化
方針を平均してしまう)ことを受け、言語進化研究で実際に体系化を促す効果が
確認されている別のメカニズム、「世代交代の伝達ボトルネック」(iterated
learning)を検証する。新しい学習者は前世代とライブで共同学習するのではなく、
圧縮された限られたサンプルから規約を再構成しなければならない、という設計。

**世代1**: community_signal_v2_prototype.pyの標準設定(4×4グリッド・衝突
ペナルティ8.0・推測ゲームによる直接報酬)で2体を3500ep(収束済み)まで学習
させる。この学習は既にcommunity_v2_qtables_seed{traj_seed}.pklとして完了
済みのため、そのまま再利用する(traj_seed=0,11,22の3系統)。

**世代2以降(反復学習の核)**:
  (1) 前世代の収束済みエージェント(送り手の移動方策・受け手のGuessAgent
      両方)を凍結し、本来の学習量よりずっと少ないN_BOTTLENECK=200エピソード
      だけロールアウトして、送り手の真の支配的逸脱クラスと送った信号の
      ペア(dom, sig)を収集する(「圧縮された限られたサンプル」)。
  (2) 新しい世代のGuessAgent(解釈側)は、このボトルネックサンプルの経験的
      頻度分布から教師あり学習で初期化する: Q(sig, guess) = GUESS_BONUS ×
      (サンプル内でsig条件下でdom=guessだった割合)。これにより、新しい
      世代は前世代の規約の「痕跡」を、直接コピーではなく限られた観測から
      再構成した形で受け継ぐ。
  (3) 送り手側の移動方策(QLearningAgent)は前世代から一切引き継がず、完全に
      新規(空のQテーブル)から始める。つまり「新しい個体」が、あらかじめ
      解釈のヒントだけを与えられた状態で、通常通り強化学習(推測ゲームに
      よる直接報酬)を通じて信号の使い方を(再)発見する。
  (4) 世代1と同じN_EPISODES=3500まで通常通り学習させ、収束後のMIを測定
      する。この収束済みエージェントが次の世代のボトルネックサンプルの
      供給源になる。

これを5世代(世代1は既存流用、世代2〜5を新規学習)繰り返し、収束後のMIが
世代を追うごとに向上・体系化する(反復学習で言語進化文献が予測する効果)か、
それとも要件4の複数世代連鎖検証で見られたような横ばい傾向になるかを確認
する。学習系列の乱数(traj_seed=0,11,22)を変えた3系統(=3つの独立した文化的
系統)で再現性も確認する。

使い方:
  python3 community_signal_iterated_prototype.py gen_chunk <traj_seed> <generation> <end_ep>
  python3 community_signal_iterated_prototype.py gen_finalize <traj_seed> <generation>
  python3 community_signal_iterated_prototype.py aggregate
"""

import sys, json, pickle, random
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

TRAJ_SEEDS = [0, 11, 22]
N_GENERATIONS = 5  # 世代1(既存流用)+世代2〜5(新規学習)
N_EPISODES = m.N_EPISODES          # 3500、世代1と同じ収束までの学習量
DECAY_EPISODES = m.DECAY_EPISODES  # 2500
CHECKPOINT_EPISODES = m.CHECKPOINT_EPISODES  # [300, 1500, 3500]
N_BOTTLENECK_EPISODES = 200        # 世代間の伝達ボトルネック(限られたサンプル)
BOTTLENECK_EPS = 0.1               # ボトルネックロールアウト時の探索率(rollout_for_signal_analysisと同じ)
GUESS_BONUS = m.GUESS_BONUS


def rollout_bottleneck_records(env, q0, q1, gq0, gq1, n_episodes, eps):
    """凍結した前世代のエージェントから、限られたエピソード数だけ
    (dominant_deviation, signal)のペアを収集する(世代間の伝達ボトルネック)。"""
    agent0 = QLearningAgent(); agent0.q = dict(q0)
    agent1 = QLearningAgent(); agent1.q = dict(q1)
    guess0 = m.GuessAgent(); guess0.q = dict(gq0)
    guess1 = m.GuessAgent(); guess1.q = dict(gq1)
    records = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            dom0 = env.dominant_deviation(0)
            dom1 = env.dominant_deviation(1)
            a0 = m.act(agent0, obs[0], eps)
            a1 = m.act(agent1, obs[1], eps)
            records.append((dom0, 1 if a0 == "signal" else 0))
            records.append((dom1, 1 if a1 == "signal" else 0))
            next_obs, rewards, done, deviations, collided = env.step([a0, a1])
            obs = next_obs
    return records


def supervised_init_guess_q(records, guess_classes=(0, 1, 2)):
    """ボトルネックサンプルの経験的頻度分布から、GuessAgentのQテーブルを
    教師あり学習で初期化する: Q(sig,guess) = GUESS_BONUS×P(dom=guess|sig)(サンプル内)。
    そのsig値がサンプル中に一度も現れなければ0のまま(情報なし、RLのみで学習)。"""
    counts = {0: [0, 0, 0], 1: [0, 0, 0]}
    for dom, sig in records:
        counts[sig][dom] += 1
    q = {}
    for sig in (0, 1):
        total = sum(counts[sig])
        for g in guess_classes:
            q[(sig, g)] = (GUESS_BONUS * counts[sig][g] / total) if total > 0 else 0.0
    return q, counts


def load_generation1(traj_seed):
    with open(f"community_v2_qtables_seed{traj_seed}.pkl", "rb") as f:
        d = pickle.load(f)
    return d["agent0_q"], d["agent1_q"], d["guess0_q"], d["guess1_q"]


def load_prev_generation(traj_seed, generation):
    """generation番目を新規学習するために必要な、generation-1番目の収束済みQテーブルを取得。
    generation=2の場合は世代1(既存のcommunity_v2結果)を、それ以外は本プロトタイプが
    保存した前世代の結果を読む。"""
    if generation == 2:
        return load_generation1(traj_seed)
    with open(f"iterated_gen_qtables_seed{traj_seed}_gen{generation - 1}.pkl", "rb") as f:
        d = pickle.load(f)
    return d["agent0_q"], d["agent1_q"], d["guess0_q"], d["guess1_q"]


def gen_chunk(traj_seed, generation, end_ep):
    state_file = f"iterated_state_seed{traj_seed}_gen{generation}.pkl"
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
        print(f"[世代{generation} seed={traj_seed}] 再開: {start_ep}ep目から{end_ep}epまで学習")
    except FileNotFoundError:
        random.seed(traj_seed * 10000 + generation)
        np.random.seed(traj_seed * 10000 + generation)

        prev_q0, prev_q1, prev_gq0, prev_gq1 = load_prev_generation(traj_seed, generation)
        bottleneck_env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        records = rollout_bottleneck_records(
            bottleneck_env, prev_q0, prev_q1, prev_gq0, prev_gq1, N_BOTTLENECK_EPISODES, BOTTLENECK_EPS
        )
        seeded_q, counts = supervised_init_guess_q(records)
        bottleneck_info = {
            "n_records": len(records), "sig_counts": {str(k): v for k, v in counts.items()},
            "seeded_q": {f"{k[0]}_{k[1]}": v for k, v in seeded_q.items()},
        }
        print(f"[世代{generation} seed={traj_seed}] ボトルネック({N_BOTTLENECK_EPISODES}ep, "
              f"{len(records)}レコード)から教師あり初期化: sig=0の分布={counts[0]}, sig=1の分布={counts[1]}")

        env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        agent0, agent1 = QLearningAgent(), QLearningAgent()  # 送り手側は完全に新規(前世代から引き継がない)
        guess0, guess1 = m.GuessAgent(), m.GuessAgent()
        guess0.q = dict(seeded_q)  # 受け手側はボトルネックサンプルからの教師あり初期化
        guess1.q = dict(seeded_q)
        avg_dev_hist, coll_hist, guess_acc_hist = [], [], []
        all_checkpoints = {}
        start_ep = 0
        print(f"[世代{generation} seed={traj_seed}] 新規開始: 0ep目から{end_ep}epまで学習")

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
    print(f"[世代{generation} seed={traj_seed}] {end_ep}epまで完了・保存 "
          f"(直近100ep衝突率={np.mean(coll_hist[-100:]):.4f}, 推測精度={np.mean(guess_acc_hist[-100:]):.4f})")


def gen_finalize(traj_seed, generation):
    state_file = f"iterated_state_seed{traj_seed}_gen{generation}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    agent0, agent1 = state["agent0"], state["agent1"]
    guess0, guess1 = state["guess0"], state["guess1"]
    avg_dev_hist, coll_hist, guess_acc_hist = state["avg_dev_hist"], state["coll_hist"], state["guess_acc_hist"]
    checkpoints = state["checkpoints"]
    bottleneck_info = state["bottleneck_info"]

    print(f"[世代{generation} seed={traj_seed}] === 土台: 衝突回避タスク自体の改善 ===")
    print(f"[世代{generation} seed={traj_seed}] 序盤(最初500ep) 衝突率={np.mean(coll_hist[:500]):.4f}, "
          f"推測精度={np.mean(guess_acc_hist[:500]):.4f}")
    print(f"[世代{generation} seed={traj_seed}] 終盤(最後500ep) 衝突率={np.mean(coll_hist[-500:]):.4f}, "
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
        print(f"[世代{generation} seed={traj_seed}] {n_ep}ep: I(signal;dominant_dev)={mi:.4f}bit, "
              f"信号送信率={signal_rate:.4f}, 推測精度={guess_acc:.4f}(チャンス=0.333)")

    final_q0, final_q1, final_gq0, final_gq1 = checkpoints[N_EPISODES]
    with open(f"iterated_gen_qtables_seed{traj_seed}_gen{generation}.pkl", "wb") as f:
        pickle.dump({"agent0_q": final_q0, "agent1_q": final_q1, "guess0_q": final_gq0, "guess1_q": final_gq1}, f)

    # この系統(seed)のこれまでの世代結果を蓄積
    results_file = f"iterated_results_seed{traj_seed}.json"
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
    print(f"saved {results_file}(世代{generation}まで) と iterated_gen_qtables_seed{traj_seed}_gen{generation}.pkl")


def aggregate():
    print("=== 要件6: 世代交代のボトルネック(反復学習)による信号創発の検証 ===")
    gens = list(range(1, N_GENERATIONS + 1))
    mi_by_gen = {g: [] for g in gens}
    guessacc_by_gen = {g: [] for g in gens}
    rate_by_gen = {g: [] for g in gens}

    for s in TRAJ_SEEDS:
        # 世代1: 既存のcommunity_v2結果を再利用
        gen1_data = json.load(open(f"community_v2_train_seed{s}.json"))
        mi_by_gen[1].append(gen1_data["mi_by_checkpoint"]["3500"]["mi"])
        guessacc_by_gen[1].append(gen1_data["mi_by_checkpoint"]["3500"].get("guess_accuracy", None))
        rate_by_gen[1].append(gen1_data["mi_by_checkpoint"]["3500"]["signal_rate"])

        # 世代2〜5: 本プロトタイプの結果
        iter_data = json.load(open(f"iterated_results_seed{s}.json"))
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

    # 世代1→5の傾向(単調増加か横ばいか)
    mi_means = [summary[g]["mi_mean"] for g in gens]
    from scipy import stats as sstats
    slope, intercept, r, p, se = sstats.linregress(gens, mi_means)
    print(f"\n世代とMIの線形回帰: 傾き={slope:.5f}bit/世代, R^2={r**2:.3f}, p値={p:.4f}")

    with open("iterated_summary.json", "w") as f:
        json.dump({"by_generation": {str(k): v for k, v in summary.items()},
                    "trend_slope": slope, "trend_r2": r ** 2, "trend_p": p}, f, ensure_ascii=False, indent=2)
    print("saved iterated_summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for s_idx, s in enumerate(TRAJ_SEEDS):
        series = [mi_by_gen[g][s_idx] for g in gens]
        axes[0].plot(gens, series, "o--", alpha=0.5, label=f"系統(seed={s})")
    mi_stds = [summary[g]["mi_std"] for g in gens]
    axes[0].errorbar(gens, mi_means, yerr=mi_stds, marker="o", color="black", linewidth=2.5, label="平均±標準偏差")
    axes[0].set_xlabel("世代")
    axes[0].set_ylabel("収束後(3500ep)のI(signal;dominant_dev)[bit]")
    axes[0].set_title(f"世代交代ボトルネックによるMIの推移(傾き={slope:.5f}/世代, p={p:.4f})")
    axes[0].set_xticks(gens)
    axes[0].legend(fontsize=7)

    gacc_means = [summary[g]["guess_acc_mean"] for g in gens]
    rate_means = [summary[g]["signal_rate_mean"] for g in gens]
    axes[1].plot(gens, gacc_means, "s-", color="#4472C4", label="推測精度(チャンス=0.333)")
    axes[1].axhline(1 / 3, color="gray", linestyle=":", alpha=0.6)
    axes[1].plot(gens, rate_means, "^-", color="#C0504D", label="信号送信率")
    axes[1].set_xlabel("世代")
    axes[1].set_ylabel("値")
    axes[1].set_title("推測精度・信号送信率の世代推移")
    axes[1].set_xticks(gens)
    axes[1].legend(fontsize=8)

    fig.suptitle("要件6: 世代交代のボトルネック(反復学習)による信号創発の検証")
    fig.tight_layout()
    fig.savefig("community_signal_iterated_comparison.png", dpi=150)
    print("グラフを community_signal_iterated_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen_chunk":
        gen_chunk(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "gen_finalize":
        gen_finalize(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "aggregate":
        aggregate()
