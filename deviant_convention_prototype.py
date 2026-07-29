"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 逸脱エージェントによる規範性の検証
==========================================================

これまでの要件6実験(community_signal_v2_prototype.py)は、信号と内部状態の
間に学習を通じて統計的な対応関係が生まれることを示したが、これが「単なる
相関」ではなく「規範を伴う本物の慣習」と言えるかは未検証だった。ウィトゲン
シュタインの生活形式論に立てば、本物の規則には「間違った使い方を共同体が
正せる」という規範的な側面が必要である。

**設計**: community_signal_v2_prototype.pyで収束済みの既存集団
(agent0/agent1、traj_seed=0,11,22)を前提とする。
  (1) 「逸脱エージェント」を、既存集団と食い違う信号-意味の対応関係を持つ
      別の小さな社会(deviant + deviant_partner)として、置換
      π: 0→1, 1→2, 2→0(完全な字亂換、どのクラスも自分自身には写らない)を
      適用した「推測の正解」基準のもとでゼロから学習させる(推測ゲームの
      報酬計算だけを、真のdominant_deviationクラスではなくπ(真のクラス)との
      一致で判定する。両エージェントが同じ置換のもとで学習するため、2体は
      「一貫しているが既存集団とは食い違う」独自の対応関係を発展させる)。
  (2) 収束した逸脱エージェント(deviant)を、既存集団の収束済みメンバー
      (established agent0、Qテーブル・推測テーブルとも凍結)と組ませ、
      **真のクラス**(実環境が定義する、恣意的でない客観的な基準)で
      採点する。
      - フェーズA(両者凍結): 即座のロールアウトで、(a)逸脱エージェントが
        元の(既存集団と食い違う)対応関係のまま信号をやり取りした場合の
        推測精度、(b)対照として、既存集団の実際のagent1(順応エージェント、
        同じ真のクラス基準で学習済み)とagent0を組ませた場合の推測精度、
        を同じ手法で比較する。(a)が(b)より明確に低ければ、対応関係には
        恣意的でない「正しさ」があることになる。
      - フェーズB(逸脱側のみ学習継続): 逸脱エージェントのQテーブル・推測
        テーブルだけ学習を継続させ(established agent0は凍結のまま)、
        真のクラス基準の報酬のもとで、逸脱エージェントの信号使用が既存
        集団の対応関係へ収束していくか(=agent0からみた推測精度が既存
        水準まで回復するか)を、チェックポイントごとのロールアウトで追跡する。

3系統(traj_seed=0,11,22、community_signal_v2_prototype.pyの既存集団と同じ
乱数系列)で確認する。45秒のbash呼び出し制限に対応するため、逸脱エージェント
の事前学習は2チャンク(0→1750ep、1750→3500ep)に分割する。

使い方:
  python3 deviant_convention_prototype.py pretrain_chunk <seed> <end_ep>
  python3 deviant_convention_prototype.py pretrain_finalize <seed>
  python3 deviant_convention_prototype.py test <seed>
  python3 deviant_convention_prototype.py aggregate
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
import community_signal_v2_prototype as m

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

TRAJ_SEEDS = [0, 11, 22]
N_PRETRAIN_EPISODES = 3500
PRETRAIN_DECAY_EPISODES = 2500
PERM = {0: 1, 1: 2, 2: 0}  # 完全な字乱換(どのクラスも自分自身には写らない)

N_PHASEA_EPISODES = 200
PHASEA_EPS = 0.1
N_PHASEB_EPISODES = 2000
PHASEB_DECAY_EPISODES = 1500
PHASEB_CHECKPOINTS = [0, 300, 800, 1500, 2000]
GUESS_EPS = m.GUESS_EPS


def run_episode_permuted(env, agent0, agent1, guess0, guess1, eps0, eps1,
                          learn0=True, learn1=True, learn_guess0=True, learn_guess1=True,
                          guess_eps=GUESS_EPS):
    """逸脱ペアの事前学習用。推測の正解基準を真のクラスではなくPERM(真のクラス)にする。
    両エージェントが同じ置換のもとで学習するため、内部的に一貫した「別の対応関係」を発展させる。"""
    obs = env.reset()
    done = False
    devs, collisions, steps = [], 0, 0
    correct0_count, correct1_count, n_guesses = 0, 0, 0
    while not done:
        dom0 = env.dominant_deviation(0)
        dom1 = env.dominant_deviation(1)

        a0 = m.act(agent0, obs[0], eps0)
        a1 = m.act(agent1, obs[1], eps1)

        sig_for_guess0 = obs[0][6]
        sig_for_guess1 = obs[1][6]
        guess0_val = guess0.act(sig_for_guess0, guess_eps)
        guess1_val = guess1.act(sig_for_guess1, guess_eps)
        correct0 = int(guess0_val == PERM[dom1])  # 置換した基準で正解判定
        correct1 = int(guess1_val == PERM[dom0])
        correct0_count += correct0
        correct1_count += correct1
        n_guesses += 1

        next_obs, base_rewards, done, deviations, collided = env.step([a0, a1])
        total_r0 = base_rewards[0] + m.GUESS_BONUS * correct0 + m.GUESS_BONUS * correct1
        total_r1 = base_rewards[1] + m.GUESS_BONUS * correct1 + m.GUESS_BONUS * correct0

        if learn0:
            agent0.update(obs[0], a0, total_r0, next_obs[0], done)
        if learn1:
            agent1.update(obs[1], a1, total_r1, next_obs[1], done)
        if learn_guess0:
            guess0.update(sig_for_guess0, guess0_val, m.GUESS_BONUS * correct0)
        if learn_guess1:
            guess1.update(sig_for_guess1, guess1_val, m.GUESS_BONUS * correct1)

        obs = next_obs
        devs.append((deviations[0] + deviations[1]) / 2.0)
        collisions += int(collided)
        steps += 1

    avg_dev = float(np.mean(devs))
    coll_rate = collisions / steps
    guess_acc = (correct0_count + correct1_count) / (2 * n_guesses)
    return avg_dev, coll_rate, guess_acc


def run_episode_mixed(env, agent0, agent1, guess0, guess1, eps0, eps1,
                       learn0, learn1, learn_guess0, learn_guess1, guess_eps=GUESS_EPS):
    """混成ペア用。採点は常に真のクラス(実環境が定義する客観的な基準)。
    agent0がagent1を当てる精度・agent1がagent0を当てる精度を別々に返す。"""
    obs = env.reset()
    done = False
    correct0_count, correct1_count, n_guesses = 0, 0, 0
    while not done:
        dom0 = env.dominant_deviation(0)
        dom1 = env.dominant_deviation(1)

        a0 = m.act(agent0, obs[0], eps0)
        a1 = m.act(agent1, obs[1], eps1)

        sig_for_guess0 = obs[0][6]
        sig_for_guess1 = obs[1][6]
        guess0_val = guess0.act(sig_for_guess0, guess_eps)
        guess1_val = guess1.act(sig_for_guess1, guess_eps)
        correct0 = int(guess0_val == dom1)  # 真のクラスで採点
        correct1 = int(guess1_val == dom0)
        correct0_count += correct0
        correct1_count += correct1
        n_guesses += 1

        next_obs, base_rewards, done, deviations, collided = env.step([a0, a1])
        total_r0 = base_rewards[0] + m.GUESS_BONUS * correct0 + m.GUESS_BONUS * correct1
        total_r1 = base_rewards[1] + m.GUESS_BONUS * correct1 + m.GUESS_BONUS * correct0

        if learn0:
            agent0.update(obs[0], a0, total_r0, next_obs[0], done)
        if learn1:
            agent1.update(obs[1], a1, total_r1, next_obs[1], done)
        if learn_guess0:
            guess0.update(sig_for_guess0, guess0_val, m.GUESS_BONUS * correct0)
        if learn_guess1:
            guess1.update(sig_for_guess1, guess1_val, m.GUESS_BONUS * correct1)

        obs = next_obs
        n_guesses = n_guesses  # (steps counted via n_guesses)

    acc0_about_1 = correct0_count / n_guesses  # agent0がagent1を当てた精度
    acc1_about_0 = correct1_count / n_guesses  # agent1がagent0を当てた精度
    return acc0_about_1, acc1_about_0


def pretrain_chunk(seed, end_ep):
    state_file = f"deviant_state_seed{seed}.pkl"
    try:
        with open(state_file, "rb") as f:
            state = pickle.load(f)
        env = state["env"]
        agent0, agent1 = state["agent0"], state["agent1"]
        guess0, guess1 = state["guess0"], state["guess1"]
        acc_hist = state["acc_hist"]
        start_ep = state["last_ep"]
        random.setstate(state["random_state"])
        np.random.set_state(state["np_random_state"])
        print(f"[逸脱seed={seed}] 再開: {start_ep}ep目から{end_ep}epまで学習")
    except FileNotFoundError:
        random.seed(seed * 31 + 1)
        np.random.seed(seed * 31 + 1)
        env = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
        agent0, agent1 = QLearningAgent(), QLearningAgent()
        guess0, guess1 = m.GuessAgent(), m.GuessAgent()
        acc_hist = []
        start_ep = 0
        print(f"[逸脱seed={seed}] 新規開始: 0ep目から{end_ep}epまで学習(置換π={PERM})")

    for ep in range(start_ep, end_ep):
        eps = ib.epsilon_for_episode(ep, PRETRAIN_DECAY_EPISODES)
        _, _, guess_acc = run_episode_permuted(env, agent0, agent1, guess0, guess1, eps, eps)
        acc_hist.append(guess_acc)

    state = {
        "env": env, "agent0": agent0, "agent1": agent1, "guess0": guess0, "guess1": guess1,
        "acc_hist": acc_hist, "last_ep": end_ep,
        "random_state": random.getstate(), "np_random_state": np.random.get_state(),
    }
    with open(state_file, "wb") as f:
        pickle.dump(state, f)
    print(f"[逸脱seed={seed}] {end_ep}epまで完了・保存(直近100ep置換基準精度={np.mean(acc_hist[-100:]):.4f})")


def pretrain_finalize(seed):
    state_file = f"deviant_state_seed{seed}.pkl"
    with open(state_file, "rb") as f:
        state = pickle.load(f)
    agent0, agent1 = state["agent0"], state["agent1"]
    guess0, guess1 = state["guess0"], state["guess1"]
    acc_hist = state["acc_hist"]
    print(f"[逸脱seed={seed}] 置換基準での推測精度: 序盤(最初500ep)={np.mean(acc_hist[:500]):.4f}, "
          f"終盤(最後500ep)={np.mean(acc_hist[-500:]):.4f}(チャンス=0.333)")
    with open(f"deviant_qtables_seed{seed}.pkl", "wb") as f:
        pickle.dump({
            "deviant_q": dict(agent1.q), "deviant_guess_q": dict(guess1.q),
            "deviant_partner_q": dict(agent0.q), "deviant_partner_guess_q": dict(guess0.q),
        }, f)
    print(f"saved deviant_qtables_seed{seed}.pkl")


def test(seed):
    with open(f"deviant_qtables_seed{seed}.pkl", "rb") as f:
        dev = pickle.load(f)
    with open(f"community_v2_qtables_seed{seed}.pkl", "rb") as f:
        est = pickle.load(f)

    deviant_agent = QLearningAgent(); deviant_agent.q = dict(dev["deviant_q"])
    deviant_guess = m.GuessAgent(); deviant_guess.q = dict(dev["deviant_guess_q"])
    established_agent0 = QLearningAgent(); established_agent0.q = dict(est["agent0_q"])
    established_guess0 = m.GuessAgent(); established_guess0.q = dict(est["guess0_q"])
    conformist_agent1 = QLearningAgent(); conformist_agent1.q = dict(est["agent1_q"])
    conformist_guess1 = m.GuessAgent(); conformist_guess1.q = dict(est["guess1_q"])

    # --- フェーズA: 両者凍結、真のクラスで採点 ---
    random.seed(seed * 97 + 1)
    np.random.seed(seed * 97 + 1)
    env_dev = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
    dev_acc0_list, dev_acc1_list = [], []
    for _ in range(N_PHASEA_EPISODES):
        a0, a1 = run_episode_mixed(env_dev, established_agent0, deviant_agent, established_guess0, deviant_guess,
                                    PHASEA_EPS, PHASEA_EPS, False, False, False, False, guess_eps=0.0)
        dev_acc0_list.append(a0); dev_acc1_list.append(a1)

    random.seed(seed * 97 + 2)
    np.random.seed(seed * 97 + 2)
    env_conf = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
    conf_acc0_list, conf_acc1_list = [], []
    for _ in range(N_PHASEA_EPISODES):
        a0, a1 = run_episode_mixed(env_conf, established_agent0, conformist_agent1, established_guess0, conformist_guess1,
                                    PHASEA_EPS, PHASEA_EPS, False, False, False, False, guess_eps=0.0)
        conf_acc0_list.append(a0); conf_acc1_list.append(a1)

    phaseA = {
        "deviant_acc0_about_1_mean": float(np.mean(dev_acc0_list)), "deviant_acc0_about_1_std": float(np.std(dev_acc0_list)),
        "deviant_acc1_about_0_mean": float(np.mean(dev_acc1_list)), "deviant_acc1_about_0_std": float(np.std(dev_acc1_list)),
        "conformist_acc0_about_1_mean": float(np.mean(conf_acc0_list)), "conformist_acc0_about_1_std": float(np.std(conf_acc0_list)),
        "conformist_acc1_about_0_mean": float(np.mean(conf_acc1_list)), "conformist_acc1_about_0_std": float(np.std(conf_acc1_list)),
    }
    print(f"[seed={seed}] フェーズA: 逸脱ペア agent0→1精度={phaseA['deviant_acc0_about_1_mean']:.4f}, "
          f"1→0精度={phaseA['deviant_acc1_about_0_mean']:.4f} | "
          f"順応ペア agent0→1精度={phaseA['conformist_acc0_about_1_mean']:.4f}, "
          f"1→0精度={phaseA['conformist_acc1_about_0_mean']:.4f}(チャンス=0.333)")

    # --- フェーズB: 逸脱側のみ学習継続、established_agent0は凍結、真のクラスで採点 ---
    random.seed(seed * 97 + 3)
    np.random.seed(seed * 97 + 3)
    env_b = m.MultiAgentHomeostasisEnv(random.Random(m.TRAIN_SEED))
    deviant_agent_b = QLearningAgent(); deviant_agent_b.q = dict(dev["deviant_q"])
    deviant_guess_b = m.GuessAgent(); deviant_guess_b.q = dict(dev["deviant_guess_q"])

    checkpoint_set = set(PHASEB_CHECKPOINTS)
    phaseB_track = {}       # 0→1方向: established agent0が逸脱者の真のクラスを当てる精度
    phaseB_track_10 = {}    # 1→0方向: 逸脱者がestablished agent0の真のクラスを当てる精度(逸脱者自身の解釈規則)

    def eval_both(n=50):
        accs0, accs1 = [], []
        for _ in range(n):
            a0, a1 = run_episode_mixed(env_b, established_agent0, deviant_agent_b, established_guess0, deviant_guess_b,
                                        PHASEA_EPS, PHASEA_EPS, False, False, False, False, guess_eps=0.0)
            accs0.append(a0); accs1.append(a1)
        return float(np.mean(accs0)), float(np.mean(accs1))

    if 0 in checkpoint_set:
        a0, a1 = eval_both()
        phaseB_track[0] = a0
        phaseB_track_10[0] = a1

    prev_ep = 0
    for cp in sorted(c for c in PHASEB_CHECKPOINTS if c > 0):
        for ep in range(prev_ep, cp):
            eps = ib.epsilon_for_episode(ep, PHASEB_DECAY_EPISODES)
            run_episode_mixed(env_b, established_agent0, deviant_agent_b, established_guess0, deviant_guess_b,
                               eps, eps, False, True, False, True, guess_eps=GUESS_EPS)
        prev_ep = cp
        a0, a1 = eval_both()
        phaseB_track[cp] = a0
        phaseB_track_10[cp] = a1
        print(f"[seed={seed}] フェーズB {cp}ep時点: agent0が逸脱者を当てる精度(0→1)={a0:.4f}, "
              f"逸脱者がagent0を当てる精度(1→0、逸脱者自身の解釈規則)={a1:.4f}")

    result = {"seed": seed, "phaseA": phaseA, "phaseB_track": phaseB_track, "phaseB_track_10": phaseB_track_10}
    with open(f"deviant_test_seed{seed}.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved deviant_test_seed{seed}.json")


def aggregate():
    data = [json.load(open(f"deviant_test_seed{s}.json")) for s in TRAJ_SEEDS]

    print("=== (1a) フェーズA 0→1方向: established agent0が相手(逸脱者/本来のagent1)を当てる精度 ===")
    dev01 = [d["phaseA"]["deviant_acc0_about_1_mean"] for d in data]
    conf01 = [d["phaseA"]["conformist_acc0_about_1_mean"] for d in data]
    print(f"逸脱ペア={np.mean(dev01):.4f}±{np.std(dev01):.4f} (系統別: {[round(v,3) for v in dev01]})")
    print(f"順応ペア={np.mean(conf01):.4f}±{np.std(conf01):.4f} (系統別: {[round(v,3) for v in conf01]})")

    print("\n=== (1b) フェーズA 1→0方向: 相手(逸脱者/本来のagent1)がestablished agent0を当てる精度 ===")
    dev10 = [d["phaseA"]["deviant_acc1_about_0_mean"] for d in data]
    conf10 = [d["phaseA"]["conformist_acc1_about_0_mean"] for d in data]
    print(f"逸脱ペア(逸脱者自身の解釈規則)={np.mean(dev10):.4f}±{np.std(dev10):.4f} (系統別: {[round(v,3) for v in dev10]})")
    print(f"順応ペア(本来のagent1)={np.mean(conf10):.4f}±{np.std(conf10):.4f} (系統別: {[round(v,3) for v in conf10]})")
    print(f"(チャンスレート=0.333)")

    from scipy import stats as sstats
    t01, p01 = sstats.ttest_rel(conf01, dev01)
    t10, p10 = sstats.ttest_rel(conf10, dev10)
    print(f"\n対応のあるt検定(順応-逸脱, n=3): 0→1方向 t={t01:.3f},p={p01:.3f} | 1→0方向 t={t10:.3f},p={p10:.3f}")

    print("\n=== (2a) フェーズB 0→1方向: agent0が逸脱者を当てる精度の推移 ===")
    cps = PHASEB_CHECKPOINTS
    track_means = {}
    for cp in cps:
        vals = [d["phaseB_track"][str(cp)] for d in data]
        track_means[cp] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"{cp}ep: {np.mean(vals):.4f}±{np.std(vals):.4f}")

    print("\n=== (2b) フェーズB 1→0方向: 逸脱者がagent0を当てる精度(逸脱者自身の解釈規則)の推移 ===")
    track_means_10 = {}
    for cp in cps:
        vals = [d["phaseB_track_10"][str(cp)] for d in data]
        track_means_10[cp] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"{cp}ep: {np.mean(vals):.4f}±{np.std(vals):.4f}")

    summary = {
        "phaseA_0to1": {
            "deviant_mean": float(np.mean(dev01)), "deviant_std": float(np.std(dev01)), "deviant_runs": dev01,
            "conformist_mean": float(np.mean(conf01)), "conformist_std": float(np.std(conf01)), "conformist_runs": conf01,
            "paired_t": float(t01), "paired_p": float(p01),
        },
        "phaseA_1to0": {
            "deviant_mean": float(np.mean(dev10)), "deviant_std": float(np.std(dev10)), "deviant_runs": dev10,
            "conformist_mean": float(np.mean(conf10)), "conformist_std": float(np.std(conf10)), "conformist_runs": conf10,
            "paired_t": float(t10), "paired_p": float(p10),
        },
        "phaseB_track_0to1": {str(cp): track_means[cp] for cp in cps},
        "phaseB_track_1to0": {str(cp): track_means_10[cp] for cp in cps},
    }
    with open("deviant_convention_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved deviant_convention_summary.json")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    labels = ["逸脱ペア", "順応ペア"]

    axes[0, 0].bar(labels, [np.mean(dev01), np.mean(conf01)], yerr=[np.std(dev01), np.std(conf01)],
                    color=["#C0504D", "#4472C4"])
    axes[0, 0].axhline(1 / 3, color="gray", linestyle="--", label="チャンスレート")
    axes[0, 0].set_ylabel("精度")
    axes[0, 0].set_title("フェーズA 0→1: agent0が相手を当てる精度")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(labels, [np.mean(dev10), np.mean(conf10)], yerr=[np.std(dev10), np.std(conf10)],
                    color=["#C0504D", "#4472C4"])
    axes[0, 1].axhline(1 / 3, color="gray", linestyle="--", label="チャンスレート")
    axes[0, 1].set_ylabel("精度")
    axes[0, 1].set_title("フェーズA 1→0: 相手がagent0を当てる精度(相手自身の解釈規則)")
    axes[0, 1].legend(fontsize=8)

    cps_arr = list(cps)
    means_b = [track_means[cp]["mean"] for cp in cps_arr]
    stds_b = [track_means[cp]["std"] for cp in cps_arr]
    axes[1, 0].errorbar(cps_arr, means_b, yerr=stds_b, marker="o", color="#4472C4")
    axes[1, 0].axhline(1 / 3, color="gray", linestyle="--", label="チャンスレート")
    axes[1, 0].axhline(np.mean(conf01), color="#9BBB59", linestyle=":", label="順応ペア水準")
    axes[1, 0].set_xlabel("逸脱者の学習継続エピソード数")
    axes[1, 0].set_ylabel("精度")
    axes[1, 0].set_title("フェーズB 0→1: agent0が逸脱者を当てる精度")
    axes[1, 0].legend(fontsize=8)

    means_b10 = [track_means_10[cp]["mean"] for cp in cps_arr]
    stds_b10 = [track_means_10[cp]["std"] for cp in cps_arr]
    axes[1, 1].errorbar(cps_arr, means_b10, yerr=stds_b10, marker="o", color="#C0504D")
    axes[1, 1].axhline(1 / 3, color="gray", linestyle="--", label="チャンスレート")
    axes[1, 1].axhline(np.mean(conf10), color="#9BBB59", linestyle=":", label="順応ペア水準")
    axes[1, 1].set_xlabel("逸脱者の学習継続エピソード数")
    axes[1, 1].set_ylabel("精度")
    axes[1, 1].set_title("フェーズB 1→0: 逸脱者がagent0を当てる精度(自身の解釈規則の修正)")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("要件6: 逸脱エージェントによる規範性の検証(n=3)")
    fig.tight_layout()
    fig.savefig("deviant_convention_comparison.png", dpi=150)
    print("グラフを deviant_convention_comparison.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pretrain_chunk":
        pretrain_chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "pretrain_finalize":
        pretrain_finalize(int(sys.argv[2]))
    elif cmd == "test":
        test(int(sys.argv[2]))
    elif cmd == "aggregate":
        aggregate()
