"""
感情AIプロジェクト フェーズ4 プロトタイプ: レガシー本能(要件4後半)
==========================================================

既に恒常性維持を学習済みの「エルダー」に、後継個体(サクセサー)へ知識を教える
"teach"行動を追加し、教えること自体に報酬(レガシー報酬)を与えた場合、
エルダーは自分の恒常性維持を一部犠牲にしてでも教える行動を学習するか(用量反応)、
またその教えが実際にサクセサーの学習を助けるかを検証する。

設計:
  - homeostasis_prototype.ACTIONSへ"teach"を追加する(共有リストへの追記。
    HomeostasisEnv.step()は"teach"を未知の行動として無視し、位置不変という
    点で"stay"と同じに扱われる。センサーの自然な変化(エネルギー減衰等)は
    teach中も進むため、教えることは無償ではなく、恒常性維持を犠牲にする
    本物のトレードオフになる)。
  - teachを選んだステップでは、エルダーのQテーブルから一部のエントリを
    サクセサーのQテーブルへ混合コピーし、エルダー自身にレガシー報酬を追加する。
  - legacy_bonus(レガシー報酬の大きさ)を0(対照群)・1・3と変え、
    (1)エルダーのteach頻度、(2)結果としてサクセサーが受け取る初期知識の質、
    を比較する。

既知の限界:
  - teachは"stay"同様、行動集合に新規追加されるため、Qテーブル上は
    未経験の(state, teach)が初期値0.0を持つ。学習済みの他の行動のQ値は
    負値(報酬が常に負のため)であることが多く、0.0初期化のままだと
    teachが実際には無価値でも「良く見える」楽観的初期化バイアスが生じ、
    最初の実行ではこれによりteachが過剰に選ばれ計算コストも急増した。
    seed_teach_baseline()で各状態の"stay"の値(相当する既存行動の値)に
    揃えることでこの偏りを補正している。
  - サクセサーの評価に用いるinstinct_bias_prototype.trainはteach行動の
    特別な意味を知らないため、サクセサー自身がteachを選んでも"stay"と
    同じに扱われるだけで実害はないが、厳密には行動選択肢が1つ余計にある。
"""

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import homeostasis_prototype as hp
from homeostasis_prototype import HomeostasisEnv, QLearningAgent
import instinct_bias_prototype as ib

# 注意: "teach"の追加は load_base_elder_q() の後で行うこと。
# ここで追加してしまうと、フェーズ1の親と同一条件のはずの基礎Qテーブル学習まで
# 6行動空間に変わってしまい、「フェーズ1の親と同一条件」という前提が崩れる
# (最初の実行時に発生したミス)。

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

ELDER_SEED = 0                    # フェーズ1の親と同じ生育環境
ELDER_EPISODES = 500
ELDER_EPS_DECAY_EPISODES = 300
EVAL_SEED = 2                     # サクセサー評価用の別環境
EVAL_EPISODES = 300
EVAL_EPS_DECAY_EPISODES = 200

LEGACY_BONUSES = [0.0, 1.0, 3.0]  # 0.0=対照群(レガシー報酬なし)
# 転写量の見直し(2026-07-28): 以前はTRANSFER_FRACTION=0.005(約140エントリ/回)を
# 使っていたが、コレクター問題により対照群のteach回数(約6000〜7000回)だけで
# Qテーブル全体(27806エントリ)がほぼ完全にカバーされてしまい、legacy_bonusに
# よる転写量の差が測定できなかった。1回あたりの転写数を固定の小さい値に絞ることで、
# teach回数の違いがカバー率の違いとして残るようにする。
TRANSFER_COUNT = 3                # teach1回あたり転写するエルダーのQエントリ数(固定)
BLEND = 0.3                       # サクセサーの既存値との混合率
REPEATS = 3
RUN_SEEDS = [100, 200, 300]        # 2026-07-29: 他実験(n=3が標準)に揃えるためseed=300を追加


def epsilon_for_episode(ep, decay_episodes, eps_start=1.0, eps_end=0.05):
    frac = min(1.0, ep / decay_episodes)
    return eps_start + (eps_end - eps_start) * frac


def load_base_elder_q():
    """フェーズ1・要件3プロトタイプの親と同一条件で、エルダーの基礎Qテーブルを学習する。"""
    random.seed(ELDER_SEED)
    np.random.seed(ELDER_SEED)
    env = HomeostasisEnv(random.Random(ELDER_SEED))
    agent = QLearningAgent()
    ib.train(env, agent, ib.PARENT_EPISODES, ib.PARENT_EPS_DECAY_EPISODES)
    return agent.q


def seed_teach_baseline(q):
    """
    (state, "teach")の初期値を、その状態の"stay"の値(なければ他行動の平均)に揃える。
    0.0で初期化すると、学習済みの他行動(すべて負値)より不自然に高く見える
    「楽観的初期化バイアス」が生じ、レガシー報酬が無くてもteachが過剰に
    選ばれてしまう(最初の実行で発生し、計算コストの急増も招いた)。
    """
    from collections import defaultdict
    state_vals = defaultdict(list)
    for (s, a), v in q.items():
        state_vals[s].append(v)
    for s, vals in state_vals.items():
        q[(s, "teach")] = q.get((s, "stay"), sum(vals) / len(vals))
    return q


def train_elder_with_teaching(env, elder_q, legacy_bonus, n_episodes, decay_episodes):
    elder = QLearningAgent()
    elder.q = seed_teach_baseline(dict(elder_q))  # 既に学習済みの土台からスタート
    successor = QLearningAgent()  # サクセサーは白紙

    teach_counts, avg_dev, total_rew = [], [], []
    for ep in range(n_episodes):
        state = env.reset()
        eps = epsilon_for_episode(ep, decay_episodes)
        done = False
        devs, rew_sum, teach_count = [], 0.0, 0
        while not done:
            if random.random() < eps:
                action = random.choice(hp.ACTIONS)
            else:
                action = elder.best_action(state)

            next_state, reward, done, deviation = env.step(action)

            if action == "teach":
                teach_count += 1
                reward = reward + legacy_bonus
                keys = list(elder.q.keys())
                if keys:
                    sample_n = min(TRANSFER_COUNT, len(keys))
                    for k in random.sample(keys, sample_n):
                        old = successor.q.get(k, 0.0)
                        successor.q[k] = old + BLEND * (elder.q[k] - old)

            elder.update(state, action, reward, next_state, done)
            state = next_state
            devs.append(deviation)
            rew_sum += reward

        avg_dev.append(float(np.mean(devs)))
        total_rew.append(rew_sum)
        teach_counts.append(teach_count)

    return elder, successor, avg_dev, total_rew, teach_counts


def evaluate_successor(successor_q, reference_size):
    """
    受け取った知識を初期値として、サクセサーを別環境で評価する(要件3プロトタイプと同じ手法)。
    要件3プロトタイプでは、本能が強いほど探索率を下げて自分のQ値を初手から信頼させる
    補正が必要だった(でないと序盤の完全ランダム行動で転写した知識が行動選択に反映
    されない)。ここでも同じ補正を、サクセサーが受け取ったQテーブルのカバー率
    (受け取ったエントリ数 / エルダーの参照サイズ)に応じて適用する。
    """
    coverage = min(1.0, len(successor_q) / reference_size) if reference_size else 0.0
    eps_start = 1.0 - 0.7 * coverage
    random.seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    env = HomeostasisEnv(random.Random(EVAL_SEED))
    agent = QLearningAgent()
    agent.q = dict(successor_q)
    avg_dev, total_rew = ib.train(env, agent, EVAL_EPISODES, EVAL_EPS_DECAY_EPISODES, eps_start)
    return avg_dev, total_rew, coverage


if __name__ == "__main__":
    base_elder_q = load_base_elder_q()
    print(f"エルダー基礎Qエントリ数: {len(base_elder_q)}")

    if "teach" not in hp.ACTIONS:
        hp.ACTIONS.append("teach")

    results = {}
    for legacy_bonus in LEGACY_BONUSES:
        teach_rate_runs, succ_first50_runs, coverage_runs = [], [], []
        for seed in RUN_SEEDS:
            random.seed(seed)
            np.random.seed(seed)
            env = HomeostasisEnv(random.Random(ELDER_SEED))
            elder, successor, avg_dev, total_rew, teach_counts = train_elder_with_teaching(
                env, base_elder_q, legacy_bonus, ELDER_EPISODES, ELDER_EPS_DECAY_EPISODES
            )
            teach_rate = np.mean(teach_counts[-100:]) / hp.MAX_STEPS
            teach_rate_runs.append(teach_rate)

            succ_avg_dev, succ_total_rew, coverage = evaluate_successor(successor.q, len(elder.q))
            succ_first50_runs.append(np.mean(succ_avg_dev[:50]))
            coverage_runs.append(coverage)

            print(
                f"legacy_bonus={legacy_bonus}, seed={seed}: "
                f"teach頻度(終盤100ep)={teach_rate:.4f}, "
                f"サクセサーカバー率={coverage:.4f}, "
                f"サクセサー最初50ep平均逸脱={np.mean(succ_avg_dev[:50]):.4f}, "
                f"サクセサーQエントリ数={len(successor.q)}"
            )

        results[legacy_bonus] = {
            "teach_rate_mean": float(np.mean(teach_rate_runs)),
            "teach_rate_std": float(np.std(teach_rate_runs)),
            "coverage_mean": float(np.mean(coverage_runs)),
            "coverage_std": float(np.std(coverage_runs)),
            "succ_first50_mean": float(np.mean(succ_first50_runs)),
            "succ_first50_std": float(np.std(succ_first50_runs)),
            "teach_rate_runs": [float(x) for x in teach_rate_runs],
            "coverage_runs": [float(x) for x in coverage_runs],
            "succ_first50_runs": [float(x) for x in succ_first50_runs],
        }

    print("\n=== まとめ(n=3) ===")
    for legacy_bonus, r in results.items():
        print(
            f"legacy_bonus={legacy_bonus}: "
            f"teach頻度={r['teach_rate_mean']:.4f}±{r['teach_rate_std']:.4f}, "
            f"カバー率={r['coverage_mean']:.4f}±{r['coverage_std']:.4f}, "
            f"サクセサー最初50ep平均逸脱={r['succ_first50_mean']:.4f}±{r['succ_first50_std']:.4f}"
        )

    import json
    with open("legacy_instinct_n3_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    print("saved legacy_instinct_n3_results.json")

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
    axes[0].set_title("レガシー報酬が強いほど教える頻度は上がるか")

    axes[1].bar([str(b) for b in bonuses], cov_means, yerr=cov_stds, color="#9BBB59")
    axes[1].set_xlabel("レガシー報酬(legacy_bonus)")
    axes[1].set_ylabel("サクセサーのカバー率")
    axes[1].set_title("転写されたQテーブルのカバー率")

    axes[2].bar([str(b) for b in bonuses], succ_means, yerr=succ_stds, color="#C0504D")
    axes[2].set_xlabel("レガシー報酬(legacy_bonus)")
    axes[2].set_ylabel("サクセサー最初50ep平均逸脱(小さいほど良い)")
    axes[2].set_title("エルダーが教えた結果、サクセサーは早く恒常性を保てるか")

    fig.suptitle(f"要件4後半プロトタイプ: レガシー本能の用量反応(n={len(RUN_SEEDS)})")
    fig.tight_layout()
    fig.savefig("legacy_instinct_comparison.png", dpi=150)
    print("グラフを legacy_instinct_comparison.png に保存しました。")
