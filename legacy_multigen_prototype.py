"""
感情AIプロジェクト フェーズ4 追加プロトタイプ: レガシー本能(要件4後半)の複数世代検証
==========================================================

legacy_instinct_prototype.pyでは、エルダー→サクセサーの1世代分の引き継ぎのみを
検証していた(転写量設計の見直し後、legacy_bonusに応じた用量反応を確認済み)。
本プロトタイプはこれを世代を超えて連鎖させる: 世代gのサクセサーは、引き継いだ
知識を初期値として自分の人生を経験して「育ち」(要件3プロトタイプと同じ、
カバー率に応じた探索率補正つきの学習)、育ったら次の世代gen+1に対する
エルダーとして教える側に回る。これをN_GENERATIONS世代繰り返し、世代を追うごとに
「引き継いだ直後の頭出し性能」が改善・横ばい・劣化のどれになるかを見る。

1系統目(lineage=0)の結果、独り立ち後の到達点に大きなばらつき(0.08〜0.62、
5世代中3世代は完全収束・2世代は不完全収束)が見つかった。これが本物の再現性のある
現象かノイズかを確かめるため、マップ構成(BASE_MAP_SEED, GROW_MAP_SEED)は固定した
まま、学習系列の乱数だけを変えたlineage(系統)を複数走らせられるようにした。

使い方:
  python3 legacy_multigen_prototype.py init <lineage>            # そのlineageの創始エルダーを学習・保存
  python3 legacy_multigen_prototype.py gen <lineage> <N>          # lineageの世代Nの継承・育成を実行
  python3 legacy_multigen_prototype.py aggregate <lineage>        # そのlineageの記録を集計・グラフ化
  python3 legacy_multigen_prototype.py aggregate_all <lineage...> # 複数lineageをまとめて集計・グラフ化
"""

import sys, json, pickle
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import homeostasis_prototype as hp
from homeostasis_prototype import HomeostasisEnv, QLearningAgent
import instinct_bias_prototype as ib
from legacy_instinct_prototype import train_elder_with_teaching

for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
    try:
        fm.fontManager.addfont(_p)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

BASE_MAP_SEED = 0     # エルダーのホームマップ(teach・自身の学習を続ける場所、全lineage共通)
GROW_MAP_SEED = 2      # サクセサーが「育つ」別環境(全lineage共通)
LEGACY_BONUS = 3.0     # 前回の実験で最も恩恵が大きかった値
TEACH_EPISODES = 500
TEACH_DECAY = 300
GROW_EPISODES = 3000   # 要件3プロトタイプ(instinct_bias)で収束が確認された設定に合わせる
GROW_DECAY = 2000
N_GENERATIONS = 5

ELDER_Q_FILE = "multigen_elder_q_L{lineage}_gen{gen}.pkl"
RECORDS_FILE = "multigen_records_L{lineage}.json"


def init_founder(lineage):
    seed_base = lineage * 100000
    random.seed(seed_base)
    np.random.seed(seed_base)
    env = HomeostasisEnv(random.Random(BASE_MAP_SEED))
    agent = QLearningAgent()
    ib.train(env, agent, ib.PARENT_EPISODES, ib.PARENT_EPS_DECAY_EPISODES)  # 3000ep, 5行動
    print(f"[L{lineage}] 創始エルダー(世代0)学習完了。Qエントリ数={len(agent.q)}")
    with open(ELDER_Q_FILE.format(lineage=lineage, gen=0), "wb") as f:
        pickle.dump(agent.q, f)
    print(f"saved {ELDER_Q_FILE.format(lineage=lineage, gen=0)}")


def grow_successor(successor_q, reference_size, seed):
    coverage = min(1.0, len(successor_q) / reference_size) if reference_size else 0.0
    eps_start = 1.0 - 0.7 * coverage
    random.seed(seed)
    np.random.seed(seed)
    env = HomeostasisEnv(random.Random(GROW_MAP_SEED))
    agent = QLearningAgent()
    agent.q = dict(successor_q)
    avg_dev, total_rew = ib.train(env, agent, GROW_EPISODES, GROW_DECAY, eps_start)
    return agent.q, avg_dev, total_rew, coverage


def run_generation(lineage, gen):
    if "teach" not in hp.ACTIONS:
        hp.ACTIONS.append("teach")

    seed_base = lineage * 100000
    with open(ELDER_Q_FILE.format(lineage=lineage, gen=gen - 1), "rb") as f:
        elder_q = pickle.load(f)

    random.seed(seed_base + gen * 100)
    np.random.seed(seed_base + gen * 100)
    teach_env = HomeostasisEnv(random.Random(BASE_MAP_SEED))
    elder, successor, avg_dev_elder, total_rew_elder, teach_counts = train_elder_with_teaching(
        teach_env, elder_q, LEGACY_BONUS, TEACH_EPISODES, TEACH_DECAY
    )
    teach_rate = float(np.mean(teach_counts[-100:]) / hp.MAX_STEPS)

    grown_q, grow_dev, grow_rew, coverage = grow_successor(
        successor.q, len(elder.q), seed=seed_base + gen * 100 + 1
    )
    head_start_first50 = float(np.mean(grow_dev[:50]))
    grown_last50 = float(np.mean(grow_dev[-50:]))

    record = {
        "lineage": lineage,
        "generation": gen,
        "elder_q_size": len(elder.q),
        "successor_raw_q_size": len(successor.q),
        "grown_q_size": len(grown_q),
        "coverage": coverage,
        "teach_rate": teach_rate,
        "head_start_first50_dev": head_start_first50,
        "grown_last50_dev": grown_last50,
    }
    print(f"[L{lineage}] 世代{gen}: {json.dumps(record, ensure_ascii=False)}")

    with open(ELDER_Q_FILE.format(lineage=lineage, gen=gen), "wb") as f:
        pickle.dump(grown_q, f)

    fname = RECORDS_FILE.format(lineage=lineage)
    try:
        with open(fname) as f:
            records = json.load(f)
    except FileNotFoundError:
        records = []
    records = [r for r in records if r["generation"] != gen] + [record]
    records.sort(key=lambda r: r["generation"])
    with open(fname, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"saved generation {gen} record to {fname}")


def load_records(lineage):
    with open(RECORDS_FILE.format(lineage=lineage)) as f:
        records = json.load(f)
    records.sort(key=lambda r: r["generation"])
    return records


def aggregate(lineage):
    records = load_records(lineage)
    print(f"=== lineage {lineage}: 世代ごとの記録 ===")
    for r in records:
        print(
            f"世代{r['generation']}: teach頻度={r['teach_rate']:.4f}, カバー率={r['coverage']:.4f}, "
            f"頭出し(最初50ep)逸脱={r['head_start_first50_dev']:.4f}, "
            f"独り立ち後(最後50ep)逸脱={r['grown_last50_dev']:.4f}"
        )


def aggregate_all(lineages):
    all_records = []
    for lineage in lineages:
        all_records.extend(load_records(lineage))

    print("=== 全lineage集計 ===")
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

    # 世代番号ごとの傾向(全lineageをまたいで平均)も見る
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
    axes[0].set_title("系統ごとの独り立ち後の到達点")
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(range(1, N_GENERATIONS + 1))

    axes[1].hist(grown_vals, bins=10, color="#4472C4", alpha=0.75)
    axes[1].axvline(0.2, color="gray", linestyle="--", linewidth=1, label="収束の目安(0.2)")
    axes[1].set_xlabel("独り立ち後(最後50ep)平均逸脱")
    axes[1].set_ylabel("度数(全lineage×全世代)")
    axes[1].set_title(f"到達点の分布(全{len(all_records)}件)")
    axes[1].legend()

    fig.suptitle(f"要件4後半 複数世代検証: 複数系統({len(lineages)}系統×{N_GENERATIONS}世代)での再現性")
    fig.tight_layout()
    fig.savefig("legacy_multigen_all_lineages.png", dpi=150)
    print("グラフを legacy_multigen_all_lineages.png に保存しました。")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "init":
        init_founder(int(sys.argv[2]))
    elif cmd == "gen":
        run_generation(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "aggregate":
        aggregate(int(sys.argv[2]))
    elif cmd == "aggregate_all":
        aggregate_all([int(a) for a in sys.argv[2:]])
