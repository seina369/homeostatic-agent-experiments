"""
系統0の詰まった2世代(世代1・世代4)だけを対象に、育成フェーズの探索率に下限を
設ける修正(eps_start = max(EPS_FLOOR, 1.0 - 0.7*coverage))で不完全収束が
解消するかを確認する、小規模な診断テスト。

教示フェーズ(elder/successorの生成)は元の実験と全く同じ乱数シードで再現し、
育成フェーズのeps_startの式だけを変えて、同じseedで比較する。
"""

import sys, json, pickle
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import numpy as np

import homeostasis_prototype as hp
from homeostasis_prototype import HomeostasisEnv, QLearningAgent
import instinct_bias_prototype as ib
from legacy_instinct_prototype import train_elder_with_teaching

BASE_MAP_SEED = 0
GROW_MAP_SEED = 2
LEGACY_BONUS = 3.0
TEACH_EPISODES = 500
TEACH_DECAY = 300
GROW_EPISODES = 3000
GROW_DECAY = 2000
EPS_FLOOR = 0.6

ELDER_Q_FILE = "multigen_elder_q_L{lineage}_gen{gen}.pkl"


def grow_successor_with_floor(successor_q, reference_size, seed, eps_floor):
    coverage = min(1.0, len(successor_q) / reference_size) if reference_size else 0.0
    eps_start_orig = 1.0 - 0.7 * coverage
    eps_start = max(eps_floor, eps_start_orig)
    random.seed(seed)
    np.random.seed(seed)
    env = HomeostasisEnv(random.Random(GROW_MAP_SEED))
    agent = QLearningAgent()
    agent.q = dict(successor_q)
    avg_dev, total_rew = ib.train(env, agent, GROW_EPISODES, GROW_DECAY, eps_start)
    return agent.q, avg_dev, total_rew, coverage, eps_start_orig, eps_start


def test_generation(lineage, gen):
    seed_base = lineage * 100000
    if "teach" not in hp.ACTIONS:
        hp.ACTIONS.append("teach")

    with open(ELDER_Q_FILE.format(lineage=lineage, gen=gen - 1), "rb") as f:
        elder_q = pickle.load(f)

    random.seed(seed_base + gen * 100)
    np.random.seed(seed_base + gen * 100)
    teach_env = HomeostasisEnv(random.Random(BASE_MAP_SEED))
    elder, successor, avg_dev_elder, total_rew_elder, teach_counts = train_elder_with_teaching(
        teach_env, elder_q, LEGACY_BONUS, TEACH_EPISODES, TEACH_DECAY
    )

    grown_q, grow_dev, grow_rew, coverage, eps_orig, eps_fixed = grow_successor_with_floor(
        successor.q, len(elder.q), seed=seed_base + gen * 100 + 1, eps_floor=EPS_FLOOR
    )
    last50 = float(np.mean(grow_dev[-50:]))
    first50 = float(np.mean(grow_dev[:50]))
    print(f"[L{lineage}] 世代{gen}: カバー率={coverage:.4f}, eps_start(元)={eps_orig:.4f} -> eps_start(修正後)={eps_fixed:.4f}, "
          f"頭出し(最初50ep)={first50:.4f}, 独り立ち後(最後50ep)={last50:.4f}")

    result = {
        "lineage": lineage, "generation": gen, "coverage": coverage,
        "eps_start_orig": eps_orig, "eps_start_fixed": eps_fixed,
        "head_start_first50_fixed": first50, "grown_last50_fixed": last50,
    }
    fname = "epsfloor_regression_check.json"
    try:
        with open(fname) as f:
            records = json.load(f)
    except FileNotFoundError:
        records = []
    records = [r for r in records if not (r["lineage"] == lineage and r["generation"] == gen)] + [result]
    with open(fname, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return last50


if __name__ == "__main__":
    lineage = int(sys.argv[1])
    gen = int(sys.argv[2])
    print(f"=== 探索率下限修正テスト(EPS_FLOOR={EPS_FLOOR}) 系統{lineage} 世代{gen} ===")
    test_generation(lineage, gen)
