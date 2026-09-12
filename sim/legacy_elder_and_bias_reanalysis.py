"""
既存ログ再解析: 2つの新仮説の検証(新規学習なし)
==========================================================

転写回数・教示タイミング・アーキテクチャ(破滅的干渉)では説明しきれなかった
要件4単一世代レガシー本能の用量反応逆転について、以下2つの仮説を、
既存のnn_legacy_split_state_b{bonus}_s{seed}.pkl(教示ヘッド分離実験)・
nn_legacy_tc_state_{A,B}_s{seed}.pkl(転写回数操作実験)の中間状態
(いずれも完了後の削除がサンドボックスの制限で失敗し、偶然残存していたもの)
を再解析するだけで検証する。新規学習は一切行わない。

(a) エルダー自身の実力低下: 各runのavg_dev(教示フェーズ500エピソード分、
    エピソードごとの平均ホメオスタシス逸脱)の終盤50エピソード平均を
    「エルダー自身の収束後の恒常性成績」とみなし、bonus水準ごとに比較する。
    avg_devはteach行動そのものへの報酬(legacy_bonus)とは無関係な環境指標
    (センサー逸脱の生値)であり、教示に費やした時間が資源探索に使われなかった
    ことの影響を直接反映する。

(b) サクセサーの教示偏りの汎化: 各runの教示フェーズ終了時点の
    successor(distillation直後、評価フェーズでの追加学習が始まる前の状態、
    st["successor"])のQネットワークを使い、elder_visited(エルダーが実際に
    訪れた状態集合)の各状態について、6次元連結Qベクトルのargmaxが
    "teach"になる割合を計算する。これは新規学習を伴わない、既存の重みへの
    フォワードパス(推論)のみの計算であり、「教示すべきでない状況を含めて
    どれだけteachを選びがちか」を、評価フェーズを実際に走らせずに直接測る。

使い方: python3 legacy_elder_and_bias_reanalysis.py
"""

import pickle
import json
import numpy as np
from scipy import stats

import __main__
import legacy_instinct_nn_prototype as base
import legacy_instinct_nn_splithead_prototype as sp
for name in ["MLPParamsGen"]:
    setattr(__main__, name, getattr(base, name))
for name in ["NNAgentSplit", "SuccessorNetSplit"]:
    setattr(__main__, name, getattr(sp, name))
import homeostasis_prototype as hp
if "teach" not in hp.ACTIONS:
    hp.ACTIONS.append("teach")
import homeostasis_nn_prototype as m

ACTIONS = hp.ACTIONS
TEACH_IDX = ACTIONS.index("teach")

BONUS_RUNS = {
    0.0: [("nn_legacy_split_state_b0_s100.pkl", 100), ("nn_legacy_split_state_b0_s200.pkl", 200), ("nn_legacy_split_state_b0_s300.pkl", 300)],
    1.0: [("nn_legacy_split_state_b1_s100.pkl", 100), ("nn_legacy_split_state_b1_s200.pkl", 200), ("nn_legacy_split_state_b1_s300.pkl", 300)],
    3.0: [("nn_legacy_split_state_b3_s100.pkl", 100), ("nn_legacy_split_state_b3_s200.pkl", 200), ("nn_legacy_split_state_b3_s300.pkl", 300)],
}
TC_RUNS = {
    "A": [("nn_legacy_tc_state_A_s100.pkl", 100), ("nn_legacy_tc_state_A_s200.pkl", 200), ("nn_legacy_tc_state_A_s300.pkl", 300)],
    "B": [("nn_legacy_tc_state_B_s100.pkl", 100), ("nn_legacy_tc_state_B_s200.pkl", 200), ("nn_legacy_tc_state_B_s300.pkl", 300)],
}


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def elder_final_dev(st):
    avg_dev = np.array(st["avg_dev"], dtype=float)
    return float(np.mean(avg_dev[-50:]))


def teach_argmax_rate(st):
    successor = st["successor"]
    states = list(st["elder_visited"])
    X = np.stack([m.encode_state(s) for s in states])
    q_move, _ = m.forward(successor.move_params, X)
    q_teach, _ = m.forward(successor.teach_params, X)
    combined = np.concatenate([q_move, q_teach], axis=1)
    argmax_idx = np.argmax(combined, axis=1)
    return float(np.mean(argmax_idx == TEACH_IDX)), len(states)


def analyze(label_runs):
    results = {}
    for label, runs in label_runs.items():
        dev_vals, bias_vals = [], []
        for path, seed in runs:
            st = load(path)
            dev = elder_final_dev(st)
            bias, n_states = teach_argmax_rate(st)
            dev_vals.append(dev)
            bias_vals.append(bias)
            print(f"[{label} seed={seed}] エルダー終盤50ep平均逸脱={dev:.4f}, "
                  f"サクセサーのteach argmax率={bias:.4f} (n_states={n_states})")
        results[label] = {
            "elder_final_dev_mean": float(np.mean(dev_vals)), "elder_final_dev_std": float(np.std(dev_vals)),
            "teach_argmax_rate_mean": float(np.mean(bias_vals)), "teach_argmax_rate_std": float(np.std(bias_vals)),
            "elder_final_dev_runs": dev_vals, "teach_argmax_rate_runs": bias_vals,
        }
        print(f"=== {label}: エルダー終盤逸脱={np.mean(dev_vals):.4f}±{np.std(dev_vals):.4f}, "
              f"teach argmax率={np.mean(bias_vals):.4f}±{np.std(bias_vals):.4f} ===\n")
    return results


def main():
    print("########## (a)(b) bonus水準ごとの再解析 ##########")
    bonus_results = analyze(BONUS_RUNS)

    dev0 = bonus_results[0.0]["elder_final_dev_runs"]
    dev1 = bonus_results[1.0]["elder_final_dev_runs"]
    dev3 = bonus_results[3.0]["elder_final_dev_runs"]
    bias0 = bonus_results[0.0]["teach_argmax_rate_runs"]
    bias1 = bonus_results[1.0]["teach_argmax_rate_runs"]
    bias3 = bonus_results[3.0]["teach_argmax_rate_runs"]

    print("--- (a) エルダー終盤逸脱: bonusが高いほど悪化(増加)するか ---")
    t01, p01 = stats.ttest_ind(dev1, dev0)
    t13, p13 = stats.ttest_ind(dev3, dev1)
    t03, p03 = stats.ttest_ind(dev3, dev0)
    print(f"bonus0={np.mean(dev0):.4f}, bonus1={np.mean(dev1):.4f}, bonus3={np.mean(dev3):.4f}")
    print(f"t検定(n=3): 1vs0 t={t01:.3f} p={p01:.4f}, 3vs1 t={t13:.3f} p={p13:.4f}, 3vs0 t={t03:.3f} p={p03:.4f}")

    print("\n--- (b) サクセサーのteach argmax率: bonusが高いほど過剰選択するか ---")
    t01b, p01b = stats.ttest_ind(bias1, bias0)
    t13b, p13b = stats.ttest_ind(bias3, bias1)
    t03b, p03b = stats.ttest_ind(bias3, bias0)
    print(f"bonus0={np.mean(bias0):.4f}, bonus1={np.mean(bias1):.4f}, bonus3={np.mean(bias3):.4f}")
    print(f"t検定(n=3): 1vs0 t={t01b:.3f} p={p01b:.4f}, 3vs1 t={t13b:.3f} p={p13b:.4f}, 3vs0 t={t03b:.3f} p={p03b:.4f}")

    print("\n########## 転写回数操作(条件A/B)側での確認 ##########")
    tc_results = analyze(TC_RUNS)

    out = {"by_bonus": bonus_results, "by_tc_condition": tc_results}
    with open("legacy_elder_and_bias_reanalysis_results.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved legacy_elder_and_bias_reanalysis_results.json")


if __name__ == "__main__":
    main()
