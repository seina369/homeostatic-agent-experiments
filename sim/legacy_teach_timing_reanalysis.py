"""
既存ログ再解析: 「legacy_bonusが高いほど教示(distillation)頻度が上がり、
その結果エルダーの未収束なQ値をより多くサクセサーに転写してしまう」仮説の検証。
新規学習は一切行わず、既存のnn_legacy_split_state_b{bonus}_s{seed}.pkl
(教示ヘッド分離実験のチャンク実行途中状態。完了後にos.remove()で削除される
はずだったが、このサンドボックスではファイル削除がPermissionErrorで
失敗するため、結果として全run分の中間状態がそのまま残存していた)に含まれる
teach_counts(各エピソードでのteach行動回数、500エピソード分)・avg_dev
(各エピソードの平均ホメオスタシス逸脱、500エピソード分)だけを読み直す。

**データ可用性の確認結果(先に記録)**:
- 共有ヘッド版(legacy_instinct_nn_prototype.py、nn_legacy_instinct_n3/n15_results.json
  の元データ)は、run_chunkが単発実行(チャンク保存なし)で設計されていたため、
  各run(bonus×seed)について最終集計値(teach_rate・coverage・succ_first50_dev
  の4フィールドのみ)しか保存されておらず、教示イベントごとのエピソード番号や
  エルダーのQ値安定性は一切記録されていない。よってこの再解析は共有ヘッド版
  には適用できない(新規計測付きの再実行が別途必要)。
- 教示ヘッド分離版(legacy_instinct_nn_splithead_prototype.py)は、時間主導
  チャンク実行のため中間状態をpickle保存する設計になっており、完了後に
  該当pklをos.remove()するはずが、このサンドボックスのファイル削除制限により
  削除に失敗して残存していた(意図せぬ副産物)。この中間状態にteach_counts
  (エピソード単位)・avg_dev(エピソード単位)が含まれているため、教示イベントの
  時間分布(設問1)については再解析が可能。ただしTD誤差やQ値の分散そのものは
  記録されておらず、「エルダー自身のQ値の安定性」の直接指標は存在しない。
  代替として、エピソード単位の平均ホメオスタシス逸脱(avg_dev)の変動性
  (隣接エピソード間の絶対差)を、エルダーの方策がまだ変化し続けている
  (=Q値が収束しきっていない)ことの間接的な代理指標として用いる。

使い方: python3 legacy_teach_timing_reanalysis.py
"""

import pickle, json
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

BONUSES = [0.0, 1.0, 3.0]
SEEDS = [100, 200, 300]


def load_state(bonus, seed):
    tag = f"b{int(bonus)}_s{seed}"
    with open(f"nn_legacy_split_state_{tag}.pkl", "rb") as f:
        st = pickle.load(f)
    return st


def analyze_run(st):
    teach_counts = np.array(st["teach_counts"], dtype=float)  # 500
    avg_dev = np.array(st["avg_dev"], dtype=float)  # 500
    n = len(teach_counts)
    ep_frac = np.arange(n) / (n - 1)  # 0(序盤)〜1(終盤)

    total_teach = teach_counts.sum()
    # 設問1: 教示イベントの時間分布(teach回数で重み付けした平均進捗割合)
    weighted_frac_mean = float(np.sum(teach_counts * ep_frac) / total_teach) if total_teach > 0 else float("nan")
    # 分散も(重み付き)
    weighted_frac_var = float(np.sum(teach_counts * (ep_frac - weighted_frac_mean) ** 2) / total_teach) if total_teach > 0 else float("nan")

    # 設問2代理指標: 隣接エピソード間のavg_devの絶対差(方策がまだ変化中=不安定の代理)
    instability = np.abs(np.diff(avg_dev, prepend=avg_dev[0]))  # 長さn、instability[0]=0
    overall_instability_mean = float(np.mean(instability))
    weighted_instability_mean = float(np.sum(teach_counts * instability) / total_teach) if total_teach > 0 else float("nan")
    # 「teach時点は平均的なエピソードより不安定か」の比
    instability_ratio = weighted_instability_mean / overall_instability_mean if overall_instability_mean > 0 else float("nan")

    return {
        "total_teach_events": float(total_teach),
        "teach_weighted_progress_frac_mean": weighted_frac_mean,
        "teach_weighted_progress_frac_std": float(np.sqrt(weighted_frac_var)),
        "overall_instability_mean": overall_instability_mean,
        "teach_weighted_instability_mean": weighted_instability_mean,
        "instability_ratio_teach_vs_overall": instability_ratio,
    }


def main():
    results = {}
    per_bonus = {}
    for bonus in BONUSES:
        runs = []
        for seed in SEEDS:
            st = load_state(bonus, seed)
            r = analyze_run(st)
            r["seed"] = seed
            runs.append(r)
            print(f"[bonus={bonus} seed={seed}] teach総数={r['total_teach_events']:.0f}, "
                  f"教示時点の平均進捗割合={r['teach_weighted_progress_frac_mean']:.4f}, "
                  f"教示時点の不安定度/全体平均不安定度比={r['instability_ratio_teach_vs_overall']:.4f}")
        per_bonus[bonus] = runs
        frac_vals = [r["teach_weighted_progress_frac_mean"] for r in runs]
        ratio_vals = [r["instability_ratio_teach_vs_overall"] for r in runs]
        results[str(bonus)] = {
            "runs": runs,
            "progress_frac_mean": float(np.mean(frac_vals)), "progress_frac_std": float(np.std(frac_vals)),
            "instability_ratio_mean": float(np.mean(ratio_vals)), "instability_ratio_std": float(np.std(ratio_vals)),
        }
        print(f"=== bonus={bonus}: 教示時点の平均進捗割合(n=3)={np.mean(frac_vals):.4f}±{np.std(frac_vals):.4f}, "
              f"不安定度比(n=3)={np.mean(ratio_vals):.4f}±{np.std(ratio_vals):.4f} ===\n")

    # bonus間の比較(小標本のため参考値として)
    frac_0 = [r["teach_weighted_progress_frac_mean"] for r in per_bonus[0.0]]
    frac_1 = [r["teach_weighted_progress_frac_mean"] for r in per_bonus[1.0]]
    frac_3 = [r["teach_weighted_progress_frac_mean"] for r in per_bonus[3.0]]
    ratio_0 = [r["instability_ratio_teach_vs_overall"] for r in per_bonus[0.0]]
    ratio_1 = [r["instability_ratio_teach_vs_overall"] for r in per_bonus[1.0]]
    ratio_3 = [r["instability_ratio_teach_vs_overall"] for r in per_bonus[3.0]]

    print("--- 教示時点の平均進捗割合(0=序盤, 1=終盤): bonusが高いほど小さい(早期偏り)か ---")
    t01, p01 = stats.ttest_ind(frac_1, frac_0)
    t13, p13 = stats.ttest_ind(frac_3, frac_1)
    t03, p03 = stats.ttest_ind(frac_3, frac_0)
    print(f"bonus0={np.mean(frac_0):.4f}, bonus1={np.mean(frac_1):.4f}, bonus3={np.mean(frac_3):.4f}")
    print(f"t検定(参考、n=3): 1vs0 t={t01:.3f} p={p01:.4f}, 3vs1 t={t13:.3f} p={p13:.4f}, 3vs0 t={t03:.3f} p={p03:.4f}")

    print("\n--- 教示時点の不安定度比(1.0=平均並み、>1で教示時点はより不安定): bonusが高いほど大きいか ---")
    t01b, p01b = stats.ttest_ind(ratio_1, ratio_0)
    t13b, p13b = stats.ttest_ind(ratio_3, ratio_1)
    t03b, p03b = stats.ttest_ind(ratio_3, ratio_0)
    print(f"bonus0={np.mean(ratio_0):.4f}, bonus1={np.mean(ratio_1):.4f}, bonus3={np.mean(ratio_3):.4f}")
    print(f"t検定(参考、n=3): 1vs0 t={t01b:.3f} p={p01b:.4f}, 3vs1 t={t13b:.3f} p={p13b:.4f}, 3vs0 t={t03b:.3f} p={p03b:.4f}")

    with open("legacy_teach_timing_reanalysis_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nsaved legacy_teach_timing_reanalysis_results.json")


if __name__ == "__main__":
    main()
