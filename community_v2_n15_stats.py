"""
感情AIプロジェクト フェーズ6 追加検証: 要件6 新参者効果のn拡大統計検定
==========================================================

前回(n=3、traj_seed=0,11,22)で見られた「既存の慣習(agent0の収束済みQテーブル)に
触れた新参者(Arm3)が、慣習に触れていない対照群(Arm2、探索率スケジュールは
完全に一致した新人パートナーと組む)を上回る」効果を、サンプル数を増やして
統計的に確認する。条件は前回成功した設定(community_signal_v2_prototype.py、
衝突ペナルティ8.0、推測ゲームによる信号への直接報酬付け)をそのまま使い、
乱数シード(traj_seed)だけを変えて15系統(既存3+新規12=15、traj_seed=
0,1,2,3,4,5,6,7,8,9,10,11,12,13,22)で繰り返した。

各系統についてArm2(対照、新人パートナー)とArm3(処置、既存agent0)の衝突率を
比較し、対応のあるt検定(paired t-test)・Wilcoxon符号順位検定・対応のある
Cohenのd・平均差の95%信頼区間を、最初100/300/800epの3つの観測窓それぞれで
算出する。Arm2・Arm3は同じtraj_seedグループ内で比較する(traj_seedごとに
乱数系列全体が変わるため、系統をペアの単位として扱うのが自然)。

使い方:
  python3 community_v2_n15_stats.py
"""

import json
import numpy as np
from scipy import stats

TRAJ_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22]
WINDOWS = [100, 300, 800]


def load_all():
    data = []
    for seed in TRAJ_SEEDS:
        with open(f"community_v2_newcomer_seed{seed}.json") as f:
            data.append(json.load(f))
    return data


def cohens_d_paired(diffs):
    diffs = np.asarray(diffs, dtype=float)
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def main():
    data = load_all()
    n = len(data)
    print(f"=== n={n}系統(traj_seed={TRAJ_SEEDS})でのArm2 vs Arm3 統計検定 ===\n")

    results = {}
    for w in WINDOWS:
        arm2 = np.array([np.mean(d["arm2_control_collision_rate_history"][:w]) for d in data])
        arm3 = np.array([np.mean(d["arm3_treatment_collision_rate_history"][:w]) for d in data])
        diffs = arm2 - arm3  # 正ならArm3(既存agent0)の方が衝突率が低い=有利

        t_stat, t_p = stats.ttest_rel(arm2, arm3)
        try:
            w_stat, w_p = stats.wilcoxon(arm2, arm3)
        except ValueError:
            w_stat, w_p = None, None
        d = cohens_d_paired(diffs)

        mean_diff = float(np.mean(diffs))
        se_diff = float(np.std(diffs, ddof=1) / np.sqrt(n))
        ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=mean_diff, scale=se_diff)

        n_seed_favor_arm3 = int(np.sum(diffs > 0))

        results[w] = {
            "n": n,
            "arm2_mean": float(np.mean(arm2)), "arm2_std": float(np.std(arm2)),
            "arm3_mean": float(np.mean(arm3)), "arm3_std": float(np.std(arm3)),
            "mean_diff_arm2_minus_arm3": mean_diff,
            "ci95_low": float(ci_low), "ci95_high": float(ci_high),
            "paired_t_stat": float(t_stat), "paired_t_pvalue": float(t_p),
            "wilcoxon_stat": float(w_stat) if w_stat is not None else None,
            "wilcoxon_pvalue": float(w_p) if w_p is not None else None,
            "cohens_d_paired": d,
            "n_seeds_favoring_arm3": n_seed_favor_arm3,
        }

        print(f"[最初{w}ep] Arm2(対照)={np.mean(arm2):.4f}±{np.std(arm2):.4f}, "
              f"Arm3(処置)={np.mean(arm3):.4f}±{np.std(arm3):.4f}")
        print(f"  平均差(Arm2-Arm3)={mean_diff:+.4f} (95%CI [{ci_low:+.4f}, {ci_high:+.4f}])")
        print(f"  対応のあるt検定: t={t_stat:.3f}, p={t_p:.4f}")
        if w_stat is not None:
            print(f"  Wilcoxon符号順位検定: W={w_stat:.3f}, p={w_p:.4f}")
        print(f"  対応のあるCohen's d={d:.3f}")
        print(f"  Arm3が有利だった系統数: {n_seed_favor_arm3}/{n}\n")

    with open("community_v2_n15_stats_summary.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved community_v2_n15_stats_summary.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for _p in ["/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"]:
        try:
            fm.fontManager.addfont(_p)
        except Exception:
            pass
    matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    xw = np.arange(len(WINDOWS))
    width = 0.35
    arm2_means = [results[w]["arm2_mean"] for w in WINDOWS]
    arm2_stds = [results[w]["arm2_std"] for w in WINDOWS]
    arm3_means = [results[w]["arm3_mean"] for w in WINDOWS]
    arm3_stds = [results[w]["arm3_std"] for w in WINDOWS]
    axes[0].bar(xw - width / 2, arm2_means, width, yerr=arm2_stds, label="Arm2:新人パートナー対照", color="#9BBB59")
    axes[0].bar(xw + width / 2, arm3_means, width, yerr=arm3_stds, label="Arm3:既存agent0処置", color="#4472C4")
    axes[0].set_xticks(xw)
    axes[0].set_xticklabels([f"最初{w}ep" for w in WINDOWS])
    axes[0].set_ylabel("平均衝突率")
    axes[0].set_title(f"新参者3群比較(n={n})")
    axes[0].legend(fontsize=8)

    for i, w in enumerate(WINDOWS):
        r = results[w]
        p_str = f"p={r['paired_t_pvalue']:.3f}"
        axes[0].text(i, max(arm2_means[i], arm3_means[i]) + max(arm2_stds[i], arm3_stds[i]) + 0.002,
                     p_str, ha="center", fontsize=8)

    d_vals = [results[w]["cohens_d_paired"] for w in WINDOWS]
    axes[1].bar(xw, d_vals, color="#C0504D")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_xticks(xw)
    axes[1].set_xticklabels([f"最初{w}ep" for w in WINDOWS])
    axes[1].set_ylabel("対応のあるCohen's d(Arm2-Arm3)")
    axes[1].set_title("効果量")

    fig.suptitle(f"要件6 新参者効果の統計的検証(n={n}系統)")
    fig.tight_layout()
    fig.savefig("community_v2_n15_stats.png", dpi=150)
    print("グラフを community_v2_n15_stats.png に保存しました。")


if __name__ == "__main__":
    main()
