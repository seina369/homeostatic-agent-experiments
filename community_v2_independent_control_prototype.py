"""
感情AIプロジェクト フェーズ6 プロトタイプ: 要件6 世代交代の伝達効果そのものの有無を検証(対照実験B)

反復学習v2(n=15)の結果は「体系化促進効果」を統計的に支持しなかった。本
プロトタイプは、そもそも「世代交代」という操作自体が収束後の結果に何らかの
因果的影響を持つのかを直接検証する対照実験(B)。

community_signal_v2_prototype.pyの標準設定(4×4グリッド・衝突ペナルティ8.0・
推測ゲーム・3500ep)で、ボトルネック初期化を一切行わない独立試行を15系統
(新規シード201〜215)実行する。これは「世代交代という操作を全く挟まない、
毎回ゼロからの単発学習」に相当する。

得られたMI分布(n=15)を、既存のv2データの「世代5」MI分布(n=15、
community_signal_iterated_v2_prototype.pyで既に得られている)とマン・ホイット
ニーのU検定で比較する。有意差がなければ、5世代分のボトルネック連鎖を経た
結果は「そもそも世代交代をしていない単発学習」と統計的に見分けがつかない
ことになる。

45秒のbash呼び出し制限に対応するため、community_signal_iterated_v2_
prototype.pyのgen_multi_chunkと同じ「時間主導」のチャンク実行方式を採用する
(community_signal_v2_prototype.pyのrun_train_chunk/run_train_finalizeを
そのまま呼び出す薄いラッパー)。

使い方:
  python3 community_v2_independent_control_prototype.py multi_chunk <traj_seed>
  python3 community_v2_independent_control_prototype.py aggregate
"""

import sys, json, pickle, time
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats

import community_signal_v2_prototype as m

NEW_SEEDS = list(range(201, 216))  # n=15の新規独立試行(ボトルネック初期化なし)
EXISTING_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22]  # 反復学習v2の既存15系統


def multi_chunk(traj_seed, target_end_ep=m.N_EPISODES, time_budget=38.0, sub_step=300):
    t_start = time.time()
    state_file = f"community_v2_state_seed{traj_seed}.pkl"
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
            print(f"[seed={traj_seed}] 時間予算({time_budget}s)到達、{cur_last_ep}epで一旦終了")
            return
        next_ep = min(cur_last_ep + sub_step, target_end_ep)
        m.run_train_chunk(traj_seed, next_ep)
        n_sub_chunks += 1

    print(f"[seed={traj_seed}] target_end_ep={target_end_ep}に到達、{n_sub_chunks}個のサブチャンクで完了、train_finalizeを実行")
    m.run_train_finalize(traj_seed)


def aggregate():
    print("=== 要件6: 世代交代の伝達効果そのものの有無(対照実験B) ===")
    new_mis = []
    for s in NEW_SEEDS:
        d = json.load(open(f"community_v2_train_seed{s}.json"))
        new_mis.append(d["mi_by_checkpoint"]["3500"]["mi"])
    print(f"新規独立試行(単発学習、n={len(new_mis)})のMI: {['%.4f' % x for x in new_mis]}")
    print(f"平均±標準偏差: {np.mean(new_mis):.4f}±{np.std(new_mis):.4f}bit")

    gen5_mis = []
    for s in EXISTING_SEEDS:
        d = json.load(open(f"iterated_v2_results_seed{s}.json"))
        gen5_mis.append(d["5"]["mi_by_checkpoint"]["3500"]["mi"])
    print(f"\n既存の反復学習v2・世代5(n={len(gen5_mis)})のMI: {['%.4f' % x for x in gen5_mis]}")
    print(f"平均±標準偏差: {np.mean(gen5_mis):.4f}±{np.std(gen5_mis):.4f}bit")

    u_stat, p_u = stats.mannwhitneyu(new_mis, gen5_mis, alternative="two-sided")
    print(f"\nマン・ホイットニーのU検定: U={u_stat:.2f}, p={p_u:.4f}")

    # 参考: Welchのt検定も併記
    t_stat, p_t = stats.ttest_ind(new_mis, gen5_mis, equal_var=False)
    print(f"(参考)Welchのt検定: t={t_stat:.3f}, p={p_t:.4f}")

    with open("independent_control_summary.json", "w") as f:
        json.dump({
            "new_seed_mis": new_mis, "new_mean": float(np.mean(new_mis)), "new_std": float(np.std(new_mis)),
            "gen5_mis": gen5_mis, "gen5_mean": float(np.mean(gen5_mis)), "gen5_std": float(np.std(gen5_mis)),
            "mannwhitney_u": float(u_stat), "mannwhitney_p": float(p_u),
            "welch_t": float(t_stat), "welch_p": float(p_t),
        }, f, ensure_ascii=False, indent=2)
    print("saved independent_control_summary.json")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "multi_chunk":
        multi_chunk(int(sys.argv[2]))
    elif cmd == "aggregate":
        aggregate()
