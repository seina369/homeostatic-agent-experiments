"""
速度・書式・エントロピー実測プローブ(事前登録手順3)
================================================================

emotion_grounding_env.GroundingEnv を、MLXPolicy(Qwen2.5-0.5B-Instruct)
またはDummyPolicy(検算用)に接続してN episode走らせ、
  - エピソードごとの秒数
  - 正答率
  - 書式不履行率(正規表現 "A: <answer>" が見つからない割合。誤答とは別に集計)
  - エントロピー(uncertainty信号そのもの)の分布
を記録する。学習は行わない(policy.respond()の呼び出しのみ、
optimizer/勾配更新は一切ない)。

使い方:
  python3 run_mlx_speed_probe.py --policy mlx --episodes 20
  python3 run_mlx_speed_probe.py --policy dummy --episodes 20   (MLXなしでの動作確認用)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# emotion_grounding_env.py / mlx_qwen_policy.py と同じフォルダに置かれている
# 前提で、スクリプト自身の場所を基準にimportする(実行環境依存のパスを
# ハードコードしない)。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_grounding_env import GroundingEnv, run_episode, extract_answer  # noqa: E402


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["mlx", "dummy"], default="mlx")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="mlx_speed_probe_results.json")
    args = parser.parse_args()

    if args.policy == "mlx":
        from mlx_qwen_policy import MLXPolicy
        policy = MLXPolicy()
    else:
        from emotion_grounding_env import DummyPolicy
        policy = DummyPolicy(seed=args.seed)
        print("[注意] --policy dummy はハーネス自体の動作確認用。"
              "本物の速度・エントロピー値ではない。")

    env = GroundingEnv(seed=args.seed)

    episode_seconds = []
    all_records = []
    for ep in range(args.episodes):
        t0 = time.perf_counter()
        records = run_episode(env, policy, ep)
        dt = time.perf_counter() - t0
        episode_seconds.append(dt)
        all_records.extend(records)
        n_correct = sum(r.correct for r in records)
        print(f"episode {ep}: {dt:.2f}s, steps={len(records)}, correct={n_correct}/{len(records)}")

    n = len(all_records)
    correct_rate = sum(r.correct for r in all_records) / n if n else 0.0
    format_fail_rate = sum(extract_answer(r.text) is None for r in all_records) / n if n else 0.0
    entropies = [r.unc_after for r in all_records]  # = resp.mean_entropy(そのステップの報告値)
    tokens = [r.n_tokens for r in all_records]

    result = {
        "policy": args.policy,
        "n_episodes": args.episodes,
        "n_steps_total": n,
        "episode_seconds": episode_seconds,
        "episode_seconds_mean": float(np.mean(episode_seconds)) if episode_seconds else 0.0,
        "episode_seconds_std": float(np.std(episode_seconds)) if episode_seconds else 0.0,
        "episode_seconds_min": float(np.min(episode_seconds)) if episode_seconds else 0.0,
        "episode_seconds_max": float(np.max(episode_seconds)) if episode_seconds else 0.0,
        "correct_rate": correct_rate,
        "format_fail_rate": format_fail_rate,
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "entropy_std": float(np.std(entropies)) if entropies else 0.0,
        "entropy_min": float(np.min(entropies)) if entropies else 0.0,
        "entropy_max": float(np.max(entropies)) if entropies else 0.0,
        "entropy_p10": percentile(entropies, 10),
        "entropy_p50": percentile(entropies, 50),
        "entropy_p90": percentile(entropies, 90),
        "entropy_values": entropies,
        "mean_tokens_per_response": float(np.mean(tokens)) if tokens else 0.0,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print()
    print("=== 集計 ===")
    print(f"エピソード秒数: 平均{result['episode_seconds_mean']:.2f}s ± {result['episode_seconds_std']:.2f}s "
          f"(最小{result['episode_seconds_min']:.2f}s, 最大{result['episode_seconds_max']:.2f}s)")
    print(f"正答率: {correct_rate:.3f}")
    print(f"書式不履行率: {format_fail_rate:.3f}")
    print(f"エントロピー: 平均{result['entropy_mean']:.3f} ± {result['entropy_std']:.3f} "
          f"(p10={result['entropy_p10']:.3f}, p50={result['entropy_p50']:.3f}, p90={result['entropy_p90']:.3f}, "
          f"最小{result['entropy_min']:.3f}, 最大{result['entropy_max']:.3f})")
    print(f"平均トークン数/応答: {result['mean_tokens_per_response']:.1f}")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
