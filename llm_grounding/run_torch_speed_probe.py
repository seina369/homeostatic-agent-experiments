"""
速度・書式・エントロピー実測プローブ(PyTorch/Colab版)
================================================================

run_mlx_speed_probe.py のColab版。emotion_grounding_env.GroundingEnv を、
TorchPolicy(Qwen2.5-1.5B-Instruct, fp16)またはDummyPolicy(検算用)に
接続してN episode走らせ、
  - エピソードごとの秒数
  - 正答率(全体、および課題の種類(kind)ごと。2026-09-12の実測時点では
    add/sub/mul/reverse/countの5種、以後はreverseを除いた4種)
  - 書式不履行率("A: <answer>" が見つからない割合。誤答とは別に集計)
  - エントロピー(uncertainty信号そのもの)の分布
  - 不正解になった応答の実例(最大10件。遭遇順、選別なし)
を記録する。学習は行わない(policy.respond()の呼び出しのみ)。

emotion_grounding_env.py は変更しない。run_episode()と全く同じ手順を
run_episode_with_tasks()としてこのファイル内に複製しているだけで(下記参照)、
環境側のロジックには一切手を入れていない。

使い方(Colab):
  python3 run_torch_speed_probe.py --policy torch --episodes 20 --out result.json
ローカルでのハーネス動作確認用(GPU不要):
  python3 run_torch_speed_probe.py --policy dummy --episodes 20
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# emotion_grounding_env.py / torch_qwen_policy.py と同じフォルダに置かれている
# 前提で、スクリプト自身の場所を基準にimportする(実行環境依存のパスを
# ハードコードしない)。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_grounding_env import (  # noqa: E402
    GroundingEnv, make_task, extract_answer,
)


def run_episode_with_tasks(env: GroundingEnv, policy, episode: int):
    """emotion_grounding_env.run_episode() と全く同じ手順の複製。

    唯一の違いは、各ステップで使ったTask本体も一緒に返すこと(不正解の
    実例をあとで人間が読める形で残すため。StepRecordにはtask_kindしか
    残らず、課題文と正解そのものは残らないため)。環境ファイル自体は
    変更しない、という方針に従い、ロジックはここで複製するだけに留める。
    """
    env.reset()
    policy.reset()
    pairs = []
    while not env.done():
        task = make_task(env.task_rng)
        prompt = env.build_prompt(task)
        resp = policy.respond(prompt)
        rec = env.step(task, resp, episode)
        pairs.append((task, rec))
    return pairs


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["torch", "dummy"], default="torch")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="torch_speed_probe_results.json")
    parser.add_argument("--max-examples", type=int, default=10,
                         help="記録する不正解の実例の最大件数")
    args = parser.parse_args()

    if args.policy == "torch":
        from torch_qwen_policy import TorchPolicy
        policy = TorchPolicy()
    else:
        from emotion_grounding_env import DummyPolicy
        policy = DummyPolicy(seed=args.seed)
        print("[注意] --policy dummy はハーネス自体(集計ロジック)の動作確認用。"
              "本物の速度・エントロピー値ではない。")

    env = GroundingEnv(seed=args.seed)

    episode_seconds = []
    all_pairs = []  # (task, StepRecord) を全ステップぶん集める
    for ep in range(args.episodes):
        t0 = time.perf_counter()
        pairs = run_episode_with_tasks(env, policy, ep)
        dt = time.perf_counter() - t0
        episode_seconds.append(dt)
        all_pairs.extend(pairs)
        records = [r for _, r in pairs]
        n_correct = sum(r.correct for r in records)
        print(f"episode {ep}: {dt:.2f}s, steps={len(records)}, correct={n_correct}/{len(records)}")

    all_records = [r for _, r in all_pairs]
    n = len(all_records)
    correct_rate = sum(r.correct for r in all_records) / n if n else 0.0
    format_fail_rate = sum(extract_answer(r.text) is None for r in all_records) / n if n else 0.0
    entropies = [r.unc_after for r in all_records]  # = resp.mean_entropy(そのステップの報告値)
    tokens = [r.n_tokens for r in all_records]

    # 課題の種類(kind)ごとの正答率
    kind_stats = {}
    for task, r in all_pairs:
        s = kind_stats.setdefault(task.kind, {"n": 0, "correct": 0})
        s["n"] += 1
        s["correct"] += int(r.correct)
    for s in kind_stats.values():
        s["accuracy"] = s["correct"] / s["n"] if s["n"] else 0.0

    # 不正解の実例(最大--max-examples件。遭遇した順のまま、選別はしない)
    incorrect_examples = []
    for task, r in all_pairs:
        if r.correct or len(incorrect_examples) >= args.max_examples:
            continue
        incorrect_examples.append({
            "episode": r.episode,
            "t": r.t,
            "kind": task.kind,
            "task_prompt": task.prompt,
            "correct_answer": task.answer,
            "model_response": r.text,
            "extracted_answer": extract_answer(r.text),
            "n_tokens": r.n_tokens,
            "entropy": r.unc_after,
        })

    result = {
        "policy": args.policy,
        "model": getattr(policy, "model_path", None) or "N/A(dummy)",
        "n_episodes": args.episodes,
        "n_steps_total": n,
        "episode_seconds": episode_seconds,
        "episode_seconds_mean": float(np.mean(episode_seconds)) if episode_seconds else 0.0,
        "episode_seconds_std": float(np.std(episode_seconds)) if episode_seconds else 0.0,
        "episode_seconds_min": float(np.min(episode_seconds)) if episode_seconds else 0.0,
        "episode_seconds_max": float(np.max(episode_seconds)) if episode_seconds else 0.0,
        "correct_rate": correct_rate,
        "format_fail_rate": format_fail_rate,
        "accuracy_by_kind": kind_stats,
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "entropy_std": float(np.std(entropies)) if entropies else 0.0,
        "entropy_min": float(np.min(entropies)) if entropies else 0.0,
        "entropy_max": float(np.max(entropies)) if entropies else 0.0,
        "entropy_p10": percentile(entropies, 10),
        "entropy_p50": percentile(entropies, 50),
        "entropy_p90": percentile(entropies, 90),
        "entropy_values": entropies,
        "mean_tokens_per_response": float(np.mean(tokens)) if tokens else 0.0,
        "incorrect_examples": incorrect_examples,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print()
    print("=== 集計 ===")
    print(f"エピソード秒数: 平均{result['episode_seconds_mean']:.2f}s ± {result['episode_seconds_std']:.2f}s "
          f"(最小{result['episode_seconds_min']:.2f}s, 最大{result['episode_seconds_max']:.2f}s)")
    print(f"正答率: {correct_rate:.3f}")
    for kind, s in sorted(kind_stats.items()):
        print(f"  - {kind}: {s['accuracy']:.3f} ({s['correct']}/{s['n']})")
    print(f"書式不履行率: {format_fail_rate:.3f}")
    print(f"エントロピー: 平均{result['entropy_mean']:.3f} ± {result['entropy_std']:.3f} "
          f"(p10={result['entropy_p10']:.3f}, p50={result['entropy_p50']:.3f}, p90={result['entropy_p90']:.3f}, "
          f"最小{result['entropy_min']:.3f}, 最大{result['entropy_max']:.3f})")
    print(f"平均トークン数/応答: {result['mean_tokens_per_response']:.1f}")
    print(f"不正解の実例: {len(incorrect_examples)}件を記録")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
