"""C0(対照条件: 学習なし・推論のみ)のランナー。事前登録4章・7章と追記欄「2026-09-12」に従う。

  - 標本: 15 seed × 30 episode/seed(既定値。--seeds / --episodes で変更可)
  - 方策: TorchPolicy(Qwen2.5-1.5B-Instruct, fp16, 温度1.0)。環境定数は
    emotion_grounding_env.py の現在値をそのまま使う(このスクリプトでは何も上書きしない)。
  - seed の意味: GroundingEnv(seed) の課題列と、方策のサンプリング乱数
    (torch.manual_seed(seed))の両方を決める。同じ seed なら同じ課題列。
  - 保存: 1 seed 終わるごとに、その seed の全 StepRecord を JSON で out-dir に書く
    (seed_records.write_seed_file。一時ファイル→置き換えなので途中で切れても壊れない)。
  - 再開: 起動時に out-dir を見て、完了済み(complete かつ n_episodes・環境定数が一致)の
    seed は飛ばす。Colab が切れたら同じコマンドをもう一度実行すればよい。
    --force で完了済みも実行し直す(上書き)。
  - 進捗: episode ごとに 1 行(seed, episode, ステップ数, 正答数, 秒数, seed 内の経過, 全体の残り見込み)。

使い方(Colab):
  python3 run_c0.py --policy torch --seeds 15 --episodes 30 --out-dir /content/drive/MyDrive/EmotionalAI/c0_v2
  (v1 の結果は Drive の c0/ とリポジトリの data/c0_v1/ に残す。試走は --seeds 3 --episodes 10 --out-dir .../c0_v2_trial)
ローカルでの乾式実行(GPU 不要。DummyPolicy。本物の値ではない):
  python3 run_c0.py --policy dummy --seeds 3 --episodes 2 --out-dir /tmp/c0_dry
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_grounding_env import GroundingEnv, run_episode  # noqa: E402
import seed_records as S  # noqa: E402

CONDITION = "C0"
DEFAULT_SEEDS = 15
DEFAULT_EPISODES = 30


def make_policy(kind, seed, temperature, model=None, use_chat_template=True, stop_rules=False,
                max_tokens=None):
    if kind == "torch":
        import torch
        from torch_qwen_policy import TorchPolicy
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        pol = make_policy.cache.get("torch")
        if pol is None:                                  # モデルの読み込みは1回だけ
            kwargs = {"temperature": temperature, "use_chat_template": use_chat_template,
                      "stop_rules": stop_rules}
            if model:
                kwargs["model_path"] = model
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            pol = TorchPolicy(**kwargs)
            make_policy.cache["torch"] = pol
        info = {"policy": "torch", "model": pol.model_path, "temperature": pol.temperature,
                "max_tokens": pol.max_tokens, "torch_seed": seed,
                "use_chat_template": pol.use_chat_template, "stop_rules": pol.stop_rules}
        return pol, info
    from emotion_grounding_env import DummyPolicy
    pol = DummyPolicy(seed=seed)
    return pol, {"policy": "dummy", "model": None, "temperature": None, "seed": seed}


make_policy.cache = {}


def fmt_min(sec):
    return f"{sec / 60:.1f}min"


def run_seed(seed, n_episodes, policy, env_seed=None):
    env = GroundingEnv(seed=env_seed if env_seed is not None else seed)
    records = []
    t_seed = time.perf_counter()
    ep_secs = []
    for ep in range(n_episodes):
        t0 = time.perf_counter()
        recs = run_episode(env, policy, ep)
        dt = time.perf_counter() - t0
        ep_secs.append(dt)
        records.extend(recs)
        n_correct = sum(r.correct for r in recs)
        print(f"  seed {seed:02d} ep {ep + 1:02d}/{n_episodes}: steps={len(recs):2d} "
              f"correct={n_correct:2d}/{len(recs):<2d} {dt:5.1f}s | seed経過 {fmt_min(time.perf_counter() - t_seed)}",
              flush=True)
    return records, ep_secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["torch", "dummy"], default="torch")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="seed 数(seed-start から連番)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true", help="完了済み seed も実行し直す")
    # 素のモデル(非Instruct)の試走用(追記欄「2026-09-13(4)」)。既定は Instruct 版と同じ挙動。
    ap.add_argument("--model", default=None, help="HF のモデル名(既定: torch_qwen_policy.MODEL_PATH)")
    ap.add_argument("--no-chat-template", action="store_true", help="チャットテンプレートを使わず生の文字列で与える")
    ap.add_argument("--stop-rules", action="store_true", help="stop_rules.py の停止規則で生成を打ち切る")
    ap.add_argument("--max-tokens", type=int, default=None, help="生成の上限トークン数(既定: 方策の既定値 200)")
    args = ap.parse_args()

    if args.policy == "dummy":
        print("[注意] --policy dummy は配管の乾式実行用。本物の応答・エントロピー値ではない。", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    todo, skipped = [], []
    for s in seeds:
        path = os.path.join(args.out_dir, S.seed_file_name(CONDITION, s))
        ok, why = S.is_complete_seed_file(path, n_episodes=args.episodes, condition=CONDITION)
        if ok and not args.force:
            skipped.append(s)
        else:
            todo.append((s, path, why))
    print(f"=== {CONDITION}: {len(seeds)} seeds × {args.episodes} episodes, temperature={args.temperature}, "
          f"out-dir={args.out_dir} ===", flush=True)
    if args.model or args.no_chat_template or args.stop_rules or args.max_tokens:
        print(f"方策オプション: model={args.model or '(既定)'} chat_template={not args.no_chat_template} "
              f"stop_rules={args.stop_rules} max_tokens={args.max_tokens or '(既定)'}", flush=True)
    print(f"環境定数: {S.env_constants()}", flush=True)
    if skipped:
        print(f"完了済みのため飛ばす seed: {skipped}", flush=True)
    for s, path, why in todo:
        if why not in ("not found",):
            print(f"seed {s:02d}: 既存ファイルを実行し直す({why})", flush=True)
    if not todo:
        print("実行する seed はない(すべて完了済み)。", flush=True)
        return 0

    t_all = time.perf_counter()
    seed_secs = []
    for i, (s, path, _) in enumerate(todo):
        policy, info = make_policy(args.policy, s, args.temperature, model=args.model,
                                   use_chat_template=not args.no_chat_template, stop_rules=args.stop_rules,
                                   max_tokens=args.max_tokens)
        print(f"--- seed {s:02d} 開始({i + 1}/{len(todo)}) ---", flush=True)
        if hasattr(policy, "stop_log"):
            policy.stop_log = []                         # この seed の停止理由だけを集める
        t0 = time.perf_counter()
        records, ep_secs = run_seed(s, args.episodes, policy)
        elapsed = time.perf_counter() - t0
        seed_secs.append(elapsed)
        extra = {"elapsed_seconds": elapsed, "episode_seconds": ep_secs,
                 "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if hasattr(policy, "stop_log"):
            # records と同じ順序・同じ長さ(応答ごとの停止理由: eos / max_tokens / fabrication)
            extra["stop_reasons"] = list(policy.stop_log)
            extra["stop_reason_counts"] = {k: policy.stop_log.count(k) for k in sorted(set(policy.stop_log))}
        S.write_seed_file(path, condition=CONDITION, seed=s, n_episodes=args.episodes,
                          records=records, policy_info=info, extra=extra)
        n_correct = sum(r.correct for r in records)
        usage = sum(r.has_emotion for r in records) / max(1, len(records))
        remaining = (len(todo) - i - 1) * (sum(seed_secs) / len(seed_secs))
        print(f"--- seed {s:02d} 完了: steps={len(records)} correct={n_correct}/{len(records)} "
              f"感情語使用率={usage:.3f} {fmt_min(elapsed)} → 保存 {os.path.basename(path)} | "
              f"残り {len(todo) - i - 1} seed、見込み {fmt_min(remaining)} ---", flush=True)

    print(f"=== 完了: {len(todo)} seed を実行({fmt_min(time.perf_counter() - t_all)})。"
          f"保存先 {args.out_dir} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
