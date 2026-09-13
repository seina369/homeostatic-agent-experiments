"""関門 G1: GRPO を現行の系に繋ぐ(事前登録(第一段階)6章 G1)。

系: v2 プロンプト + `status:` 行(状態を文字で見せる形)、注入なし。モデル + LoRA。
報酬 = −逸脱(その応答を環境が受けたときの budget / error / uncertainty の逸脱の和。
環境の compute_deviation。文の内容・語彙は一切見ない)。

seed ごとの手順:
  1. 学習前の評価: 評価用の環境 GroundingEnv(seed=EVAL_SEED_BASE + seed) の課題列で
     eval_episodes 本のエピソードを 1 応答ずつ生成(温度 1.0)し、平均逸脱・正答率・
     書式不履行率・平均トークン数を記録する。
  2. 学習: 学習用の環境 GroundingEnv(seed=TRAIN_SEED_BASE + seed) でエピソードを進めながら、
     各ステップで同じプロンプトから G 本サンプルし、各本の報酬(そのステップでその応答を
     受けたときの −逸脱)でグループを作る。先頭の 1 本で環境を進める(on-policy の軌跡)。
     groups_per_update 個のグループごとに 1 回更新(GRPOTrainer.step)。更新回数 updates。
  3. 学習後の評価: 1 と同じ評価用の seed(同じ課題列)で同じ手順。
  4. JSON(g1_seedNN.json): 設定、前後の評価、更新ごとの平均報酬・KL・損失・トークン数・
     経過秒、所要時間(読み込み・評価・学習ごと)、NaN/inf の有無、判定。
  5. 3 seed が終わったら g1_summary.json に seed ごとの判定と全体の合否。

合格判定(6章): seed ごとに (a) 相対低下 (before − after) / before ≥ 0.05、(b) 発散なし
(損失・報酬・KL に NaN/inf がない)、(c) 学習+評価の所要時間 ≤ 90 分。3 seed すべて合格で G1 通過。

既定値の根拠(G1 で固定する値の初期候補。G1 の結果を見て追記欄で確定する):
  - G=8: T4(16GB)で 1.5B fp16 の G 本一括採点(勾配つき)が収まる範囲。GRPO 原論文の 64 より
    小さいが、報酬は決定的(逸脱の式)なので優位の推定は安定する。--micro-batch で採点を分割できる。
  - lr=1e-5: LoRA + AdamW で崩壊(文の定型化、書式不履行)を避ける保守的な値。CPU の玩具課題で
    動くことを確認した 5e-2 とは桁が違うが、本番モデルでは小さい方から始め、G1 で効かなければ上げる。
  - beta_kl=0.04: DeepSeekMath の GRPO の既定値。参照方策は「同じモデルで LoRA を無効化したもの」
    (G1 では注入がないので、3章の「注入あり・LoRA なし」と同じ)。
  - updates=40、groups_per_update=4: 160 グループ = 160 ステップ ≈ 16 エピソード分の学習。
    1 グループ ≈ サンプリング G 本 + 採点 3 回(old / ref / new+backward)で 10 秒前後の見込み
    → 学習 ≈ 30 分。評価 2 回 × 10 エピソード ≈ 4 分。読み込み ≈ 2 分。合計で 90 分に収まる見込み。
  - eval_episodes=10: 1 seed 100 応答。前後の差 5% を見る最低限。
  - max_new_tokens=200、temperature=1.0: C0 と同じ。LoRA r=8 / alpha=16(注意層 + MLP)。
  - EVAL_SEED_BASE=2000、TRAIN_SEED_BASE=1000: C0 の seed(0..14)と重ならない。

実行(Colab / Kaggle、GPU):
  python3 run_g1.py --seeds 3 --out-dir <dir>
  python3 run_g1.py --judge-only --out-dir <dir>     # 既存の JSON から判定だけ出す
CPU の極小モデルでの検算は test_run_g1.py。
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emotion_grounding_env as E  # noqa: E402
from emotion_grounding_env import GroundingEnv, Response, make_task, is_correct  # noqa: E402
from seed_records import env_constants  # noqa: E402

EVAL_SEED_BASE = 2000
TRAIN_SEED_BASE = 1000
MIN_REL_DROP = 0.05
TIME_LIMIT_MIN = 90.0


@dataclass
class G1Config:
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    G: int = 8
    lr: float = 1e-5
    beta_kl: float = 0.04
    updates: int = 40
    groups_per_update: int = 4
    eval_episodes: int = 10
    max_new_tokens: int = 200
    temperature: float = 1.0
    lora_r: int = 8
    lora_alpha: int = 16
    micro_batch: int = 0
    time_limit_min: float = TIME_LIMIT_MIN
    min_rel_drop: float = MIN_REL_DROP


# ------------------------------------------------------------
# 判定(torch 不要)
# ------------------------------------------------------------
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def judge_seed(result: dict, min_rel_drop: float = MIN_REL_DROP, time_limit_min: float = TIME_LIMIT_MIN) -> dict:
    """1 seed の結果 dict から合否を出す。result は run_seed の戻り値(または JSON を読んだもの)。"""
    before = result["eval_before"]["mean_deviation"]
    after = result["eval_after"]["mean_deviation"]
    rel_drop = (before - after) / before if _finite(before) and before > 0 and _finite(after) else float("nan")
    drop_ok = _finite(rel_drop) and rel_drop >= min_rel_drop
    nan_found = bool(result.get("nan_found", False)) or not (_finite(before) and _finite(after))
    elapsed_min = result["elapsed_seconds"]["total"] / 60.0
    time_ok = elapsed_min <= time_limit_min
    reasons = []
    if not drop_ok:
        reasons.append(f"相対低下 {rel_drop:.3f} < {min_rel_drop}" if _finite(rel_drop) else "相対低下が計算できない")
    if nan_found:
        reasons.append("NaN/inf あり(発散)")
    if not time_ok:
        reasons.append(f"所要 {elapsed_min:.1f} 分 > {time_limit_min:.0f} 分")
    return {"seed": result.get("seed"), "pass": drop_ok and not nan_found and time_ok,
            "rel_drop": rel_drop, "nan_found": nan_found, "elapsed_min": elapsed_min,
            "before": before, "after": after, "reasons": reasons}


def judge_all(results, min_rel_drop: float = MIN_REL_DROP, time_limit_min: float = TIME_LIMIT_MIN,
              required_seeds: int = 3) -> dict:
    per_seed = [judge_seed(r, min_rel_drop, time_limit_min) for r in results]
    n_pass = sum(1 for j in per_seed if j["pass"])
    overall = len(per_seed) >= required_seeds and n_pass == len(per_seed)
    return {"per_seed": per_seed, "n_seeds": len(per_seed), "n_pass": n_pass,
            "required_seeds": required_seeds, "pass": overall,
            "criteria": {"min_rel_drop": min_rel_drop, "time_limit_min": time_limit_min,
                         "no_nan": True, "all_seeds_must_pass": True}}


def format_judgement(j: dict) -> str:
    lines = [f"G1 判定: {'合格' if j['pass'] else '不合格'}({j['n_pass']}/{j['n_seeds']} seed 合格、必要 {j['required_seeds']} seed)"]
    for s in j["per_seed"]:
        rd = f"{s['rel_drop']:+.3f}" if _finite(s["rel_drop"]) else "nan"
        lines.append(f"  seed {s['seed']}: {'合格' if s['pass'] else '不合格'} 逸脱 {s['before']:.3f} → {s['after']:.3f} "
                     f"(相対低下 {rd}) NaN={s['nan_found']} 所要 {s['elapsed_min']:.1f} 分 {'; '.join(s['reasons'])}")
    return "\n".join(lines)


# ------------------------------------------------------------
# 生成と環境(torch が要る部分)
# ------------------------------------------------------------
def prompt_to_ids(tokenizer, prompt: str):
    """TorchPolicy と同じ: user メッセージ 1 つのチャットテンプレート(system なし)。"""
    formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                              add_generation_prompt=True, tokenize=False)
    return list(tokenizer(formatted)["input_ids"])


def hypothetical_outcome(env: GroundingEnv, task, resp: Response):
    """環境を進めずに、その応答を受けたときの逸脱・報酬を GroundingEnv.step と同じ式で出す。"""
    correct = is_correct(resp.text, task)
    budget_after = max(env.budget - resp.n_tokens, 0.0)
    error_after = env.error + (0.0 if correct else 1.0)
    dev, reward = E.compute_deviation(budget_after, error_after, float(resp.mean_entropy))
    return reward, dev, correct


def responses_from_ids(trainer, tokenizer, prompt_ids, resp_ids_list):
    """トークン id の応答列 → Response(text, n_tokens, mean_entropy)。エントロピーは生ロジットから(TorchPolicy と同じ定義)。"""
    out = []
    for idx in trainer._chunks(len(resp_ids_list)):          # micro_batch ごとに採点(メモリ)
        sub = [resp_ids_list[i] for i in idx]
        scored = trainer.score(prompt_ids, sub, grad=False)
        for ids, (_, ent) in zip(sub, scored):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            mean_ent = float(ent.mean()) if ent.numel() else 0.0
            out.append(Response(text=text, n_tokens=len(ids), mean_entropy=mean_ent))
    return out


def evaluate(trainer, tokenizer, seed: int, n_episodes: int) -> dict:
    """評価用の環境(seed 固定 → 同じ課題列)で 1 応答ずつ生成し、平均逸脱などを返す。"""
    env = GroundingEnv(seed=seed)
    devs, corrects, fails, toks, per_episode = [], [], [], [], []
    for ep in range(n_episodes):
        env.reset()
        ep_devs = []
        while not env.done():
            task = make_task(env.task_rng)
            pids = prompt_to_ids(tokenizer, env.build_prompt(task))
            ids = trainer.sample_group(pids, 1)[0]
            resp = responses_from_ids(trainer, tokenizer, pids, [ids])[0]
            rec = env.step(task, resp, ep)
            devs.append(rec.deviation); ep_devs.append(rec.deviation)
            corrects.append(rec.correct); fails.append(E.extract_answer(resp.text) is None); toks.append(resp.n_tokens)
        per_episode.append(sum(ep_devs) / len(ep_devs) if ep_devs else float("nan"))
    n = len(devs)
    return {"seed": seed, "n_episodes": n_episodes, "n_steps": n,
            "mean_deviation": sum(devs) / n if n else float("nan"),
            "correct_rate": sum(corrects) / n if n else float("nan"),
            "format_fail_rate": sum(fails) / n if n else float("nan"),
            "mean_tokens": sum(toks) / n if n else float("nan"),
            "per_episode_mean_deviation": per_episode}


def _gpu_peak_reset():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _gpu_peak_gb():
    """直前の _gpu_peak_reset() 以降の GPU メモリ最大値(GB)。GPU がなければ None。"""
    import torch
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def train(trainer, tokenizer, cfg: G1Config, seed: int, log_fn=None) -> dict:
    """学習用の環境でグループを集めて更新する。戻り値: 更新ごとの記録と NaN の有無。"""
    env = GroundingEnv(seed=seed)
    env.reset()
    episode = 0
    log = []
    nan_found = False
    t0 = time.time()
    for u in range(cfg.updates):
        t_u = time.time()
        _gpu_peak_reset()
        groups, rewards_all = [], []
        for _ in range(cfg.groups_per_update):
            if env.done():
                env.reset()
                episode += 1
            task = make_task(env.task_rng)
            pids = prompt_to_ids(tokenizer, env.build_prompt(task))
            resp_ids = trainer.sample_group(pids, cfg.G)
            resps = responses_from_ids(trainer, tokenizer, pids, resp_ids)
            rewards = [hypothetical_outcome(env, task, r)[0] for r in resps]
            groups.append({"prompt_ids": pids, "responses": resp_ids, "rewards": rewards})
            rewards_all.extend(rewards)
            env.step(task, resps[0], episode)          # 先頭の 1 本で環境を進める
        stats = trainer.step(groups)
        entry = {"update": u, "mean_reward": sum(rewards_all) / len(rewards_all),
                 "min_reward": min(rewards_all), "max_reward": max(rewards_all),
                 "loss": stats["loss"], "pg": stats["pg"], "kl": stats["kl"], "n_tokens": stats["n_tokens"],
                 "episode": episode, "elapsed_seconds": time.time() - t0,
                 "update_seconds": time.time() - t_u, "peak_gpu_gb": _gpu_peak_gb()}
        if not all(_finite(entry[k]) for k in ("mean_reward", "loss", "pg", "kl")):
            nan_found = True
        log.append(entry)
        if log_fn:
            pk = f"{entry['peak_gpu_gb']:.2f} GB" if entry["peak_gpu_gb"] is not None else "-"
            log_fn(f"  update {u + 1}/{cfg.updates}: reward {entry['mean_reward']:.3f} kl {entry['kl']:.4f} "
                   f"loss {entry['loss']:.4f} tokens {entry['n_tokens']} | この更新 {entry['update_seconds']:.0f} 秒 "
                   f"({entry['update_seconds'] / cfg.groups_per_update:.0f} 秒/グループ) GPU最大 {pk} | 累計 {entry['elapsed_seconds'] / 60:.1f} 分")
    return {"log": log, "nan_found": nan_found, "episodes_used": episode + 1, "seconds": time.time() - t0}


def run_seed(model, tokenizer, cfg: G1Config, seed: int, out_dir: str, device=None, log_fn=print) -> dict:
    """1 seed 分: LoRA を付け、評価 → 学習 → 評価し、JSON を書いて結果 dict を返す。
    model は読み込み済みの因果言語モデル(seed ごとに新しく読み込んで渡す。LoRA は seed ごとに新規)。"""
    import torch
    from grpo_trainer import GRPOConfig, GRPOTrainer, attach_lora
    times = {}
    t_all = time.time()
    gcfg = GRPOConfig(group_size=cfg.G, lr=cfg.lr, beta_kl=cfg.beta_kl, temperature=cfg.temperature,
                      max_new_tokens=cfg.max_new_tokens, lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      micro_batch=cfg.micro_batch)
    lmodel = attach_lora(model, gcfg)
    trainer = GRPOTrainer(lmodel, tokenizer, gcfg, device=device)
    n_trainable = sum(p.numel() for p in lmodel.parameters() if p.requires_grad)

    devices = sorted({str(p.device) for p in lmodel.parameters()})
    log_fn(f"seed {seed}: モデルのデバイス {devices}(LoRA 込み。学習対象 {n_trainable:,} パラメータ)")
    if len(devices) != 1:
        raise RuntimeError(f"モデルが複数デバイスに分かれている: {devices}(cuda:0 の 1 枚に固定する)")
    pk = lambda: (f"{_gpu_peak_gb():.2f} GB" if _gpu_peak_gb() is not None else "-")

    torch.manual_seed(EVAL_SEED_BASE + seed)
    t = time.time()
    _gpu_peak_reset()
    eval_before = evaluate(trainer, tokenizer, EVAL_SEED_BASE + seed, cfg.eval_episodes)
    times["eval_before"] = time.time() - t
    eval_before["peak_gpu_gb"] = _gpu_peak_gb()
    log_fn(f"seed {seed}: 学習前 平均逸脱 {eval_before['mean_deviation']:.3f} 正答率 {eval_before['correct_rate']:.2f} "
           f"| {times['eval_before']:.0f} 秒 GPU最大 {pk()}")

    torch.manual_seed(TRAIN_SEED_BASE + seed)
    tr = train(trainer, tokenizer, cfg, TRAIN_SEED_BASE + seed, log_fn)
    times["train"] = tr["seconds"]

    torch.manual_seed(EVAL_SEED_BASE + seed)
    t = time.time()
    _gpu_peak_reset()
    eval_after = evaluate(trainer, tokenizer, EVAL_SEED_BASE + seed, cfg.eval_episodes)
    times["eval_after"] = time.time() - t
    eval_after["peak_gpu_gb"] = _gpu_peak_gb()
    times["total"] = time.time() - t_all
    log_fn(f"seed {seed}: 学習後 平均逸脱 {eval_after['mean_deviation']:.3f} 正答率 {eval_after['correct_rate']:.2f} "
           f"| {times['eval_after']:.0f} 秒 GPU最大 {pk()} | 合計 {times['total'] / 60:.1f} 分")

    n_groups = cfg.updates * cfg.groups_per_update
    peaks = [x for x in [eval_before["peak_gpu_gb"], eval_after["peak_gpu_gb"]] + [e["peak_gpu_gb"] for e in tr["log"]]
             if x is not None]
    peak_gb = max(peaks) if peaks else None
    if peak_gb is not None:
        log_fn(f"seed {seed}: GPU メモリ最大(全段階) {peak_gb:.2f} GB、1 グループあたり {tr['seconds'] / max(1, n_groups):.1f} 秒")
    result = {"condition": "G1", "seed": seed, "config": asdict(cfg), "env_constants": env_constants(),
              "prompt_version": E.PROMPT_VERSION, "n_trainable_params": int(n_trainable),
              "eval_before": eval_before, "eval_after": eval_after, "training": tr["log"],
              "episodes_used_for_training": tr["episodes_used"], "nan_found": tr["nan_found"],
              "elapsed_seconds": times,
              "seconds_per_group": (tr["seconds"] / n_groups) if n_groups else None,
              "peak_gpu_memory_gb": peak_gb}
    result["judgement"] = judge_seed(result, cfg.min_rel_drop, cfg.time_limit_min)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"g1_seed{seed:02d}.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    log_fn(f"保存: {path}")
    return result


def load_results(out_dir: str):
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(out_dir, "g1_seed*.json"))):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def write_summary(out_dir: str, cfg: G1Config):
    results = load_results(out_dir)
    j = judge_all(results, cfg.min_rel_drop, cfg.time_limit_min)
    with open(os.path.join(out_dir, "g1_summary.json"), "w", encoding="utf-8") as f:
        json.dump(j, f, ensure_ascii=False, indent=1)
    print(format_judgement(j))
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--model", default=G1Config.model)
    ap.add_argument("--G", type=int, default=G1Config.G)
    ap.add_argument("--lr", type=float, default=G1Config.lr)
    ap.add_argument("--beta-kl", type=float, default=G1Config.beta_kl)
    ap.add_argument("--updates", type=int, default=G1Config.updates)
    ap.add_argument("--groups-per-update", type=int, default=G1Config.groups_per_update)
    ap.add_argument("--eval-episodes", type=int, default=G1Config.eval_episodes)
    ap.add_argument("--max-new-tokens", type=int, default=G1Config.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=G1Config.temperature)
    ap.add_argument("--lora-r", type=int, default=G1Config.lora_r)
    ap.add_argument("--lora-alpha", type=int, default=G1Config.lora_alpha)
    ap.add_argument("--micro-batch", type=int, default=G1Config.micro_batch)
    ap.add_argument("--time-limit-min", type=float, default=G1Config.time_limit_min)
    ap.add_argument("--judge-only", action="store_true", help="既存の g1_seedNN.json から判定だけ出す")
    ap.add_argument("--force", action="store_true", help="完了済み seed も実行し直す")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):          # パイプ経由(Kaggle/Colab のログ)でも print を逐次流す
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)
    cfg = G1Config(model=args.model, G=args.G, lr=args.lr, beta_kl=args.beta_kl, updates=args.updates,
                   groups_per_update=args.groups_per_update, eval_episodes=args.eval_episodes,
                   max_new_tokens=args.max_new_tokens, temperature=args.temperature, lora_r=args.lora_r,
                   lora_alpha=args.lora_alpha, micro_batch=args.micro_batch, time_limit_min=args.time_limit_min)
    if args.judge_only:
        write_summary(args.out_dir, cfg)
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU が見つからない(G1 は GPU で実行する)")
    device = "cuda:0"        # 1.5B fp16 は 1 枚に収まる。device_map は使わず、モデル全体を cuda:0 に置く(T4×2 でも分散しない)
    print("G1 設定:", json.dumps(asdict(cfg), ensure_ascii=False))
    print(f"GPU 数 {torch.cuda.device_count()}、使用 {device}: {torch.cuda.get_device_name(0)} "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("環境定数:", env_constants(), "prompt_version:", E.PROMPT_VERSION)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        path = os.path.join(args.out_dir, f"g1_seed{seed:02d}.json")
        if os.path.exists(path) and not args.force:
            print(f"seed {seed}: 完了済み(スキップ)")
            continue
        t = time.time()
        model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float16).to(device)   # seed ごとに新規
        model.eval()
        devs = sorted({str(p.device) for p in model.parameters()})
        print(f"seed {seed}: モデル読み込み {time.time() - t:.0f} 秒、デバイス {devs}、"
              f"GPU 使用中 {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        if devs != [device]:
            raise RuntimeError(f"モデルが {device} 以外にある: {devs}")
        run_seed(model, tokenizer, cfg, seed, args.out_dir, device=device)
        del model
        torch.cuda.empty_cache()
    write_summary(args.out_dir, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
