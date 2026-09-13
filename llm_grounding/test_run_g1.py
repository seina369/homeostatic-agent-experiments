"""run_g1.py(関門 G1 のランナー)の動作確認。

  - 判定(judge_seed / judge_all)の分岐は torch 不要でテストする。
  - 極小の Qwen2(乱数初期化、語彙 64)と玩具トークナイザで、評価 → 2 回更新 → 評価が CPU で通り、
    JSON の形式が揃うことを確認する(torch / transformers / peft がない環境ではスキップ)。
実行: cd llm_grounding && python3 -m pytest -q test_run_g1.py
"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_g1 as R  # noqa: E402

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    import peft  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def _result(before=1.0, after=0.9, nan=False, minutes=30.0, seed=0):
    return {"seed": seed, "eval_before": {"mean_deviation": before}, "eval_after": {"mean_deviation": after},
            "nan_found": nan, "elapsed_seconds": {"total": minutes * 60.0}}


def test_judge_seed_branches():
    ok = R.judge_seed(_result())
    assert ok["pass"] and abs(ok["rel_drop"] - 0.1) < 1e-9 and ok["reasons"] == []
    small = R.judge_seed(_result(after=0.96))                   # 相対低下 4% < 5%
    assert not small["pass"] and any("相対低下" in r for r in small["reasons"])
    exact = R.judge_seed(_result(after=0.95))                   # ちょうど 5% は合格
    assert exact["pass"]
    nan = R.judge_seed(_result(nan=True))
    assert not nan["pass"] and nan["nan_found"]
    slow = R.judge_seed(_result(minutes=95.0))
    assert not slow["pass"] and any("所要" in r for r in slow["reasons"])
    zero = R.judge_seed(_result(before=0.0, after=0.0))         # before=0 では相対低下が定義できない
    assert not zero["pass"] and math.isnan(zero["rel_drop"])
    worse = R.judge_seed(_result(after=1.2))
    assert not worse["pass"] and worse["rel_drop"] < 0


def test_judge_all_requires_three_passing_seeds():
    three = R.judge_all([_result(seed=s) for s in range(3)])
    assert three["pass"] and three["n_pass"] == 3 and three["n_seeds"] == 3
    two = R.judge_all([_result(seed=s) for s in range(2)])
    assert not two["pass"] and two["n_pass"] == 2
    one_bad = R.judge_all([_result(seed=0), _result(seed=1, after=0.99), _result(seed=2)])
    assert not one_bad["pass"] and one_bad["n_pass"] == 2
    text = R.format_judgement(one_bad)
    assert "不合格" in text and "seed 1" in text


# ------------------------------------------------------------
# 極小モデル(CPU)
# ------------------------------------------------------------
class ToyTokenizer:
    """文字を語彙 64 に落とす玩具トークナイザ(HF トークナイザの使う部分だけを真似る)。"""
    eos_token_id = 3
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return messages[-1]["content"] + "\n"

    def __call__(self, text, **kwargs):
        return {"input_ids": [1] + [4 + (ord(c) % 56) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + (i % 26)) for i in ids if i > 3)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=3, bos_token_id=1, pad_token_id=0, tie_word_embeddings=False)
    m = Qwen2ForCausalLM(cfg)
    m.generation_config.eos_token_id = 3
    m.generation_config.pad_token_id = 0
    m.eval()
    return m


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/transformers/peft がない環境ではスキップ")
def test_tiny_model_two_updates_write_json_and_summary(tmp_path):
    cfg = R.G1Config(model="tiny-qwen2", G=4, lr=1e-2, beta_kl=0.04, updates=2, groups_per_update=2,
                     eval_episodes=1, max_new_tokens=6, micro_batch=3, time_limit_min=90.0)
    out_dir = str(tmp_path)
    res = R.run_seed(_tiny_model(), ToyTokenizer(), cfg, seed=0, out_dir=out_dir, device="cpu", log_fn=lambda s: None)
    path = os.path.join(out_dir, "g1_seed00.json")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        loaded = json.load(f)
    for key in ("condition", "seed", "config", "env_constants", "prompt_version", "n_trainable_params",
                "eval_before", "eval_after", "training", "episodes_used_for_training", "nan_found",
                "elapsed_seconds", "judgement"):
        assert key in loaded, key
    assert loaded["condition"] == "G1" and loaded["config"]["G"] == 4 and loaded["n_trainable_params"] > 0
    assert len(loaded["training"]) == 2
    for entry in loaded["training"]:
        for k in ("update", "mean_reward", "min_reward", "max_reward", "loss", "pg", "kl", "n_tokens", "episode", "elapsed_seconds"):
            assert k in entry, k
        assert math.isfinite(entry["mean_reward"]) and math.isfinite(entry["kl"]) and entry["n_tokens"] > 0
        assert entry["mean_reward"] <= 0.0                      # 報酬 = −逸脱 ≤ 0
    for ev in (loaded["eval_before"], loaded["eval_after"]):
        assert ev["n_episodes"] == 1 and ev["n_steps"] >= 1 and math.isfinite(ev["mean_deviation"])
        assert 0.0 <= ev["format_fail_rate"] <= 1.0 and len(ev["per_episode_mean_deviation"]) == 1
    assert loaded["nan_found"] is False
    assert set(loaded["elapsed_seconds"]) == {"eval_before", "train", "eval_after", "total"}
    assert loaded["elapsed_seconds"]["total"] > 0
    j = loaded["judgement"]
    assert set(j) >= {"pass", "rel_drop", "nan_found", "elapsed_min", "reasons"} and j["nan_found"] is False
    assert res["judgement"]["pass"] == j["pass"]
    # 1 seed しかないので全体は不合格(必要 3 seed)。summary が書かれる。
    summary = R.write_summary(out_dir, cfg)
    assert summary["n_seeds"] == 1 and summary["pass"] is False
    assert os.path.exists(os.path.join(out_dir, "g1_summary.json"))
