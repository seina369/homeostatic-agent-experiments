"""GRPO の動作確認。

  - numpy の核(grpo_core.py)は常にテストする(GPU・torch 不要)。
  - torch トレーナー(grpo_trainer.py)は、torch / transformers / peft がある環境でだけ
    極小の Qwen2(隠れ次元 32、2層、語彙 64、乱数初期化)を CPU で動かして確認する。
    ない環境では skip。Colab の CPU ランタイムで実行できる。
実行: cd llm_grounding && python3 -m pytest -q test_grpo.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grpo_core as C  # noqa: E402

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    import peft  # noqa: F401
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


# ------------------------------------------------------------
# numpy の核
# ------------------------------------------------------------
def test_group_advantages_are_standardised_and_zero_when_equal():
    a = C.group_advantages([1.0, 2.0, 3.0, 4.0])
    assert abs(a.mean()) < 1e-12 and abs(a.std() - 1.0) < 1e-3
    assert np.all(C.group_advantages([0.5, 0.5, 0.5]) == 0.0)
    assert C.group_advantages([]).size == 0


def test_kl_k3_is_zero_when_equal_and_positive_otherwise():
    lp = np.log(np.array([0.2, 0.5, 0.3]))
    assert np.allclose(C.kl_k3(lp, lp), 0.0)
    other = np.log(np.array([0.1, 0.6, 0.3]))
    assert np.all(C.kl_k3(lp, other) >= 0.0) and C.kl_k3(lp, other).sum() > 0


def test_clipped_term_equals_minus_advantage_when_on_policy():
    lp = np.array([-1.0, -2.0, -0.5])
    term = C.clipped_pg_term(lp, lp, advantage=0.7)
    assert np.allclose(term, -0.7)
    # 比率が大きくずれると、優位が正のときはクリップ側(小さい方)が選ばれる
    big = C.clipped_pg_term(lp + 1.0, lp, advantage=1.0, clip_eps=0.2)
    assert np.allclose(big, -1.2)


def test_grpo_loss_combines_pg_and_kl():
    resp = [dict(logp=np.array([-1.0, -1.0]), logp_old=np.array([-1.0, -1.0]),
                 logp_ref=np.array([-1.0, -1.0]), advantage=1.0),
            dict(logp=np.array([-2.0]), logp_old=np.array([-2.0]),
                 logp_ref=np.array([-1.0]), advantage=-1.0)]
    loss, pg, kl = C.grpo_loss(resp, beta_kl=0.5)
    assert abs(pg - 0.0) < 1e-12                       # (-1 + 1) / 2
    expected_kl = (0.0 + (np.exp(1.0) - 1.0 - 1.0)) / 2
    assert abs(kl - expected_kl) < 1e-12
    assert abs(loss - (pg + 0.5 * kl)) < 1e-12


# ------------------------------------------------------------
# torch トレーナー(極小モデル、CPU)
# ------------------------------------------------------------
class _ToyTokenizer:
    eos_token_id = 3
    pad_token_id = 0


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      eos_token_id=3, bos_token_id=1, pad_token_id=0, tie_word_embeddings=False)
    m = Qwen2ForCausalLM(cfg)
    m.generation_config.eos_token_id = 3
    m.generation_config.pad_token_id = 0
    return m


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/transformers/peft がない環境ではスキップ")
def test_score_matches_manual_forward_and_torch_kl_matches_numpy():
    from grpo_trainer import GRPOConfig, GRPOTrainer, attach_lora
    model = attach_lora(_tiny_model(), GRPOConfig(lora_targets=("q_proj", "v_proj")))
    tr = GRPOTrainer(model, _ToyTokenizer(), GRPOConfig(max_new_tokens=8), device="cpu")
    prompt = [1, 5, 9, 12]
    resp = [7, 20, 3]
    (logp, ent), = tr.score(prompt, [resp], grad=False)
    with torch.no_grad():
        ids = torch.tensor([prompt + resp])
        lp_all = torch.log_softmax(model(input_ids=ids).logits.float(), dim=-1)[0]
    manual = torch.stack([lp_all[len(prompt) - 1 + t, resp[t]] for t in range(len(resp))])
    assert torch.allclose(logp, manual, atol=1e-5)
    assert ent.shape == (3,) and bool((ent > 0).all())
    # 参照方策(LoRA 無効化)は、LoRA がゼロ初期化(B=0)の直後は現在の方策と一致する
    ref = tr.score_ref(prompt, [resp])[0]
    assert torch.allclose(ref, logp, atol=1e-5)
    # torch 版の KL 推定量が numpy 版と一致する
    r = (ref - logp + 0.3).detach()
    kl_t = (torch.exp(r) - r - 1.0).numpy()
    kl_n = C.kl_k3((ref + 0.3).numpy(), logp.numpy())
    assert np.allclose(kl_t, kl_n, atol=1e-6)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch/transformers/peft がない環境ではスキップ")
def test_one_grpo_step_updates_lora_only_and_moves_policy_toward_reward():
    from grpo_trainer import GRPOConfig, GRPOTrainer, attach_lora
    cfg = GRPOConfig(group_size=6, lr=5e-2, beta_kl=0.0, max_new_tokens=4,
                     lora_targets=("q_proj", "v_proj"), lora_r=4)
    model = attach_lora(_tiny_model(), cfg)
    tr = GRPOTrainer(model, _ToyTokenizer(), cfg, device="cpu")
    base_before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora" not in n}
    lora_before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora" in n}
    prompt = [1, 5, 9]
    LUCKY = 7

    def reward(resp):                      # 報酬: LUCKY トークンを含めば 1、それ以外 0(文の内容は見ない)
        return 1.0 if LUCKY in resp else 0.0

    def p_lucky_first():
        with torch.no_grad():
            lp = torch.log_softmax(model(input_ids=torch.tensor([prompt])).logits[0, -1].float(), dim=-1)
        return float(lp[LUCKY].exp())

    p0 = p_lucky_first()
    stats = None
    for _ in range(25):
        torch.manual_seed(0)
        groups = []
        for _g in range(2):
            resps = tr.sample_group(prompt, cfg.group_size)
            # 全員同じ報酬だと優位が 0 で学習しないので、1本だけ LUCKY を含む応答に差し替えて信号を作る
            resps[0] = [LUCKY, 3]
            groups.append({"prompt_ids": prompt, "responses": resps, "rewards": [reward(r) for r in resps]})
        stats = tr.step(groups)
        assert np.isfinite(stats["loss"]) and stats["n_tokens"] > 0
    p1 = p_lucky_first()
    for n, p in model.named_parameters():
        if "lora" not in n:
            assert torch.equal(p.detach(), base_before[n]), f"ベースの重みが動いた: {n}"
    assert any(not torch.equal(p.detach(), lora_before[n]) for n, p in model.named_parameters() if "lora" in n), \
        "LoRA の重みが動いていない"
    assert p1 > p0, f"報酬のあるトークンの確率が上がっていない: {p0:.4f} -> {p1:.4f}"
