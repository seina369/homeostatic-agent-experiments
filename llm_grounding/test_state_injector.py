"""state_injector.py(関門 G2 の注入経路)の動作確認。極小の Qwen2(乱数初期化)を CPU で動かす。

確認すること(事前登録(第一段階)6章 G2 の「ゼロ初期化で注入なしと出力が一致する」を含む):
  1. ゼロ初期化では、注入ありと注入なしのロジットが完全一致(torch.equal)。
  2. 乱数初期化では、状態 z ごとにロジットが変わる。z=0 なら注入なしと一致。
  3. 層ごとのフロベニウスノルムを測り、指定値に揃えられる(random_init / scale_to)。
  4. KV キャッシュつきの逐次 forward が、フル forward と一致する(注入あり)。
  5. 勾配が W_l に流れ、GRPOTrainer(extra_params=...) で W_l が更新される。
  6. detach で元に戻る。バッチごとの z(B,3)は行ごとに効く。state_dict の往復。
torch / transformers / peft がない環境ではスキップ。Colab の CPU ランタイムで実行できる。
実行: cd llm_grounding && python3 -m pytest -q test_state_injector.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    import peft  # noqa: F401
    from state_injector import StateInjector, find_decoder_layers
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

pytestmark = pytest.mark.skipif(not HAVE_TORCH, reason="torch/transformers/peft がない環境ではスキップ")


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                      eos_token_id=3, bos_token_id=1, pad_token_id=0, tie_word_embeddings=False)
    m = Qwen2ForCausalLM(cfg)
    m.generation_config.eos_token_id = 3
    m.generation_config.pad_token_id = 0
    m.eval()
    return m


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor(ids)).logits.float()


IDS = [[1, 5, 9, 12, 7, 20]]
Z1 = [0.9, 0.1, 0.3]
Z2 = [0.2, 0.7, 0.9]


def test_zero_init_is_exactly_identity():
    model = _tiny_model()
    base = _logits(model, IDS)
    inj = StateInjector.for_model(model).attach(model)
    assert inj.n_layers == 3 and inj.hidden_size == 32 and inj.attached
    assert inj.layer_norms() == [0.0, 0.0, 0.0]
    inj.set_state(Z1)
    assert torch.equal(_logits(model, IDS), base)
    inj.set_state(Z2)
    assert torch.equal(_logits(model, IDS), base)
    inj.detach()
    assert not inj.attached and torch.equal(_logits(model, IDS), base)


def test_random_init_changes_output_per_state_and_zero_state_is_identity():
    model = _tiny_model()
    base = _logits(model, IDS)
    inj = StateInjector.for_model(model).attach(model)
    inj.random_init([0.5, 0.5, 0.5], seed=1)
    inj.set_state(Z1)
    out1 = _logits(model, IDS)
    inj.set_state(Z2)
    out2 = _logits(model, IDS)
    assert not torch.allclose(out1, base, atol=1e-6) and not torch.allclose(out2, base, atol=1e-6)
    assert not torch.allclose(out1, out2, atol=1e-6)
    inj.set_state([0.0, 0.0, 0.0])
    assert torch.allclose(_logits(model, IDS), base, atol=1e-6)
    inj.set_state(None)
    assert torch.equal(_logits(model, IDS), base)
    inj.enabled = False
    inj.set_state(Z1)
    assert torch.equal(_logits(model, IDS), base)


def test_layer_norms_can_be_measured_and_set():
    model = _tiny_model()
    inj = StateInjector.for_model(model)
    inj.random_init([0.1, 1.0, 2.5], seed=3)
    assert np.allclose(inj.layer_norms(), [0.1, 1.0, 2.5], atol=1e-5)
    inj.scale_to([0.3, 0.3, 0.0])
    assert np.allclose(inj.layer_norms(), [0.3, 0.3, 0.0], atol=1e-5)
    with pytest.raises(ValueError):
        inj.scale_to([0.3, 0.3, 0.3])           # ノルム 0 の層は方向がない
    with pytest.raises(ValueError):
        inj.scale_to([0.3, 0.3])                # 長さ違い
    inj.zero_()
    assert inj.layer_norms() == [0.0, 0.0, 0.0]


def test_incremental_forward_with_kv_cache_matches_full_forward():
    model = _tiny_model()
    inj = StateInjector.for_model(model).attach(model)
    inj.random_init([0.5, 0.5, 0.5], seed=2)
    inj.set_state(Z1)
    ids = torch.tensor(IDS)
    with torch.no_grad():
        full = model(input_ids=ids).logits.float()[0, -1]
        out = model(input_ids=ids[:, :4], use_cache=True)
        past = out.past_key_values
        out = model(input_ids=ids[:, 4:5], past_key_values=past, use_cache=True)
        past = out.past_key_values
        out = model(input_ids=ids[:, 5:6], past_key_values=past, use_cache=True)
        inc = out.logits.float()[0, -1]
    assert torch.allclose(full, inc, atol=1e-5)


def test_gradient_flows_and_grpo_step_updates_injector():
    from grpo_trainer import GRPOConfig, GRPOTrainer, attach_lora
    model = _tiny_model()
    inj = StateInjector.for_model(model).attach(model)
    inj.random_init([0.3, 0.3, 0.3], seed=4)
    inj.set_state(Z1)
    out = model(input_ids=torch.tensor(IDS)).logits.float().sum()
    out.backward()
    assert all(W.grad is not None and float(W.grad.abs().sum()) > 0 for W in inj.weights)
    model.zero_grad()
    for W in inj.weights:
        W.grad = None
    # GRPO: LoRA と注入器を一緒に更新(peft でラップしてもフックは同じ層に残る)
    cfg = GRPOConfig(group_size=4, lr=1e-2, beta_kl=0.0, max_new_tokens=3,
                     lora_targets=("q_proj", "v_proj"), lora_r=2)
    lmodel = attach_lora(model, cfg)
    assert len(find_decoder_layers(lmodel)) == 3
    tr = GRPOTrainer(lmodel, type("T", (), {"eos_token_id": 3, "pad_token_id": 0})(), cfg,
                     extra_params=list(inj.parameters()), device="cpu")
    before = [W.detach().clone() for W in inj.weights]
    torch.manual_seed(0)
    prompt = [1, 5, 9]
    resps = tr.sample_group(prompt, cfg.group_size)
    resps[0] = [7, 3]
    stats = tr.step([{"prompt_ids": prompt, "responses": resps,
                      "rewards": [1.0 if 7 in r else 0.0 for r in resps]}])
    assert np.isfinite(stats["loss"])
    assert any(not torch.equal(W.detach(), b) for W, b in zip(inj.weights, before)), "注入器が更新されていない"
    assert inj.attached


def test_norm_cap_and_reference_policy_keeps_injection():
    """3章: (a) 層ごとのノルム上限 clamp_norms を post_step で呼ぶと、更新後も上限を超えない。
    (b) 参照方策(LoRA 無効化)でも注入は有効のまま(注入ありの LoRA なしモデルと一致)。"""
    from grpo_trainer import GRPOConfig, GRPOTrainer, attach_lora
    model = _tiny_model()
    inj = StateInjector.for_model(model).attach(model)
    inj.random_init([0.3, 0.3, 0.3], seed=6)
    inj.set_state(Z1)
    # clamp_norms 単体: 上限を超えた層だけ押し戻す(方向は保つ)
    w0 = inj.weights[0].detach().clone()
    clipped = inj.clamp_norms([0.1, 1.0, 0.3])
    assert clipped == [0] and np.allclose(inj.layer_norms(), [0.1, 0.3, 0.3], atol=1e-5)
    assert torch.allclose(inj.weights[0].detach() / 0.1, w0 / 0.3, atol=1e-5)
    assert inj.clamp_norms(1.0) == []
    # (b) 参照方策: LoRA を付ける前の「注入あり」モデルの対数確率 = LoRA 付与後の score_ref
    cfg = GRPOConfig(group_size=4, lr=5e-2, beta_kl=0.0, max_new_tokens=3,
                     lora_targets=("q_proj", "v_proj"), lora_r=2)
    prompt, resps = [1, 5, 9], [[7, 3], [12, 20, 3]]
    with torch.no_grad():
        ref_expected = []
        for r in resps:
            lp_all = torch.log_softmax(model(input_ids=torch.tensor([prompt + r])).logits.float(), dim=-1)[0]
            ref_expected.append(torch.stack([lp_all[len(prompt) - 1 + t, r[t]] for t in range(len(r))]))
    lmodel = attach_lora(model, cfg)
    with torch.no_grad():                    # LoRA をゼロでなくして、現在の方策と参照方策を分ける
        for n, p in lmodel.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.5)
    cap = [0.3, 0.3, 0.3]
    tr = GRPOTrainer(lmodel, type("T", (), {"eos_token_id": 3, "pad_token_id": 0})(), cfg,
                     extra_params=list(inj.parameters()), device="cpu",
                     post_step=lambda: inj.clamp_norms(cap))
    cur = [lp.detach() for lp, _ in tr.score(prompt, resps, grad=False)]
    ref = tr.score_ref(prompt, resps)
    assert not torch.allclose(cur[1], ref_expected[1], atol=1e-5)     # LoRA ありは変わる
    for a, b in zip(ref, ref_expected):                                # LoRA なし・注入ありと一致
        assert torch.allclose(a, b, atol=1e-5)
    inj.set_state(Z2)                        # 参照方策も z に追従する(状況として見せる)
    ref2 = tr.score_ref(prompt, resps)
    assert not torch.allclose(ref2[1], ref[1], atol=1e-5)
    inj.set_state(Z1)
    # (a) 大きな学習率で数ステップ更新しても、post_step の押し戻しで上限を超えない
    for _ in range(5):
        torch.manual_seed(0)
        rs = tr.sample_group(prompt, cfg.group_size)
        rs[0] = [7, 3]
        tr.step([{"prompt_ids": prompt, "responses": rs, "rewards": [1.0 if 7 in r else 0.0 for r in rs]}])
        assert all(n <= c + 1e-5 for n, c in zip(inj.layer_norms(), cap)), inj.layer_norms()


def test_batched_state_rows_and_state_dict_roundtrip():
    model = _tiny_model()
    inj = StateInjector.for_model(model).attach(model)
    inj.random_init([0.5, 0.5, 0.5], seed=5)
    ids = torch.tensor(IDS * 2)
    inj.set_state(torch.tensor([Z1, Z2]))
    with torch.no_grad():
        both = model(input_ids=ids).logits.float()
    inj.set_state(Z1)
    with torch.no_grad():
        one = model(input_ids=torch.tensor(IDS)).logits.float()
    assert torch.allclose(both[0], one[0], atol=1e-5)
    assert not torch.allclose(both[1], one[0], atol=1e-5)
    with pytest.raises(ValueError):
        inj.set_state(torch.zeros(3, 3))          # バッチ 3 と hidden のバッチ 2 が合わない
        with torch.no_grad():
            model(input_ids=ids)
    inj2 = StateInjector(3, 32)
    inj2.load_state_dict(inj.state_dict())
    assert np.allclose(inj2.layer_norms(), inj.layer_norms())
    assert all(torch.equal(a, b) for a, b in zip(inj2.weights, inj.weights))
