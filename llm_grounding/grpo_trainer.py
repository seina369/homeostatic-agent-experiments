"""GRPO トレーナー(torch / transformers / peft)。事前登録(第一段階)3章の学習部分。

方策クラス(TorchPolicy)には依存しない。HF の因果言語モデルとトークナイザ、それに
「プロンプト ids → 応答 ids → 報酬」を与える呼び出し側があれば動く。

構成:
  - attach_lora(model, ...): peft の LoRA を注意層・MLP 層に付ける(ベースは凍結)。
  - GRPOTrainer:
      sample_group(prompt_ids, G)        同じプロンプトから G 本サンプル(温度つき)。
      score(prompt_ids, resp_ids)        応答トークンの対数確率(勾配つき)と、生ロジットの
                                         エントロピー(環境の uncertainty 用。温度をかける前)。
      step(groups)                       グループ相対の優位 → クリップつき方策勾配 + β·KL → 更新。
                                         参照方策 π_ref は「同じモデルで LoRA を無効化したもの」
                                         (peft の disable_adapter)。モデルの複製は持たない。
                                         注入器(state_injector)のフックは無効化しないので、
                                         参照方策は「注入あり・LoRA なし」になる(3章の定義)。
  - 追加の学習対象(第一段階の注入変換器など)は extra_params で渡す。更新のたびに呼ぶ処理
    (注入器のノルム上限 clamp_norms)は post_step で渡す。

数式は grpo_core.py(numpy)と同じ。test_grpo.py で両者の一致と、極小モデルでの
1ステップ(CPU)を確認する。本番(Qwen2.5-1.5B-Instruct、T4)は関門 G1 で行う。
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    group_size: int = 8
    lr: float = 1e-5
    beta_kl: float = 0.04
    clip_eps: float = 0.2
    temperature: float = 1.0
    max_new_tokens: int = 200
    epochs_per_batch: int = 1     # 1 なら on-policy(比率=1)。>1 でクリップが効く。
    grad_clip: float = 1.0
    adv_eps: float = 1e-6
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: Sequence[str] = field(default_factory=lambda: (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))


def attach_lora(model, cfg: GRPOConfig):
    """LoRA を付けたモデルを返す(ベースの重みは凍結)。peft が必要。"""
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(cfg.lora_targets), bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, lcfg)


def _pad_batch(seqs: List[List[int]], pad_id: int, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask[i, :len(s)] = 1
    return ids, mask


class GRPOTrainer:
    def __init__(self, model, tokenizer, cfg: GRPOConfig, extra_params: Optional[list] = None,
                 device: Optional[str] = None, post_step: Optional[Callable[[], None]] = None):
        """post_step: 各更新(optimizer.step)の直後に呼ぶ関数(例: 注入器のノルム上限
        lambda: injector.clamp_norms(max_norms)。事前登録(第一段階)3章)。"""
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.post_step = post_step
        self.device = device or next(model.parameters()).device
        params = [p for p in model.parameters() if p.requires_grad]
        if extra_params:
            params += list(extra_params)
        if not params:
            raise ValueError("学習対象のパラメータがない(LoRA を付けるか extra_params を渡す)")
        self.optimizer = torch.optim.AdamW(params, lr=cfg.lr)
        self.eos_ids = self._eos_ids()
        self.pad_id = getattr(tokenizer, "pad_token_id", None)
        if self.pad_id is None:
            self.pad_id = self.eos_ids[0] if self.eos_ids else 0

    def _eos_ids(self):
        gen_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if gen_eos is None:
            gen_eos = getattr(self.tok, "eos_token_id", None)
        if gen_eos is None:
            return []
        return [gen_eos] if isinstance(gen_eos, int) else list(gen_eos)

    # ------------------------------------------------------------
    # サンプリング
    # ------------------------------------------------------------
    @torch.no_grad()
    def sample_group(self, prompt_ids: List[int], G: Optional[int] = None) -> List[List[int]]:
        """同じプロンプトから G 本、温度 cfg.temperature で独立にサンプルする(EOS または上限で停止)。
        戻り値は応答トークン id のリスト(EOS を含む場合はそれも含む)。"""
        G = G or self.cfg.group_size
        self.model.eval()
        ids = torch.tensor([prompt_ids] * G, dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids)
        past = None
        out_ids = [[] for _ in range(G)]
        done = [False] * G
        cur = ids
        for _ in range(self.cfg.max_new_tokens):
            out = self.model(input_ids=cur, attention_mask=mask, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :].float()
            past = out.past_key_values
            probs = torch.softmax(logits / self.cfg.temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)          # (G, 1)
            for i in range(G):
                if not done[i]:
                    t = int(nxt[i, 0])
                    out_ids[i].append(t)
                    if t in self.eos_ids:
                        done[i] = True
            if all(done):
                break
            cur = nxt
            mask = torch.cat([mask, torch.ones((G, 1), dtype=mask.dtype, device=self.device)], dim=1)
        return out_ids

    # ------------------------------------------------------------
    # 採点: 応答トークンの対数確率とエントロピー
    # ------------------------------------------------------------
    def score(self, prompt_ids: List[int], responses: List[List[int]], grad: bool = True):
        """各応答について (logp: (T_i,), entropy: (T_i,)) を返す。
        logp は温度をかけないロジットの log_softmax(方策の定義)。entropy も同じ生ロジットから。
        grad=False なら no_grad で計算(参照方策や old の値に使う)。"""
        seqs = [list(prompt_ids) + list(r) for r in responses]
        ids, mask = _pad_batch(seqs, self.pad_id, self.device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = out.logits.float()                              # (B, L, V)
            logp_all = torch.log_softmax(logits, dim=-1)
            P = len(prompt_ids)
            results = []
            for i, r in enumerate(responses):
                T = len(r)
                if T == 0:
                    z = torch.zeros(0, device=self.device)
                    results.append((z, z))
                    continue
                # 応答トークン t (位置 P+t) を予測するロジットは位置 P+t-1
                pos = torch.arange(P - 1, P - 1 + T, device=self.device)
                lp = logp_all[i, pos, :]                             # (T, V)
                tgt = torch.tensor(r, dtype=torch.long, device=self.device)
                logp = lp.gather(1, tgt[:, None])[:, 0]              # (T,)
                ent = -(lp.exp() * lp).sum(dim=-1)                   # (T,)
                results.append((logp, ent))
        return results

    def score_ref(self, prompt_ids, responses):
        """参照方策(LoRA 無効化)での対数確率。peft でなければ現在のモデルをそのまま使う。"""
        if hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                return [lp.detach() for lp, _ in self.score(prompt_ids, responses, grad=False)]
        return [lp.detach() for lp, _ in self.score(prompt_ids, responses, grad=False)]

    # ------------------------------------------------------------
    # 更新
    # ------------------------------------------------------------
    @staticmethod
    def advantages(rewards, eps):
        r = torch.tensor(rewards, dtype=torch.float32)
        if r.numel() == 0:
            return r
        return (r - r.mean()) / (r.std(unbiased=False) + eps)

    def step(self, groups):
        """groups: list of dict(prompt_ids=[...], responses=[[...], ...], rewards=[...])。
        戻り値: dict(loss, pg, kl, n_tokens)。cfg.epochs_per_batch 回更新する。"""
        cfg = self.cfg
        # old と ref は更新前に一度だけ計算(no grad)
        prepared = []
        for g in groups:
            adv = self.advantages(g["rewards"], cfg.adv_eps)
            old = [lp.detach() for lp, _ in self.score(g["prompt_ids"], g["responses"], grad=False)]
            ref = self.score_ref(g["prompt_ids"], g["responses"])
            prepared.append((g, adv, old, ref))
        stats = {}
        for _ in range(cfg.epochs_per_batch):
            self.model.train()
            self.optimizer.zero_grad()
            pg_terms, kl_terms, n_tok = [], [], 0
            for g, adv, old, ref in prepared:
                new = self.score(g["prompt_ids"], g["responses"], grad=True)
                for i, (logp, _) in enumerate(new):
                    if logp.numel() == 0:
                        continue
                    ratio = torch.exp(logp - old[i])
                    a = adv[i].to(logp.device)
                    unclipped = ratio * a
                    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * a
                    pg = -torch.minimum(unclipped, clipped).mean()
                    r = ref[i] - logp
                    kl = (torch.exp(r) - r - 1.0).mean()
                    pg_terms.append(pg)
                    kl_terms.append(kl)
                    n_tok += logp.numel()
            if not pg_terms:
                return {"loss": 0.0, "pg": 0.0, "kl": 0.0, "n_tokens": 0}
            pg_mean = torch.stack(pg_terms).mean()
            kl_mean = torch.stack(kl_terms).mean()
            loss = pg_mean + cfg.beta_kl * kl_mean
            loss.backward()
            params = [p for grp in self.optimizer.param_groups for p in grp["params"]]
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            self.optimizer.step()
            if self.post_step is not None:
                self.post_step()
            stats = {"loss": loss.detach().item(), "pg": pg_mean.detach().item(),
                     "kl": kl_mean.detach().item(), "n_tokens": n_tok}
        return stats
