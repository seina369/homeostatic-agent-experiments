"""状態注入器(関門 G2 の中身。事前登録(第一段階)2章「内部状態の注入」)。

三信号の正規化ベクトル z ∈ R^3 を、層ごとの線形写像 W_l ∈ R^{3×d}(バイアスなし)で隠れ次元 d に
写し、各デコーダ層に入る残差ストリームに**全トークン位置**で加算する: h_l ← h_l + z·W_l。
  - ゼロ初期化(既定): 注入なしと出力(ロジット)が完全一致する(加算されるのが厳密に 0)。
  - 乱数初期化 random_init(norms): 各層の W_l を乱数方向にし、フロベニウスノルムを指定値に揃える
    (G2 のスケール掃引、C0' の「T1 と同じノルムの乱数方向」用)。
  - ノルム: layer_norms() で層ごとの ||W_l||_F を測り、scale_to(norms) で指定値に揃える。
  - 学習: W_l は nn.Parameter。GRPOTrainer(model, tok, cfg, extra_params=list(injector.parameters()))
    で LoRA と一緒に更新する。
実装は各デコーダ層の forward pre-hook(層の入力 hidden_states に加算)。KV キャッシュつきの逐次生成
でも新しい位置に同じベクトルが足されるので、フル forward と一致する(test_state_injector.py で確認)。
z は set_state(z) で与える(1 本 (3,) か、バッチごと (B,3))。None なら何も足さない。
TorchPolicy からは policy.set_state(z) → respond() の順で使う(TorchPolicy(injector=...))。
参照方策(GRPO の KL の相手)は「注入を有効にした状態(同じ W、LoRA なし)」とする(3章。注入は
方策の変更ではなく状況なので、参照に見せないと KL が状態依存そのものを罰するため)。GRPOTrainer の
score_ref は LoRA だけを無効化するので、フックを付けたままにすればこの定義になる。
W の層ごとのフロベニウスノルムには上限を設け、更新のたびに clamp_norms(max_norms) で押し戻す
(3章。上限の値は G2 の「壊れない最大スケール」で、G2 後に追記欄で固定)。GRPOTrainer(post_step=...)
に渡す。
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


def find_decoder_layers(model) -> List[nn.Module]:
    """HF の因果言語モデル(素のモデル / peft でラップしたもの)からデコーダ層のリストを取り出す。"""
    for path in ("model.layers", "base_model.model.model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        ok = True
        for name in path.split("."):
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok and isinstance(obj, (nn.ModuleList, list)) and len(obj) > 0:
            return list(obj)
    raise ValueError("デコーダ層が見つからない(model.layers / base_model.model.model.layers / transformer.h)")


class StateInjector(nn.Module):
    def __init__(self, n_layers: int, hidden_size: int, n_signals: int = 3, dtype=torch.float32):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.n_signals = n_signals
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_signals, hidden_size, dtype=dtype)) for _ in range(n_layers)])
        self.enabled = True
        self.z: Optional[torch.Tensor] = None
        self._handles = []

    @classmethod
    def for_model(cls, model, n_signals: int = 3, dtype=torch.float32) -> "StateInjector":
        layers = find_decoder_layers(model)
        cfg = getattr(model, "config", None)
        hidden = getattr(cfg, "hidden_size", None)
        if hidden is None:                     # peft ラップなど
            hidden = model.get_input_embeddings().weight.shape[1]
        inj = cls(len(layers), hidden, n_signals, dtype)
        dev = next(model.parameters()).device
        return inj.to(dev)

    # ------------------------------------------------------------
    # 状態
    # ------------------------------------------------------------
    def set_state(self, z):
        """z: None | 長さ n_signals の列 | tensor (n_signals,) | tensor (B, n_signals)。"""
        if z is None:
            self.z = None
            return
        t = torch.as_tensor(z, dtype=self.weights[0].dtype, device=self.weights[0].device)
        if t.dim() == 1:
            t = t[None, :]
        if t.dim() != 2 or t.shape[1] != self.n_signals:
            raise ValueError(f"z の形が不正: {tuple(t.shape)}(期待: ({self.n_signals},) か (B, {self.n_signals}))")
        self.z = t

    def clear_state(self):
        self.z = None

    # ------------------------------------------------------------
    # フック
    # ------------------------------------------------------------
    def _make_hook(self, layer_idx: int):
        W = self.weights[layer_idx]

        def hook(module, args, kwargs):
            if not self.enabled or self.z is None:
                return None
            if args:
                h = args[0]
            else:
                h = kwargs["hidden_states"]
            inj = (self.z @ W).to(h.dtype)[:, None, :]              # (B or 1, 1, d)
            if inj.shape[0] != 1 and inj.shape[0] != h.shape[0]:
                raise ValueError(f"z のバッチ {inj.shape[0]} と hidden のバッチ {h.shape[0]} が合わない")
            h2 = h + inj
            if args:
                return (h2,) + tuple(args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = h2
            return args, kwargs
        return hook

    def attach(self, model) -> "StateInjector":
        """各デコーダ層に pre-hook を付ける(二重に付けない)。"""
        self.detach()
        layers = find_decoder_layers(model)
        if len(layers) != self.n_layers:
            raise ValueError(f"層数が合わない: injector {self.n_layers}, model {len(layers)}")
        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_pre_hook(self._make_hook(i), with_kwargs=True))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    # ------------------------------------------------------------
    # ノルム
    # ------------------------------------------------------------
    @torch.no_grad()
    def layer_norms(self) -> List[float]:
        """層ごとの ||W_l||_F。"""
        return [float(torch.linalg.norm(W.float())) for W in self.weights]

    @torch.no_grad()
    def scale_to(self, norms: Sequence[float]):
        """各層の W_l を、方向を保ったまま指定のフロベニウスノルムに揃える(ノルム 0 の層は揃えられない)。"""
        norms = list(norms)
        if len(norms) != self.n_layers:
            raise ValueError(f"norms の長さ {len(norms)} が層数 {self.n_layers} と違う")
        for W, target in zip(self.weights, norms):
            cur = float(torch.linalg.norm(W.float()))
            if target == 0.0:
                W.zero_()
            elif cur == 0.0:
                raise ValueError("ノルム 0 の層は方向がないので揃えられない(random_init を使う)")
            else:
                W.mul_(target / cur)
        return self

    @torch.no_grad()
    def clamp_norms(self, max_norms):
        """層ごとのフロベニウスノルムが上限を超えていれば、方向を保って上限まで押し戻す。
        max_norms: 層ごとの列、または全層共通のスカラー。戻り値: 押し戻した層の番号のリスト。"""
        if isinstance(max_norms, (int, float)):
            max_norms = [float(max_norms)] * self.n_layers
        max_norms = list(max_norms)
        if len(max_norms) != self.n_layers:
            raise ValueError(f"max_norms の長さ {len(max_norms)} が層数 {self.n_layers} と違う")
        clipped = []
        for i, (W, cap) in enumerate(zip(self.weights, max_norms)):
            cur = float(torch.linalg.norm(W.float()))
            if cur > cap * (1.0 + 1e-6):            # 丸め誤差の分は超過とみなさない
                W.mul_(cap / cur if cur > 0 else 0.0)
                clipped.append(i)
        return clipped

    @torch.no_grad()
    def random_init(self, norms: Sequence[float], seed: int = 0):
        """各層を乱数方向(正規乱数)にし、フロベニウスノルムを norms に揃える。"""
        g = torch.Generator(device="cpu").manual_seed(seed)
        for W in self.weights:
            W.copy_(torch.randn(W.shape, generator=g, dtype=torch.float32).to(W.dtype))
        return self.scale_to(norms)

    @torch.no_grad()
    def zero_(self):
        for W in self.weights:
            W.zero_()
        return self
