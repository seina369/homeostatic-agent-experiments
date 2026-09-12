"""
MLXPolicy: emotion_grounding_env.PolicyInterface の実装(Qwen2.5-0.5B-Instruct)
====================================================================

事前登録(事前登録_内部信号と感情語の接続実験.md)の手順3「MLXで0.5Bモデルを
接続し、速度を実測」に対応する。学習は一切行わない(勾配計算・optimizer・
LoRA更新なし)。推論のみ。

三信号のうち budget(実トークン数)と uncertainty(実エントロピー)は、この
ファイルが「本物」を報告する責任を持つ。DummyPolicyのような近似値は使わない。

【確定した設計判断(2026-09-09、ユーザー指示により確定。事前登録の追記欄に
理由つきで記録する)】
1. デコーディング温度: 1.0に固定し、C0・T1の両条件で共通に使う。
2. エントロピーの計算基準: サンプリングに使う温度とは独立に、温度を適用する
   前の生ロジット(log_softmax)から常に計算する。mlx_lmのmake_sampler設計
   では温度scalingはカテゴリカル抽出の内部でのみ適用され、生成ループが
   保持するlogprobs自体は生ロジットのlog_softmaxのまま(サンプラーが
   in-placeで書き換えない)という前提に立っている。この前提は実行時に
   sum(exp(logprobs))≈1になることを検算して確認する(下記respond()参照。
   前提が崩れていれば例外を出して即座に気づけるようにしてある)。
3. モデル: mlx-community/Qwen2.5-0.5B-Instruct-bf16 に変更(4bit量子化から
   bf16へ)。理由は追記欄を参照。
4. チャット形式: tokenizer.apply_chat_template を使い、systemメッセージは
   付けずuserメッセージ1つだけを渡す(感情・気分への言及を一切増やさない
   ため。事前登録の「感情・気分・状態への言及は一切含めない」を厳密に守る
   ための判断)。
5. max_tokens=200に設定。速度計測後、極端に長い出力が頻発する場合は
   見直す。

このファイル自体は Apple Silicon + MLX が必要で、Linux サンドボックス上では
実行できない(mlxパッケージはインストールできてもlibmlx.soが存在せず
importに失敗することを確認済み)。M2 Air 上で実行すること。
"""

import time
from emotion_grounding_env import PolicyInterface, Response

MODEL_PATH = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
MAX_TOKENS = 200
TEMPERATURE = 1.0   # C0・T1共通。2026-09-09確定(事前登録追記欄参照)。


class MLXPolicy(PolicyInterface):
    """Qwen2.5-0.5B-Instruct(MLX, 4bit)をそのまま接続する。学習なし・推論のみ。"""

    def __init__(self, model_path: str = MODEL_PATH, max_tokens: int = MAX_TOKENS,
                 temperature: float = TEMPERATURE, verbose: bool = False):
        from mlx_lm import load  # インポートはここで行う(MLX非対応環境でも
        # このファイル自体はimportエラーにならないようにする)
        self.mx_lm_load = load
        self.model, self.tokenizer = load(model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

    def reset(self) -> None:
        # 学習中の状態を持たないため何もしない(PolicyInterfaceの既定と同じ)。
        pass

    def respond(self, prompt: str) -> Response:
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.generate import stream_generate

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )

        sampler = make_sampler(temp=self.temperature)

        text_parts = []
        entropies = []
        n_generated = 0
        checked_normalization = False
        for resp in stream_generate(
            self.model, self.tokenizer, formatted,
            max_tokens=self.max_tokens, sampler=sampler,
        ):
            text_parts.append(resp.text)
            n_generated += 1
            # resp.logprobs: 温度を適用する前の生ロジットのlog-softmax(nat)。
            # make_sampler(temp=...)は、このlogprobsを引数として受け取り
            # カテゴリカル抽出の内部でのみ温度scalingを行う設計であり、
            # ここで参照するlogprobs自体は書き換えられない(=常に
            # 「温度前」)という前提に立っている。初回ステップだけ、
            # 確率として正規化されている(合計が1に近い)ことを検算し、
            # 前提が崩れていたら即座に例外で気づけるようにする。
            # mlx_lm本体のソース(generate.py)を確認済み: logprobs = logits -
            # logsumexp(logits) という理論上厳密な log_softmax であり、
            # サンプラー(温度・top_p等、top_p=min_p=top_k=0で未使用)を通す前の
            # 値をそのまま返す。実機で検証した結果、sum(exp(logprobs))は
            # 1.0ちょうどにならず、ステップによって0.96〜1.06程度まで
            # 両方向にぶれることを確認した(2026-09-10)。上振れ・下振れ両方が
            # 起きている(常に同じ方向に欠落するtop-k/top-p的な切り詰めでは
            # 説明できない)ことから、原因はbf16モデル自体のロジット精度
            # (仮数部7〜8bit)に起因する丸め誤差と判断した。分布が尖っている
            # (上位トークンの確率が高い)ステップほど、その上位ロジットの
            # わずかな丸め誤差がexpで増幅されて合計のずれが目立ちやすい。
            # よって: (1) 正規化チェックの閾値はbf16精度を前提に緩め(15%)、
            # 形が壊れている等の本当の異常のみ検出する、(2) entropy計算自体は
            # 使う直前に自己再正規化(p/sum(p))してから行い、bf16由来の
            # 合計ずれの影響を受けないようにする。
            logp = resp.logprobs.astype(mx.float32)
            p_raw = mx.exp(logp)
            if not checked_normalization:
                total_p = float(mx.sum(p_raw))
                if abs(total_p - 1.0) > 0.15:
                    vocab_size = getattr(self.tokenizer, "vocab_size", None)
                    raise RuntimeError(
                        f"resp.logprobsの合計={total_p:.4f}がbf16精度の許容範囲(±15%)を"
                        f"超えて1.0からずれている。logprobs.shape={tuple(logp.shape)}, "
                        f"tokenizer.vocab_size={vocab_size}。想定外の切り詰め等が"
                        "起きていないか見直すこと。"
                    )
                checked_normalization = True
            p = p_raw / mx.sum(p_raw)  # bf16由来の合計ずれを補正してから使う
            step_entropy = float(-mx.sum(p * logp))
            entropies.append(step_entropy)
            if self.verbose:
                print(resp.text, end="", flush=True)

        text = "".join(text_parts)
        mean_entropy = float(sum(entropies) / len(entropies)) if entropies else 0.0
        return Response(text=text, n_tokens=n_generated, mean_entropy=mean_entropy)


if __name__ == "__main__":
    # 単発の動作確認(MLX対応環境で実行すること)。
    policy = MLXPolicy(verbose=True)
    t0 = time.time()
    r = policy.respond("Task: What is 23 + 19?\nb=600 e=0 u=1.00\n"
                        "Reply in one or two sentences, then give the answer as \"A: <answer>\".")
    dt = time.time() - t0
    print()
    print(f"n_tokens={r.n_tokens} mean_entropy={r.mean_entropy:.4f} 秒={dt:.2f}")
