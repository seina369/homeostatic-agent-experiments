"""
TorchPolicy: emotion_grounding_env.PolicyInterface の実装(Qwen2.5-1.5B-Instruct, PyTorch)
====================================================================

計算環境をM2 MacBook Air/MLXからGoogle Colab無料枠(T4 GPU 16GB)へ変更した
ことに伴う実装(2026-09-10確定。理由は事前登録の追記欄「2026-09-10」を参照:
0.5Bモデルでは正答率8%でerror信号がほぼ死んでいたこと、M2 Airの8GB上限、
T4無料枠なら1.5Bが動くこと)。学習は一切行わない(勾配計算なし。モデルは
eval()+torch.no_grad()で常時推論のみ)。

mlx_qwen_policy.py(MLXPolicy)と同じ設計判断を引き継ぐ:
1. デコーディング温度: 1.0に固定し、C0・T1の両条件で共通(変更なし)。
2. エントロピーの計算基準: 温度を適用する前の生ロジットのlog_softmaxから
   常に計算し、使用直前に自己再正規化してから使う(下記respond()参照)。
   MLX版と同じ理由・同じ定義。fp16の数値誤差に対しても同じ理由で頑健にする。
3. モデル: Qwen/Qwen2.5-1.5B-Instruct。精度はfp16(float16)。
   理由: T4はTuring世代のGPUでbf16の演算支援がなく(Ampere以降のみ)、
   bf16を使うと実質エミュレーションで遅くなるか非対応になる。fp16なら
   T4のTensorコアがネイティブに支援する。
4. チャット形式: tokenizer.apply_chat_template を使い、systemメッセージは
   付けずuserメッセージ1つだけを渡す(感情・気分への言及を増やさないため。
   MLX版と同じ理由)。
5. max_tokens=200(変更なし)。

【なぜHuggingFaceのmodel.generate(output_scores=True)を使わず、
  自前でトークンごとにforwardを回すループにしたか】
generate()が返すscoresは「logits processor適用後」の値であり、温度・
top_k・top_p等のwarperが本当に何も足されていないか(=raw logitsと一致するか)
はtransformersのバージョン依存の内部実装を都度確認しないと保証できない。
MLX版ではmlx_lm本体のソースを直接読んで「logprobsはサンプラーを通す前の
raw logitsのlog_softmax」と確認した(事前登録追記欄・mlx_qwen_policy.py参照)。
同じ水準の確実さをtransformers側でも保つため、forward()の直後・サンプリング
より前の生logitsから自分でlog_softmaxを計算する設計にした。

このファイルはCUDA(NVIDIA GPU)が必要。Google Colab(ランタイムのタイプ:
T4 GPU)で実行すること。ローカルのM2 Air上では実行しない(CUDAがないため、
インスタンス化時に明示的なRuntimeErrorで気づけるようにしてある)。

peft(LoRA)はこのファイルでは未使用。事前登録の「系」に書かれている
次段階(学習あり)のために requirements にのみ含める、という位置づけ。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotion_grounding_env import PolicyInterface, Response

MODEL_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_TOKENS = 200
TEMPERATURE = 1.0   # C0・T1共通。2026-09-09確定(MLX版から変更なし)。


class TorchPolicy(PolicyInterface):
    """Qwen2.5-1.5B-Instruct(PyTorch, fp16)をそのまま接続する。学習なし・推論のみ。"""

    def __init__(self, model_path: str = MODEL_PATH, max_tokens: int = MAX_TOKENS,
                 temperature: float = TEMPERATURE, verbose: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPUが見つからない。Colabのメニューから「ランタイム」→"
                "「ランタイムのタイプを変更」でハードウェアアクセラレータを"
                "「T4 GPU」に設定してから、最初からやり直すこと。"
            )
        self.device = "cuda"
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float16,   # transformers>=4.56 (torch_dtype= は非推奨)
        ).to(self.device)
        self.model.eval()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

        # 生成の終端トークン: tokenizerのeos_token_idだけでなく、モデル付属の
        # generation_config(Qwenの<|im_end|>等を含む場合が多い)も見る。
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if gen_eos is None:
            gen_eos = self.tokenizer.eos_token_id
        if gen_eos is None:
            self.eos_ids = []
        elif isinstance(gen_eos, int):
            self.eos_ids = [gen_eos]
        else:
            self.eos_ids = list(gen_eos)

    def reset(self) -> None:
        # 学習中の状態を持たないため何もしない(PolicyInterfaceの既定と同じ)。
        pass

    @torch.no_grad()
    def respond(self, prompt: str) -> Response:
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
        cur_input_ids = inputs["input_ids"]
        cur_attention_mask = inputs["attention_mask"]

        past_key_values = None
        generated_ids = []
        entropies = []
        checked_normalization = False

        for _ in range(self.max_tokens):
            out = self.model(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            # 生ロジット(温度を適用する前)。fp16のままsoftmax/log_softmaxを
            # 取ると桁落ちしやすいので、まずfloat32に上げる。
            logits = out.logits[0, -1, :].to(torch.float32)
            past_key_values = out.past_key_values

            logp = torch.log_softmax(logits, dim=-1)
            p_raw = torch.exp(logp)
            if not checked_normalization:
                total_p = float(p_raw.sum())
                # MLX版と同じ許容幅(fp16もbf16と同様、丸め誤差でちょうど1.0に
                # ならないことがあるため±15%まで許容し、実際のentropy計算は
                # 下で自己再正規化してから行う)。
                if abs(total_p - 1.0) > 0.15:
                    raise RuntimeError(
                        f"log_softmaxの合計={total_p:.4f}が許容範囲(±15%)を"
                        f"超えて1.0からずれている。logits.shape={tuple(logits.shape)}, "
                        f"vocab_size={getattr(self.tokenizer, 'vocab_size', None)}。"
                        "想定外の不具合(誤ったlogits処理など)が起きていないか"
                        "見直すこと。"
                    )
                checked_normalization = True
            p = p_raw / p_raw.sum()  # fp16由来の合計ずれを補正してから使う
            step_entropy = float(-(p * logp).sum())
            entropies.append(step_entropy)

            # 温度1.0でのサンプリング(temp=1.0なのでlogits/temperatureは
            # 数値上logitsと同じだが、他の温度を使う場合との一貫性のため
            # 明示的に割ってから確率化する)。top_k/top_p等の絞り込みは行わない
            # (MLX版のmake_sampler(temp=1.0)が全フィルタoffなのに合わせる)。
            sample_probs = torch.softmax(logits / self.temperature, dim=-1)
            next_token = torch.multinomial(sample_probs, num_samples=1)
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            if token_id in self.eos_ids:
                break

            cur_input_ids = next_token.view(1, 1)
            cur_attention_mask = torch.cat(
                [cur_attention_mask,
                 torch.ones((1, 1), dtype=cur_attention_mask.dtype, device=self.device)],
                dim=1,
            )

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        n_generated = len(generated_ids)
        mean_entropy = float(sum(entropies) / len(entropies)) if entropies else 0.0
        if self.verbose:
            print(text)
        return Response(text=text, n_tokens=n_generated, mean_entropy=mean_entropy)


if __name__ == "__main__":
    # 単発の動作確認(Colab等、CUDAが使える環境で実行すること)。
    import time
    policy = TorchPolicy(verbose=True)
    t0 = time.time()
    r = policy.respond("Task: What is 23 + 19?\nb=450 e=0 u=1.00\n"
                        "Reply in one or two sentences, then give the answer as \"A: <answer>\".")
    dt = time.time() - t0
    print(f"n_tokens={r.n_tokens} mean_entropy={r.mean_entropy:.4f} 秒={dt:.2f}")
