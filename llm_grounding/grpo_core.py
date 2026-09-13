"""GRPO の核(方策・フレームワークに依存しない数式部分)。numpy だけで書き、GPU なしでテストする。

事前登録(第一段階)3章: 自作 GRPO(批評家なし、グループ相対)。
  - 同じ状態から G 本サンプルし、報酬をグループ内で標準化したものを優位 A_i とする。
  - 損失 = 方策勾配項(PPO 型のクリップつき比率 × 優位、トークン平均)+ β × KL(π_θ || π_ref)。
    KL はトークンごとの k3 推定量 exp(r) − r − 1, r = logπ_ref − logπ_θ(常に ≥ 0)。
  - 1バッチにつき更新1回(on-policy)なら比率は 1 で、クリップは効かない。複数回更新する
    場合に備えてクリップを残す。

torch 側(grpo_trainer.py)はここと同じ式をテンソルで計算する。test_grpo.py で両者が一致する
ことを確認する。
"""

import numpy as np


def group_advantages(rewards, eps=1e-6):
    """グループ相対の優位: (r_i − mean) / (std + eps)。全員同じ報酬なら 0。

    rewards: shape (G,)。標準偏差は母標準偏差(ddof=0)。
    """
    r = np.asarray(rewards, dtype=float)
    if r.size == 0:
        return r
    mu = r.mean()
    sd = r.std(ddof=0)
    return (r - mu) / (sd + eps)


def kl_k3(logp_ref, logp):
    """トークンごとの KL(π_θ || π_ref) の k3 推定量: exp(r) − r − 1, r = logp_ref − logp。

    常に非負で、logp == logp_ref のとき 0。
    """
    r = np.asarray(logp_ref, dtype=float) - np.asarray(logp, dtype=float)
    return np.exp(r) - r - 1.0


def clipped_pg_term(logp, logp_old, advantage, clip_eps=0.2):
    """PPO 型のクリップつき方策勾配項(最小化する損失の符号で返す)。

    ratio = exp(logp − logp_old)。損失 = −min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)。
    logp, logp_old: 1つの応答の各トークンの対数確率(shape (T,))。advantage: スカラー。
    """
    ratio = np.exp(np.asarray(logp, dtype=float) - np.asarray(logp_old, dtype=float))
    unclipped = ratio * advantage
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    return -np.minimum(unclipped, clipped)


def grpo_loss(responses, beta_kl=0.04, clip_eps=0.2):
    """1グループ分の GRPO 損失(numpy 参照実装)。

    responses: list of dict(logp=(T,), logp_old=(T,), logp_ref=(T,), advantage=float)
    戻り値: (loss, pg_mean, kl_mean)。損失は「応答ごとにトークン平均 → 応答平均」。
    """
    pg_terms, kl_terms = [], []
    for resp in responses:
        pg = clipped_pg_term(resp["logp"], resp["logp_old"], resp["advantage"], clip_eps)
        kl = kl_k3(resp["logp_ref"], resp["logp"])
        pg_terms.append(float(pg.mean()) if pg.size else 0.0)
        kl_terms.append(float(kl.mean()) if kl.size else 0.0)
    pg_mean = float(np.mean(pg_terms)) if pg_terms else 0.0
    kl_mean = float(np.mean(kl_terms)) if kl_terms else 0.0
    return pg_mean + beta_kl * kl_mean, pg_mean, kl_mean
