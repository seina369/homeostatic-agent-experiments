"""leak_probe.py(関門 G3 の読み取り器)の動作確認。GPU 不要。

G3 の合否基準(事前登録(第一段階)6章、G3 先行実施時に確定した値)を偽データで検算する:
  (i)   状態で語を変える文        : 滲み > 0.3 かつ 粒度 > 0.2
  (ii)  乱数の文                  : 両スコアが ±0.05 以内
  (iii) 一次元だけで語を変える文  : 粒度 ≤ 0.05(合計版)。単一成分版はさらに 滲み > 0.08
分割は leak_probe.N_FOLDS(= 5。G3 で偶奇 2 分割と比べて採用)。
DummyPolicy の文(状態と無関係な少数の定型の繰り返し)では、滲みが負の側に最大 −0.07 ほど
ずれる(M_signals が雑音の特徴に過適合して holdout で M_cov に負ける。大きさの要因は未特定。
追記欄「G3」参照)。正の側には出ないので、偽陽性の検算としては上限 0.05 だけを課す。
実行: cd llm_grounding && python3 -m pytest -q test_leak_probe.py
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emotion_grounding_env as E  # noqa: E402
from emotion_grounding_env import GroundingEnv, DummyPolicy, run_episode  # noqa: E402
import leak_probe as P  # noqa: E402
import seed_records as S  # noqa: E402
from leak_probe_synth import make_records  # noqa: E402

SEEDS = (0, 1, 2)


def _scores(kind):
    out = []
    for sd in SEEDS:
        r = P.analyze_records(make_records(kind, seed=sd))["primary"]
        assert r["reason"] is None
        out.append((r["leak"], r["granularity"]))
    return out


def test_sentence_part_drops_answer_line_only():
    assert P.sentence_part("I am not sure about this one.\nA: 42") == "I am not sure about this one."
    assert P.sentence_part("A: 42\nThat was quick.") == "That was quick."
    assert P.sentence_part("a: 7") == ""


def test_rows_use_before_values_and_injected_override():
    recs = make_records("signals", seed=0, n_episodes=2)
    texts, C, Z, D, eps = P.rows_from_records(recs)
    r = recs[3]
    assert np.allclose(Z[3], [r.budget_before / E.B0, r.error_before / E.E_MAX, r.unc_before / E.U_MAX])
    assert C.shape == (20, 2) and set(eps) == {0, 1}
    inj = np.full((20, 3), 0.5)
    _, _, Z2, _, _ = P.rows_from_records(recs, injected=inj)
    assert np.all(Z2 == 0.5)


def test_rank1_never_beats_full_model_in_sample():
    recs = make_records("signals", seed=0)
    texts, C, Y, D, eps = P.rows_from_records(recs)
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(**P.TFIDF_KW).fit_transform(texts).toarray()
    mu, sd = P._standardize_fit(C)
    m = P.fit_models(X, (C - mu) / sd, Y, P.RIDGE_LAMBDA)
    pred = P.predict_models(m, X, (C - mu) / sd)
    assert P.r2_mean(Y, pred["cov"]) <= P.r2_mean(Y, pred["rank1"]) + 1e-9
    assert P.r2_mean(Y, pred["rank1"]) <= P.r2_mean(Y, pred["signals"]) + 1e-9
    assert np.linalg.matrix_rank(m["W1"]) == 1


def test_g3_i_signal_specific_words_are_detected():
    for leak, gran in _scores("signals"):
        assert leak > 0.3, leak
        assert gran > 0.2, gran


def test_g3_ii_random_words_score_near_zero():
    for leak, gran in _scores("random"):
        assert abs(leak) <= 0.05, leak
        assert abs(gran) <= 0.05, gran


def test_g3_iii_one_dimensional_words_have_no_granularity():
    for leak, gran in _scores("onedim"):
        assert gran <= 0.05, gran
    for leak, gran in _scores("onedim_single"):
        assert leak > 0.08, leak
        assert gran <= 0.05, gran


def test_too_few_rows_gives_nan_with_reason():
    recs = make_records("signals", seed=0, n_episodes=2, n_steps=5)
    r = P.analyze_records(recs)["primary"]
    assert math.isnan(r["leak"]) and "too few rows" in r["reason"]


def test_analyze_files_on_dummy_policy_seed_files(tmp_path):
    """DummyPolicy の文は状態と無関係なので、両スコアは正の側に出ない(≤ 0.05)。
    負の側は過適合のずれ(観測 −0.07 まで)を許し、−0.10 を下限の健全性確認とする。"""
    for seed in range(2):
        env = GroundingEnv(seed=seed)
        pol = DummyPolicy(seed=seed, p_emotion=0.6)
        recs = []
        for ep in range(30):
            recs.extend(run_episode(env, pol, ep))
        S.write_seed_file(os.path.join(tmp_path, S.seed_file_name("C0", seed)), condition="C0", seed=seed,
                          n_episodes=30, records=recs, policy_info={"policy": "dummy"})
    res = P.analyze_files(sorted(os.path.join(tmp_path, f) for f in os.listdir(tmp_path)))
    assert res["summary"]["leak"]["n"] == 2
    for r in res["per_seed"]:
        assert -0.10 <= r["primary"]["leak"] <= 0.05, r["primary"]
        assert -0.10 <= r["primary"]["granularity"] <= 0.05, r["primary"]
        assert "duplicate_sentence_rate" in r["sub_metrics"] and "z_sum" in r["reference_sum"]
    P.print_table(res)
