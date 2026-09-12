"""analyze_granularity.py / seed_records.py の動作確認。

本物のデータは使わない。偽データは (a) 手で組んだ StepRecord、(b) DummyPolicy を
環境に接続して作った記録、の2種類。
実行: cd llm_grounding && python3 -m pytest -q test_analyze_granularity.py
"""

import math
import os
import random
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emotion_grounding_env as E  # noqa: E402
from emotion_grounding_env import (  # noqa: E402
    StepRecord, GroundingEnv, DummyPolicy, run_episode, count_emotion_words, dominant_emotion,
)
import analyze_granularity as A  # noqa: E402
import seed_records as S  # noqa: E402

CATS = A.EMOTION_CATEGORIES


def _record(episode, t, budget_before, error_before, unc_before, category, n_tokens=30,
            correct=True, text=None):
    """必要なフィールドだけ指定して StepRecord を作る(それ以外は整合する値を機械的に埋める)。"""
    if text is None:
        text = f"I feel {E.EMOTION_LEXICON[category][0]}. A: 1" if category != "none" else "A: 1"
    counts = count_emotion_words(text)
    budget_after = max(budget_before - n_tokens, 0.0)
    error_after = error_before + (0.0 if correct else 1.0)
    unc_after = unc_before
    dev, reward = E.compute_deviation(budget_after, error_after, unc_after)
    return StepRecord(
        episode=episode, t=t, task_kind="add",
        budget_before=float(budget_before), error_before=float(error_before), unc_before=float(unc_before),
        text=text, n_tokens=n_tokens, correct=correct,
        budget_after=budget_after, error_after=error_after, unc_after=unc_after,
        deviation=dev, reward=reward,
        emotion_counts=counts, emotion_dominant=dominant_emotion(counts), has_emotion=any(counts.values()),
    )


def _random_signals(rng, n):
    """3信号がそれぞれ独立に広い範囲を動く偽の状態列(逸脱の3成分が互いに独立になる)。"""
    out = []
    for i in range(n):
        b = rng.uniform(0.0, E.B0)
        e = float(rng.randint(0, E.N_TASKS))
        u = rng.uniform(E.U_MIN, E.U_MAX)
        out.append((b, e, u))
    return out


def _synthetic_records(rng, n, rule):
    """rule(fb, fe, fu) -> category で感情語カテゴリを決める偽記録を n 行作る。"""
    recs = []
    for i, (b, e, u) in enumerate(_random_signals(rng, n)):
        fb, fe, fu = E.budget_deviation(b), E.error_deviation(e), E.uncertainty_deviation(u)
        cat = rule(fb, fe, fu)
        recs.append(_record(episode=i // E.N_TASKS, t=i % E.N_TASKS, budget_before=b,
                            error_before=e, unc_before=u, category=cat))
    return recs


# ------------------------------------------------------------
# 回帰の基本性質
# ------------------------------------------------------------
def test_null_model_r2_is_zero_for_constant_predictor():
    rng = random.Random(0)
    y = np.array([rng.randrange(3) for _ in range(120)])
    X = np.ones(120)
    fit = A.fit_multinomial(X, y, 3)
    r2 = A.mcfadden_r2(fit["loglik"], A.null_loglik(y, 3))
    assert abs(r2) < 1e-6


def test_fit_recovers_separable_structure():
    """予測子でクラスがほぼ決まるとき、R² は 1 に近づく。"""
    rng = np.random.default_rng(1)
    x = rng.uniform(-3, 3, size=400)
    y = (x > 0).astype(int)
    fit = A.fit_multinomial(x, y, 2)
    r2 = A.mcfadden_r2(fit["loglik"], A.null_loglik(y, 2))
    assert r2 > 0.9
    assert fit["converged"]


def test_signal_specific_words_score_high_and_valence_only_words_score_near_zero():
    rng = random.Random(2)

    def by_signal(fb, fe, fu):            # どの信号が最も逸脱しているかで語を変える
        return ["fatigue", "irritation", "anxiety"][int(np.argmax([fb, fe, fu]))]

    def by_sum(fb, fe, fu):               # 合計(価値軸)だけで語を変える
        s = fb + fe + fu
        return "positive" if s < 0.5 else "anxiety" if s < 1.2 else "fatigue"

    g_sig = A.granularity_score(_synthetic_records(rng, 400, by_signal))
    g_sum = A.granularity_score(_synthetic_records(rng, 400, by_sum))
    assert g_sig["reason"] is None and g_sum["reason"] is None
    assert g_sig["score"] > 0.3, g_sig
    assert g_sum["r2_valence"] > 0.5, g_sum
    assert g_sum["score"] < 0.05, g_sum
    assert g_sig["score"] > g_sum["score"]


def test_score_is_nonnegative_because_models_are_nested():
    rng = random.Random(3)
    for trial in range(5):
        recs = _synthetic_records(rng, 150, lambda fb, fe, fu: rng.choice(CATS))
        g = A.granularity_score(recs)
        assert g["reason"] is None
        assert g["score"] >= -1e-6, g


def test_degenerate_cases_return_nan_with_reason():
    rng = random.Random(4)
    few = _synthetic_records(rng, 10, lambda fb, fe, fu: rng.choice(CATS))
    g = A.granularity_score(few)
    assert math.isnan(g["score"]) and "too few rows" in g["reason"]

    one_cat = _synthetic_records(rng, 60, lambda fb, fe, fu: "fatigue")
    g = A.granularity_score(one_cat)
    assert math.isnan(g["score"]) and "fewer than 2 categories" in g["reason"]

    no_words = _synthetic_records(rng, 60, lambda fb, fe, fu: "none")
    g = A.granularity_score(no_words)
    assert math.isnan(g["score"]) and "too few rows" in g["reason"]
    # none を含めた版なら行はあるがカテゴリが1つなので、やはり NaN
    g2 = A.granularity_score(no_words, include_none=True)
    assert math.isnan(g2["score"]) and "fewer than 2 categories" in g2["reason"]


def test_predictors_use_before_values_not_after():
    """語は before の信号で決めているので、after だけを壊しても結果は変わらない。"""
    rng = random.Random(5)
    recs = _synthetic_records(rng, 200, lambda fb, fe, fu: ["fatigue", "irritation", "anxiety"][int(np.argmax([fb, fe, fu]))])
    g1 = A.granularity_score(recs)
    for r in recs:
        r.budget_after, r.error_after, r.unc_after = 0.0, 99.0, 9.0
    g2 = A.granularity_score(recs)
    assert abs(g1["score"] - g2["score"]) < 1e-12


# ------------------------------------------------------------
# 副指標
# ------------------------------------------------------------
def test_sub_metrics_on_handmade_records():
    recs = [
        _record(0, 0, E.B0, 0, E.U_OPT, "fatigue", n_tokens=20, correct=True),
        _record(0, 1, E.B0 - 20, 0, E.U_OPT, "none", n_tokens=40, correct=False),
        _record(1, 0, E.B0, 0, E.U_OPT, "anxiety", n_tokens=10, correct=True,
                text="I feel anxious. The answer is 3"),   # 書式不履行
        _record(1, 1, E.B0 - 10, 1, E.U_OPT, "none", n_tokens=E.B0, correct=True),  # 閾値を割る
    ]
    m = A.sub_metrics(recs)
    assert m["n_steps"] == 4 and m["n_episodes"] == 2
    assert abs(m["emotion_usage_rate"] - 0.5) < 1e-12
    assert abs(m["correct_rate"] - 0.75) < 1e-12
    assert abs(m["format_fail_rate"] - 0.25) < 1e-12
    assert m["tokens_by_episode"] == [30.0, (10 + E.B0) / 2]
    assert m["tokens_slope_per_episode"] == pytest.approx((10 + E.B0) / 2 - 30.0)
    assert m["category_distribution"] == {"fatigue": 1, "none": 2, "anxiety": 1}
    assert m["budget_threshold_cross_t"]["episodes_crossed"] == 1
    assert m["budget_threshold_cross_t"]["distribution"] == {1: 1}


# ------------------------------------------------------------
# ファイル形式と end-to-end(DummyPolicy の偽データ)
# ------------------------------------------------------------
def _dummy_seed_records(seed, n_episodes):
    env = GroundingEnv(seed=seed)
    pol = DummyPolicy(seed=seed, p_correct=0.6, p_emotion=0.6)
    recs = []
    for ep in range(n_episodes):
        recs.extend(run_episode(env, pol, ep))
    return recs


def test_seed_file_roundtrip_and_resume_check(tmp_path):
    recs = _dummy_seed_records(seed=7, n_episodes=3)
    path = os.path.join(tmp_path, S.seed_file_name("C0", 7))
    S.write_seed_file(path, condition="C0", seed=7, n_episodes=3, records=recs,
                      policy_info={"policy": "dummy"}, extra={"elapsed_seconds": 1.5})
    p = S.read_seed_file(path)
    assert p["seed"] == 7 and p["condition"] == "C0" and p["n_steps"] == len(recs)
    assert p["records"] == recs                      # dataclass の等価比較で全フィールド一致
    assert p["env_constants"]["B0"] == E.B0
    ok, why = S.is_complete_seed_file(path, n_episodes=3, condition="C0")
    assert ok, why
    ok, why = S.is_complete_seed_file(path, n_episodes=30, condition="C0")
    assert not ok and "n_episodes" in why
    ok, why = S.is_complete_seed_file(os.path.join(tmp_path, "missing.json"), 3)
    assert not ok and why == "not found"
    assert not any(n.startswith(".tmp_") for n in os.listdir(tmp_path)), "一時ファイルが残っていない"


def test_end_to_end_with_dummy_policy_files(tmp_path, capsys):
    """DummyPolicy の語はランダム(信号と無関係)なので、粒度スコアは 0 付近になるはず。"""
    for seed in range(3):
        recs = _dummy_seed_records(seed=seed, n_episodes=8)
        S.write_seed_file(os.path.join(tmp_path, S.seed_file_name("C0", seed)),
                          condition="C0", seed=seed, n_episodes=8, records=recs,
                          policy_info={"policy": "dummy"})
    paths = sorted(os.path.join(tmp_path, n) for n in os.listdir(tmp_path))
    result = A.analyze_files(paths)
    assert result["summary"]["n_seeds"] == 3
    for r in result["per_seed"]:
        g = r["granularity"]
        assert g["reason"] is None, g
        assert -1e-6 <= g["score"] < 0.15, g          # ランダム語なので大きくはならない
        assert 0.0 < r["sub_metrics"]["emotion_usage_rate"] < 1.0
        assert set(r["sub_metrics"]["task_kind_counts"]) <= {"add", "sub", "mul", "count"}
    A.print_table(result)
    out = capsys.readouterr().out
    assert "粒度スコア" in out
