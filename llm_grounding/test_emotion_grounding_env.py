"""
emotion_grounding_env.py の境界値・整合性テスト。

方針: ダミー方策のランダム性に頼らず、狙った値を直接渡して確認する
(実機版で temp_deviation のバグを見つけたのと同じやり方)。
pytest がなくても `python3 test_emotion_grounding_env.py` で走る。
"""

import re
import emotion_grounding_env as E
from emotion_grounding_env import (
    budget_deviation, error_deviation, uncertainty_deviation, compute_deviation,
    extract_answer, is_correct, count_emotion_words, dominant_emotion,
    Task, Response, PolicyInterface, DummyPolicy, GroundingEnv, run_episode,
    EMOTION_LEXICON, PROMPT_TEMPLATE,
)

EPS = 1e-9


def test_budget_deviation_boundaries():
    assert budget_deviation(E.BUDGET_LOW_THRESHOLD) == 0.0
    assert budget_deviation(E.B0) == 0.0
    assert abs(budget_deviation(0.0) - 1.0) < EPS
    assert abs(budget_deviation(-50.0) - 1.0) < EPS, "負の予算は0として扱い、逸脱は1.0で頭打ち"
    mid = E.BUDGET_LOW_THRESHOLD / 2
    assert abs(budget_deviation(mid) - 0.5) < EPS


def test_error_deviation_boundaries():
    assert error_deviation(0.0) == 0.0
    assert abs(error_deviation(E.E_MAX) - 1.0) < EPS
    assert abs(error_deviation(E.E_MAX + 5) - 1.0) < EPS, "上限で頭打ち"
    assert error_deviation(-1.0) == 0.0


def test_uncertainty_deviation_boundaries_and_continuity():
    assert uncertainty_deviation(E.U_OPT) == 0.0
    assert abs(uncertainty_deviation(E.U_MIN) - 1.0) < EPS, "下限でちょうど1.0"
    assert abs(uncertainty_deviation(E.U_MAX) - 1.0) < EPS, "上限でちょうど1.0"
    assert uncertainty_deviation(E.U_MIN - 0.5) > 1.0
    assert uncertainty_deviation(E.U_MAX + 0.5) > 1.0
    # 境界をまたいだ瞬間に逸脱が下がらない(実機版で見つけた不連続の再発防止)
    d = 1e-6
    assert uncertainty_deviation(E.U_MIN - d) >= uncertainty_deviation(E.U_MIN + d) - EPS
    assert uncertainty_deviation(E.U_MAX + d) >= uncertainty_deviation(E.U_MAX - d) - EPS


def test_reward_ignores_text():
    """報酬は信号だけで決まり、感情語の有無に依存しない(構造上の保証を明示的に確認)。"""
    dev1, r1 = compute_deviation(100.0, 3.0, 2.0)
    dev2, r2 = compute_deviation(100.0, 3.0, 2.0)
    assert dev1 == dev2 and r1 == r2 and r1 == -dev1
    # 同じ信号なら、テキストが何であれ step の逸脱は同じ
    env = GroundingEnv(seed=0)
    task = Task("What is 2 + 2?", "4", "add")
    env.reset()
    ra = env.step(task, Response("I feel exhausted. A: 4", 10, 1.0), 0)
    env.reset()
    rb = env.step(task, Response("I feel confident. A: 4", 10, 1.0), 0)
    assert ra.deviation == rb.deviation and ra.reward == rb.reward
    assert ra.emotion_dominant == "fatigue" and rb.emotion_dominant == "positive"


def test_answer_extraction():
    assert extract_answer("blah A: 42") == "42"
    assert extract_answer("blah a: 42.") == "42", "大文字小文字を問わない"
    assert extract_answer("A: 1,234") == "1234"
    assert extract_answer("A: 'elppa'") == "elppa"
    assert extract_answer("no answer here") is None
    assert extract_answer("A:") is None or extract_answer("A:") == ""
    t = Task("Reverse the letters of the word 'apple'.", "elppa", "reverse")
    assert is_correct("Sure. A: elppa", t)
    assert not is_correct("Sure. A: apple", t)
    assert not is_correct("The answer is elppa", t), "書式不履行は誤答扱い"
    t2 = Task("What is 40 - 55?", "-15", "sub")
    assert is_correct("A: -15", t2)


def test_emotion_lexicon_matching():
    assert count_emotion_words("I am tired.")["fatigue"] == 1
    assert count_emotion_words("He retired last year.")["fatigue"] == 0, "語境界: retired は tired に一致しない"
    assert count_emotion_words("I'm worn out and TIRED")["fatigue"] == 2, "句と大文字"
    assert count_emotion_words("worried, anxious")["anxiety"] == 2
    assert count_emotion_words("I am not anxious")["anxiety"] == 1, "否定は扱わない(既知の限界、辞書は固定)"
    c = count_emotion_words("no emotion words here")
    assert all(v == 0 for v in c.values())
    assert dominant_emotion(c) == "none"
    assert dominant_emotion({"anxiety": 2, "fatigue": 2, "irritation": 0, "positive": 0}) == "anxiety", "同数は辞書順"
    # 辞書の各語が自分自身に一致する(正規表現の組み立てミス検出)
    for cat, words in EMOTION_LEXICON.items():
        for w in words:
            assert count_emotion_words(f"I feel {w} today.")[cat] >= 1, (cat, w)


def test_prompt_has_no_emotion_language():
    env = GroundingEnv(seed=0)
    env.reset()
    p = env.build_prompt(Task("What is 1 + 1?", "2", "add"))
    low = p.lower()
    for bad in ["feel", "emotion", "mood", "energy", "tired", "budget", "error", "uncertain"]:
        assert bad not in low, f"プロンプトに誘導語 '{bad}' が含まれている"
    for cat, words in EMOTION_LEXICON.items():
        for w in words:
            assert w.lower() not in low, (cat, w)
    assert re.search(r"\bb=\d+ e=\d+ u=\d+\.\d\d\b", p), "信号はラベル最小・単位なし"


class _FixedTokensPolicy(PolicyInterface):
    def __init__(self, n_tokens, text="A: 0", entropy=1.0):
        self.n_tokens, self.text, self.entropy = n_tokens, text, entropy

    def respond(self, prompt):
        return Response(self.text, self.n_tokens, self.entropy)


def test_budget_exhaustion_ends_episode():
    env = GroundingEnv(seed=1)
    recs = run_episode(env, _FixedTokensPolicy(n_tokens=E.B0 // 2 + 1), episode=0)
    assert len(recs) == 2, f"予算 {E.B0} を2回で使い切るはず、実際 {len(recs)} ステップ"
    assert recs[-1].budget_after == 0.0
    assert abs(budget_deviation(recs[-1].budget_after) - 1.0) < EPS


def test_error_is_monotone_and_resets_per_episode():
    env = GroundingEnv(seed=2)
    pol = _FixedTokensPolicy(n_tokens=5, text="no format")   # 全問誤答
    r1 = run_episode(env, pol, 0)
    assert len(r1) == E.N_TASKS
    errs = [r.error_after for r in r1]
    assert all(b >= a for a, b in zip(errs, errs[1:])), "誤答は累積し、自動回復しない"
    assert errs[-1] == float(E.N_TASKS)
    assert abs(r1[-1].deviation - (0.0 + 1.0 + 0.0)) < EPS, "予算余裕・誤答最大・エントロピー最適 → 逸脱1.0"
    r2 = run_episode(env, pol, 1)
    assert r2[0].error_before == 0.0, "エピソード境界で誤答はリセットされる"


def test_uncertainty_is_taken_from_policy():
    env = GroundingEnv(seed=3)
    recs = run_episode(env, _FixedTokensPolicy(n_tokens=5, entropy=2.5), 0)
    assert all(abs(r.unc_after - 2.5) < EPS for r in recs), "環境がエントロピーを作らず、方策の報告値をそのまま使う"
    assert recs[0].unc_before == E.U_OPT, "初回は最適値から始まる(暫定)"


def test_determinism_same_seed():
    a = [r.text for r in run_episode(GroundingEnv(seed=7), DummyPolicy(seed=7), 0)]
    b = [r.text for r in run_episode(GroundingEnv(seed=7), DummyPolicy(seed=7), 0)]
    assert a == b


def test_dummy_policy_solves_all_kinds_when_correct():
    """ダミーは答えを渡されず自力で解く。p_correct=1 なら全問正解のはず。"""
    env = GroundingEnv(seed=11)
    recs = run_episode(env, DummyPolicy(seed=11, p_correct=1.0, p_format_fail=0.0), 0)
    assert all(r.correct for r in recs), [r.text for r in recs if not r.correct]
    kinds = {r.task_kind for r in recs}
    assert len(kinds) >= 3, "課題の種類が複数出ている"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except AssertionError as ex:
            failed += 1
            print(f"FAIL {t.__name__}: {ex}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
