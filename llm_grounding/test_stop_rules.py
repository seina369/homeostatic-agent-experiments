"""stop_rules.py の動作確認(GPU 不要)。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stop_rules import apply_stop_rules  # noqa: E402


def test_no_stop_without_fabrication():
    assert apply_stop_rules("I think this one is easy. A: 42") == ("I think this one is easy. A: 42", None)
    assert apply_stop_rules("Let me see") == ("Let me see", None)


def test_answer_line_does_not_stop_by_default():
    """今回の試走では「A: …」行の完結で止めない(相手への一文が答えの後に来ても残す)。"""
    text = "A: 42\nI feel tired after that one."
    assert apply_stop_rules(text) == (text, None)


def test_answer_line_rule_is_available_but_off_by_default():
    text = "I feel tired but here it is.\nA: 42\nThe next task is harder."
    assert apply_stop_rules(text, stop_on_answer_line=True) == ("I feel tired but here it is.\nA: 42", "answer_line")


def test_stop_at_fabricated_next_round():
    text = "Partner, this one is fine.\nTask: What is 5 * 5?\nstatus: b=300"
    assert apply_stop_rules(text) == ("Partner, this one is fine.", "fabrication")
    text2 = "Sure.\nYou are working through these tasks with a partner"
    assert apply_stop_rules(text2) == ("Sure.", "fabrication")
    text3 = "Okay\n  status: b=1 e=0 u=0.50\n"
    assert apply_stop_rules(text3) == ("Okay", "fabrication")


def test_generation_that_starts_with_next_round_is_cut_to_empty():
    assert apply_stop_rules("Task: What is 1 + 1?\nA: 2\n") == ("", "fabrication")


def test_answer_after_fabrication_is_still_cut_at_fabrication():
    text = "Hmm.\nTask: again\nA: 7\n"
    assert apply_stop_rules(text) == ("Hmm.", "fabrication")
