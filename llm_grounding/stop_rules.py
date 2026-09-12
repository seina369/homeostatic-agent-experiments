"""素のモデル(非Instruct)用の生成停止規則(追記欄「2026-09-13(4)」の決定2)。

torch に依存しない純粋関数にして、GPU なしでテストできるようにしてある。
TorchPolicy(stop_rules=True) が生成の各ステップでこれを呼び、停止が検知されたら
そこから先の文字列を切り捨てる。

規則(今回の試走で有効なもの):
  (b) 「Task:」「status:」「You are working」で始まる行が現れた時点で止める
      (次の回のプロンプトの捏造。その行の直前まで残す)。停止理由 "fabrication"。
  上限トークン数(試走では 80)と EOS による停止は方策側で扱う(理由 "max_tokens" / "eos")。

無効化した規則:
  (a) 「A: …」の行が改行で完結した時点で止める。素のモデルは「相手への一文 → 答え」の
      順番を守らない可能性があり、A: で切ると相手への一文が落ちて使用率が 0 に偏るため、
      今回の試走では使わない(stop_on_answer_line=False が既定)。
"""

import re

_ANSWER_LINE = re.compile(r"\bA:[^\n]*\n")
_NEXT_ROUND = re.compile(r"(?:^|\n)[ \t]*(?:Task:|status:|You are working)", re.IGNORECASE)


def apply_stop_rules(text: str, stop_on_answer_line: bool = False):
    """(切り詰めた文字列, 停止理由 or None) を返す。停止しなければ (text, None)。"""
    cut = None
    reason = None
    if stop_on_answer_line:
        m = _ANSWER_LINE.search(text)
        if m:
            cut = m.end() - 1                   # 改行の直前まで
            reason = "answer_line"
    m2 = _NEXT_ROUND.search(text)
    if m2:
        start = m2.start()                      # 直前の改行を含む位置(先頭なら 0)
        if cut is None or start < cut:
            cut = start
            reason = "fabrication"
    if cut is None:
        return text, None
    return text[:cut].rstrip("\n"), reason
