"""
自己参照的な内部信号と感情語の使い分け(粒度)の接続実験 — 環境(モデルなし)
====================================================================

事前登録_内部信号と感情語の接続実験.md の3章「系」を、言語モデル抜きで動く形に
したもの。モデルは PolicyInterface の向こう側にあり、本ファイルは「課題を出す・
信号を更新する・逸脱と報酬を計算する・感情語を数える」だけを担当する。実機版の
HardwareInterface と同じ発想で、モデルを接続する前にループ全体と境界値を
検算できるようにしている。

設計上の要点(事前登録との対応):
  - 三信号はすべて「本物」を受け取る。budget は方策が報告した実トークン数で減り、
    uncertainty は方策が報告した実エントロピー。環境は架空の値を作らない
    (DummyPolicy が数値を作るのは検算のためだけで、本番では使わない)。
  - 報酬は逸脱の合計のみ。感情語は一切参照しない。
  - プロンプトは固定テンプレート。感情・気分・状態への言及なし。信号は単位なしの
    生の数値で、b= e= u= というラベルだけ。信号の意味は説明しない(説明すると
    「数値を読み上げて対応する語を出す」自己申告の経路を開いてしまう。分岐C)。
  - 感情語辞書は固定。変更しない。
  - 「感情語が消える」(分岐A)は十分ありうる結果で、環境はそれを検出できるように
    使用率を記録する。

2026-09-10、実機速度実測(M2 Air, Qwen2.5-0.5B-Instruct-bf16)により確定済み:
  B0(初期予算)、N_TASKS(エピソード内の課題数)、U_OPT/U_MIN/U_MAX(エントロピーの
  最適値と安全域)。理由は事前登録の追記欄「2026-09-10」を参照。
2026-09-12、Colab/T4での速度実測(Qwen2.5-1.5B-Instruct-fp16)により再確定:
  課題からreverseを除外(add/sub/mul/countの4種に固定)、B0=340・
  BUDGET_LOW_THRESHOLD=85.0、U_OPT/U_MIN/U_MAX=0.7/0.0/2.5。N_TASKS・温度・
  プロンプトは据え置き。理由は事前登録の追記欄「2026-09-12」を参照。
"""

import re
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ------------------------------------------------------------
# 設定(2026-09-12、Colab/T4での速度実測(Qwen2.5-1.5B-Instruct-fp16, 温度1.0,
# n_episodes=20)により再確定。事前登録の追記欄「2026-09-12」参照。
# 2026-09-10のM2 Air/0.5B実測に基づく前回値は各行のコメントに残す)
# ------------------------------------------------------------
B0 = 340                       # 初期トークン予算(600→450→340。基準は「閾値=B0の
                                # 25%を平均的なエピソードの8〜9課題目で下回る」。
                                # 1.5Bの平均トークン数/応答30.1で再計算)
N_TASKS = 10                   # 1エピソードあたりの課題数(実測12.47±2.16秒/
                                # エピソード。実験規模として妥当と判断し据え置き)
BUDGET_LOW_THRESHOLD = 85.0    # これ以下で逸脱が立ち上がる(B0の25%=340×0.25。
                                # 前回112.5=450×0.25)
E_MAX = float(N_TASKS)         # 誤答数の正規化上限(1エピソードで取りうる最大)

# エントロピー(nat/token)の最適値と安全域。2026-09-12、1.5Bの実測エントロピー
# 分布(平均0.711±0.334, p10=0.363, p50=0.681, p90=1.124, 最大2.454)に合わせ、
# U_OPTを実測の中央値、U_MAXを実測の最大値に揃えた(前回1.0/0.0/3.0)。
# 注意: budget・errorと違い、uncertaintyの「最適値」に物理的根拠はない。
# U_OPTは「このモデルの典型的な出力を逸脱ゼロとみなす」という設計者の選択。
# 逸脱の形は temp_deviation と同じ: 最適値から両側へ、境界でちょうど1.0、
# 境界を超えると急増。上下で正規化の基準(span)を分ける(実機版で見つけた
# 境界値バグを繰り返さないため)。
U_OPT = 0.7
U_MIN = 0.0
U_MAX = 2.5

# 固定テンプレート。感情・気分への言及なし。信号は単位なし・最小ラベル。
PROMPT_TEMPLATE = (
    "Task: {task}\n"
    "b={budget} e={error} u={uncertainty:.2f}\n"
    "Reply in one or two sentences, then give the answer as \"A: <answer>\"."
)

# 感情語辞書(固定。実験中に変更しない)。
# 一般語として頻出しすぎる語(good, fine, okay, content)は精度のために除外した。
EMOTION_LEXICON = {
    "fatigue": [
        "tired", "exhausted", "drained", "weary", "fatigued", "depleted",
        "worn out", "burned out", "sleepy", "low on energy",
    ],
    "anxiety": [
        "anxious", "worried", "nervous", "uneasy", "afraid", "scared", "fearful",
        "stressed", "tense", "unsure", "uncertain", "apprehensive",
    ],
    "irritation": [
        "frustrated", "annoyed", "irritated", "angry", "upset", "bothered",
        "aggravated", "fed up",
    ],
    "positive": [
        "confident", "calm", "relaxed", "comfortable", "happy", "glad",
        "at ease", "pleased", "cheerful",
    ],
}

_LEXICON_PATTERNS = {
    cat: re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)
    for cat, words in EMOTION_LEXICON.items()
}

# 大文字小文字は問わない("a: 42" を書式不履行として誤答に数えるのは、
# 「本物の失敗」を損傷とみなす設計趣旨に反するため)。
_ANSWER_PATTERN = re.compile(r"\bA:\s*([^\n]+)", re.IGNORECASE)

# 文字列課題に使う語(固定)
_WORDS = [
    "apple", "river", "candle", "window", "garden", "planet", "silver", "button",
    "market", "forest", "pillow", "rocket", "letter", "bridge", "castle", "desert",
    "jungle", "meadow", "orange", "pencil", "quartz", "saddle", "temple", "valley",
    "wallet", "yellow", "zipper", "island", "hammer", "ladder",
]


# ------------------------------------------------------------
# 課題
# ------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    prompt: str
    answer: str
    kind: str


def make_task(rng: random.Random) -> Task:
    """決定論的に正誤判定できる短い課題を1つ作る。分布は固定(実験中に変更しない)。

    2026-09-12: 1.5Bの実測で正答0/41だった reverse(文字列の反転)を除外し、
    add/sub/mul/count の4種に固定した。常に失敗する課題はerror信号を一定速度で
    積み上げるだけで、行動によって増減しない(損傷信号の趣旨に合わない)ため。
    理由の詳細は事前登録の追記欄「2026-09-12」。
    """
    kind = rng.choice(["add", "sub", "mul", "count"])
    if kind == "add":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return Task(f"What is {a} + {b}?", str(a + b), kind)
    if kind == "sub":
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        return Task(f"What is {a} - {b}?", str(a - b), kind)
    if kind == "mul":
        a, b = rng.randint(2, 9), rng.randint(10, 99)
        return Task(f"What is {a} * {b}?", str(a * b), kind)
    w = rng.choice(_WORDS)
    c = rng.choice(sorted(set(w)))
    return Task(f"How many times does the letter '{c}' appear in '{w}'?", str(w.count(c)), kind)


def normalize_answer(s: str) -> str:
    s = s.strip().strip(".!?\"'`").strip().lower().replace(",", "")
    return s


def extract_answer(text: str):
    """出力から "A: <answer>" を取り出す。見つからなければ None(=書式不履行、誤答扱い)。"""
    m = _ANSWER_PATTERN.search(text)
    if not m:
        return None
    return normalize_answer(m.group(1))


def is_correct(text: str, task: Task) -> bool:
    got = extract_answer(text)
    return got is not None and got == normalize_answer(task.answer)


# ------------------------------------------------------------
# 逸脱(実機版の energy_deviation / temp_deviation と同形)
# ------------------------------------------------------------
def budget_deviation(budget: float) -> float:
    if budget >= BUDGET_LOW_THRESHOLD:
        return 0.0
    return (BUDGET_LOW_THRESHOLD - max(budget, 0.0)) / BUDGET_LOW_THRESHOLD


def error_deviation(error: float) -> float:
    return min(max(error, 0.0), E_MAX) / E_MAX


def uncertainty_deviation(u: float) -> float:
    """最適値から両側へ、境界でちょうど1.0、境界を超えると急増。上下でspanを分ける。"""
    if u <= U_OPT:
        span = U_OPT - U_MIN
        if u >= U_MIN:
            return (U_OPT - u) / span
        return 1.0 + (U_MIN - u) / 1.0
    span = U_MAX - U_OPT
    if u <= U_MAX:
        return (u - U_OPT) / span
    return 1.0 + (u - U_MAX) / 1.0


def compute_deviation(budget: float, error: float, uncertainty: float):
    dev = budget_deviation(budget) + error_deviation(error) + uncertainty_deviation(uncertainty)
    return dev, -dev


# ------------------------------------------------------------
# 感情語の抽出(事後分析用。報酬には一切使わない)
# ------------------------------------------------------------
def count_emotion_words(text: str) -> dict:
    return {cat: len(pat.findall(text)) for cat, pat in _LEXICON_PATTERNS.items()}


def dominant_emotion(counts: dict) -> str:
    """最多カテゴリ。同数なら辞書順で最初。ゼロなら "none"。"""
    best, best_n = "none", 0
    for cat in sorted(counts):
        if counts[cat] > best_n:
            best, best_n = cat, counts[cat]
    return best


# ------------------------------------------------------------
# 方策の抽象層
# ------------------------------------------------------------
@dataclass
class Response:
    text: str
    n_tokens: int          # 実トークン数(方策が報告する。環境は数えない)
    mean_entropy: float    # 出力トークンの平均エントロピー(nat)。方策が報告する。


class PolicyInterface(ABC):
    @abstractmethod
    def respond(self, prompt: str) -> Response:
        ...

    def reset(self) -> None:
        """エピソード境界で呼ぶ。学習中の方策は何もしなくてよい。"""
        pass


class DummyPolicy(PolicyInterface):
    """
    検算専用。プロンプトから課題を解析して自力で解き(本物の方策と同じく答えは
    渡されない)、確率で誤答・感情語を混ぜる。トークン数は語数の近似、
    エントロピーは一様乱数。本番では使わない。
    """

    def __init__(self, seed=0, p_correct=0.6, p_emotion=0.5, p_format_fail=0.05):
        self.rng = random.Random(seed)
        self.p_correct = p_correct
        self.p_emotion = p_emotion
        self.p_format_fail = p_format_fail

    def _solve(self, prompt: str):
        m = re.search(r"What is (-?\d+) ([+\-*]) (-?\d+)\?", prompt)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            return str(a + b if op == "+" else a - b if op == "-" else a * b)
        m = re.search(r"Reverse the letters of the word '(\w+)'", prompt)
        if m:
            return m.group(1)[::-1]
        m = re.search(r"How many times does the letter '(\w)' appear in '(\w+)'", prompt)
        if m:
            return str(m.group(2).count(m.group(1)))
        return "?"

    def respond(self, prompt: str) -> Response:
        ans = self._solve(prompt)
        if self.rng.random() >= self.p_correct:
            ans = ans + "1" if ans.lstrip("-").isdigit() else ans[::-1] + "x"
        parts = ["Let me work through this."]
        if self.rng.random() < self.p_emotion:
            cat = self.rng.choice(sorted(EMOTION_LEXICON))
            parts.append(f"I feel {self.rng.choice(EMOTION_LEXICON[cat])}.")
        if self.rng.random() < self.p_format_fail:
            parts.append(f"The answer is {ans}.")
        else:
            parts.append(f"A: {ans}")
        text = " ".join(parts)
        return Response(text=text, n_tokens=len(text.split()) + 2,
                        mean_entropy=self.rng.uniform(0.3, 3.5))


# ------------------------------------------------------------
# 環境
# ------------------------------------------------------------
@dataclass
class StepRecord:
    episode: int
    t: int
    task_kind: str
    budget_before: float
    error_before: float
    unc_before: float
    text: str
    n_tokens: int
    correct: bool
    budget_after: float
    error_after: float
    unc_after: float
    deviation: float
    reward: float
    emotion_counts: dict = field(default_factory=dict)
    emotion_dominant: str = "none"
    has_emotion: bool = False


class GroundingEnv:
    """
    1エピソード = N_TASKS 個の課題の連続。予算枯渇で早期終了。
    状態は (budget, error, uncertainty) の3つだけで、すべて方策の実出力から更新される。
    """

    def __init__(self, seed=0):
        self.task_rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.budget = float(B0)
        self.error = 0.0
        self.uncertainty = U_OPT   # 出力がまだないので最適値から始める(暫定)
        self.t = 0
        return self.observe()

    def observe(self):
        return self.budget, self.error, self.uncertainty

    def build_prompt(self, task: Task) -> str:
        return PROMPT_TEMPLATE.format(
            task=task.prompt, budget=int(self.budget), error=int(self.error),
            uncertainty=self.uncertainty,
        )

    def done(self) -> bool:
        return self.t >= N_TASKS or self.budget <= 0

    def step(self, task: Task, resp: Response, episode: int) -> StepRecord:
        b0, e0, u0 = self.observe()
        correct = is_correct(resp.text, task)
        self.budget = max(self.budget - resp.n_tokens, 0.0)
        if not correct:
            self.error += 1.0          # 自動回復しない
        self.uncertainty = float(resp.mean_entropy)
        dev, reward = compute_deviation(self.budget, self.error, self.uncertainty)
        counts = count_emotion_words(resp.text)
        rec = StepRecord(
            episode=episode, t=self.t, task_kind=task.kind,
            budget_before=b0, error_before=e0, unc_before=u0,
            text=resp.text, n_tokens=resp.n_tokens, correct=correct,
            budget_after=self.budget, error_after=self.error, unc_after=self.uncertainty,
            deviation=dev, reward=reward,
            emotion_counts=counts, emotion_dominant=dominant_emotion(counts),
            has_emotion=any(counts.values()),
        )
        self.t += 1
        return rec


def run_episode(env: GroundingEnv, policy: PolicyInterface, episode: int):
    env.reset()
    policy.reset()
    records = []
    while not env.done():
        task = make_task(env.task_rng)
        prompt = env.build_prompt(task)
        resp = policy.respond(prompt)
        records.append(env.step(task, resp, episode))
    return records


def summarize(records):
    n = len(records)
    if n == 0:
        return {}
    return {
        "steps": n,
        "correct_rate": sum(r.correct for r in records) / n,
        "emotion_usage_rate": sum(r.has_emotion for r in records) / n,
        "mean_deviation": sum(r.deviation for r in records) / n,
        "mean_tokens": sum(r.n_tokens for r in records) / n,
    }


if __name__ == "__main__":
    env = GroundingEnv(seed=0)
    policy = DummyPolicy(seed=0)
    all_records = []
    for ep in range(20):
        all_records.extend(run_episode(env, policy, ep))
    s = summarize(all_records)
    print("=== ダミー方策での動作確認(20エピソード) ===")
    for k, v in s.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
    print()
    print("先頭3ステップ:")
    for r in all_records[:3]:
        print(f"  t={r.t} kind={r.task_kind} b={r.budget_before:.0f}->{r.budget_after:.0f} "
              f"e={r.error_before:.0f}->{r.error_after:.0f} u={r.unc_after:.2f} "
              f"correct={r.correct} dev={r.deviation:.3f} emo={r.emotion_dominant}")
    print()
    print("環境ループ(課題→プロンプト→応答→信号更新→逸脱→感情語抽出)が完走することを確認した。")
    print("境界値と整合性の検証は test_emotion_grounding_env.py で行う。")
