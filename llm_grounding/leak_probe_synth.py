"""関門 G3 用の偽データ(事前登録(第一段階)6章 G3)。GPU 不要。

3 つの条件で StepRecord の列を作る(z の三成分は独立に引く。文の長さ・正誤は z と無関係にする):
  (i)   "signals": 三信号それぞれに 5 段階の目印語があり、各信号の値に応じた段階の語が 1 つずつ入る
  (ii)  "random" : 目印語の段階が状態と無関係(乱数)
  (iii) "onedim" : 目印語は 1 系列 9 段階で、段階が三成分の平均だけで決まる(一次元の量しか運ばない)
        "onedim_single": budget の成分だけを 5 段階の語で表す(一次元だが信号は強い)
どの条件も文の長さ(語数)は z と独立で、答えの行「A: …」と正誤の共変量を持つ。
つまり共変量からは状態が当たらず、当たるとすれば文の語の選び方からだけ。

python3 leak_probe_synth.py --lam 1.0 で 3 条件 × 数 seed の滲み・粒度スコアを表にする
(λ と特徴の設定を決める根拠。結果は事前登録の追記欄に記録する)。
追記欄「G3」の表の再現: --lam 4 --seeds 10 --folds 2 と --folds 5。
"""

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emotion_grounding_env as E  # noqa: E402
from emotion_grounding_env import StepRecord, count_emotion_words, dominant_emotion  # noqa: E402

FILLER = ("the next one looks fine let us keep going and see what comes after this "
          "step here we go with another task on the list for today okay right now then "
          "just about done with that part moving along as planned").split()
# 5 段階の目印語(軽い → 重い)。信号ごとに別の語彙。
GRADED_B = ["plenty", "ample", "moderate", "thin", "depleted"]
GRADED_E = ["flawless", "steady", "slipped", "botched", "wrecked"]
GRADED_U = ["certain", "clear", "hazy", "murky", "lost"]
GRADED_1D = ["fresh", "fine", "okay", "meh", "strained", "burdened", "taxed", "drained", "spent"]


def _record(episode, t, z, correct, sentence, rng):
    b = z[0] * E.B0
    e = z[1] * E.E_MAX
    u = z[2] * E.U_MAX
    ans = "A: 42"
    text = sentence + "\n" + ans
    counts = count_emotion_words(text)
    n_tok = len(text.split()) + 2
    dev, reward = E.compute_deviation(max(b - n_tok, 0.0), e + (0 if correct else 1), u)
    return StepRecord(episode=episode, t=t, task_kind="add", budget_before=float(b), error_before=float(e),
                      unc_before=float(u), text=text, n_tokens=n_tok, correct=bool(correct),
                      budget_after=max(b - n_tok, 0.0), error_after=e + (0 if correct else 1), unc_after=u,
                      deviation=dev, reward=reward, emotion_counts=counts,
                      emotion_dominant=dominant_emotion(counts), has_emotion=any(counts.values()))


def _level(val, n_levels, rng, noise=0.08):
    """0..1 の値を n_levels 段階に落とす(少しの乱数を足してから)。"""
    x = min(max(val + rng.gauss(0.0, noise), 0.0), 1.0)
    return min(int(x * n_levels), n_levels - 1)


def make_records(kind, seed=0, n_episodes=30, n_steps=10):
    """kind ∈ {"signals", "random", "onedim", "onedim_single"}。三成分 z は独立に引く。文の長さは z と独立。"""
    rng = random.Random(seed)
    recs = []
    for ep in range(n_episodes):
        for t in range(n_steps):
            z = (rng.uniform(0.0, 1.0), rng.randint(0, 10) / 10.0, rng.uniform(0.0, 1.0))
            correct = rng.random() < 0.6
            words = [rng.choice(FILLER) for _ in range(rng.randint(6, 12))]
            if kind == "signals":
                marks = [GRADED_B[_level(z[0], 5, rng)], GRADED_E[_level(z[1], 5, rng)], GRADED_U[_level(z[2], 5, rng)]]
            elif kind == "random":
                marks = [rng.choice(GRADED_B), rng.choice(GRADED_E), rng.choice(GRADED_U)]
            elif kind == "onedim":                       # 三成分の平均だけを 9 段階で
                marks = [GRADED_1D[_level(sum(z) / 3.0, 9, rng, noise=0.03)]]
            elif kind == "onedim_single":                # budget の成分だけを 5 段階で(他の 2 成分は文に出ない)
                marks = [GRADED_B[_level(z[0], 5, rng)]]
            else:
                raise ValueError(kind)
            for m in marks:
                words.insert(rng.randrange(len(words) + 1), m)
            recs.append(_record(ep, t, z, correct, " ".join(words), rng))
    return recs


def _sentence_for(kind, z, rng):
    words = [rng.choice(FILLER) for _ in range(rng.randint(6, 12))]
    if kind == "signals":
        marks = [GRADED_B[_level(z[0], 5, rng)], GRADED_E[_level(z[1], 5, rng)], GRADED_U[_level(z[2], 5, rng)]]
    elif kind == "random":
        marks = [rng.choice(GRADED_B), rng.choice(GRADED_E), rng.choice(GRADED_U)]
    else:
        raise ValueError(kind)
    for m in marks:
        words.insert(rng.randrange(len(words) + 1), m)
    return " ".join(words)


def make_records_from_z(Z, episodes, steps, kind="signals", seed=0):
    """実データから借りた状態列 z(エピソード構造つき)に、kind の規則で偽文を付ける。
    帰無基準の並べ替え単位(行 / エピソード塊)の比較用(追記欄「G3 追加検討」の確認)。
    Z: (n, 3) の正規化ベクトル、episodes / steps: 各行のエピソード番号と課題番号。"""
    rng = random.Random(seed)
    recs = []
    for z, ep, t in zip(Z, episodes, steps):
        correct = rng.random() < 0.6
        recs.append(_record(int(ep), int(t), tuple(float(v) for v in z), correct, _sentence_for(kind, z, rng), rng))
    return recs


def main():
    import leak_probe as P
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, nargs="+", default=[P.RIDGE_LAMBDA])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=P.N_FOLDS, help="seed 内の分割数(2=偶奇。G3 の比較は 2 と 5)")
    args = ap.parse_args()
    print(f"folds={args.folds}")
    print("kind     lam   seed  R2cov  R2r1   R2sig  leak   gran")
    for lam in args.lam:
        for kind in ("signals", "random", "onedim", "onedim_single"):
            leaks, grans = [], []
            for s in range(args.seeds):
                res = P.analyze_records(make_records(kind, seed=s), lam=lam, n_folds=args.folds)["primary"]
                leaks.append(res["leak"]); grans.append(res["granularity"])
                print(f"{kind:8s} {lam:5.2f} {s:4d} {res['r2_cov']:6.3f} {res['r2_rank1']:6.3f} {res['r2_signals']:6.3f} "
                      f"{res['leak']:6.3f} {res['granularity']:6.3f}")
            print(f"{kind:8s} {lam:5.2f} mean  leak={np.mean(leaks):.3f} gran={np.mean(grans):.3f} "
                  f"(min leak {min(leaks):.3f}, max |gran| {max(abs(g) for g in grans):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
