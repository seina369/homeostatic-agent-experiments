"""C0 / T1 の実行結果ファイル(1 seed = 1 JSON)の形式と読み書き。

ランナー(run_c0.py 等)が seed ごとに書き、分析(analyze_granularity.py)が読む。
中身は StepRecord をそのまま dict にしたもの(全ステップ・全フィールド、選別なし)と、
再現に必要な環境定数・方策情報のスナップショット。

ファイル名: {condition}_seed{NN}.json(例: c0_seed03.json)
書き込みは同じフォルダに一時ファイルを書いてから os.replace で置き換える
(Colab が途中で切れても、壊れた JSON が残らないようにするため)。
"""

import json
import os
import sys
import tempfile
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emotion_grounding_env as E  # noqa: E402
from emotion_grounding_env import StepRecord  # noqa: E402

FORMAT_VERSION = 1


def env_constants() -> dict:
    """結果ファイルに残す環境定数のスナップショット(再開時の整合チェックにも使う)。"""
    return {
        "B0": E.B0,
        "N_TASKS": E.N_TASKS,
        "BUDGET_LOW_THRESHOLD": E.BUDGET_LOW_THRESHOLD,
        "E_MAX": E.E_MAX,
        "U_OPT": E.U_OPT,
        "U_MIN": E.U_MIN,
        "U_MAX": E.U_MAX,
        "task_kinds": ["add", "sub", "mul", "count"],
        "emotion_categories": sorted(E.EMOTION_LEXICON),
    }


def seed_file_name(condition: str, seed: int) -> str:
    return f"{condition.lower()}_seed{seed:02d}.json"


def records_to_dicts(records) -> list:
    return [asdict(r) for r in records]


def records_from_dicts(dicts) -> list:
    return [StepRecord(**d) for d in dicts]


def write_seed_file(path: str, *, condition: str, seed: int, n_episodes: int,
                    records, policy_info: dict, extra: dict = None) -> dict:
    """1 seed 分を書く。records は StepRecord のリスト(全エピソード分)。"""
    payload = {
        "format_version": FORMAT_VERSION,
        "condition": condition,
        "seed": seed,
        "n_episodes": n_episodes,
        "n_steps": len(records),
        "complete": True,
        "env_constants": env_constants(),
        "policy": dict(policy_info),
    }
    if extra:
        payload.update(extra)
    payload["records"] = records_to_dicts(records)

    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return payload


def read_seed_file(path: str) -> dict:
    """payload を返す。payload["records"] は StepRecord のリストに変換済み。"""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    payload["records"] = records_from_dicts(payload.get("records", []))
    return payload


def is_complete_seed_file(path: str, n_episodes: int, condition: str = None) -> tuple:
    """再開判定。(True, "") なら飛ばしてよい。(False, 理由) なら実行し直す。"""
    if not os.path.exists(path):
        return False, "not found"
    try:
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
    except (OSError, ValueError) as e:
        return False, f"unreadable ({e})"
    if not p.get("complete"):
        return False, "incomplete"
    if p.get("n_episodes") != n_episodes:
        return False, f"n_episodes mismatch ({p.get('n_episodes')} != {n_episodes})"
    if condition is not None and p.get("condition") != condition:
        return False, f"condition mismatch ({p.get('condition')} != {condition})"
    if p.get("env_constants") != env_constants():
        return False, "env_constants mismatch (environment changed since this file was written)"
    return True, ""
