"""
感情AIプロジェクト フェーズ1 実機移行プロトタイプ(ハードウェア抽象層版)
================================================================

計画書「感情を持つAIプロジェクト計画書」フェーズ1(基盤研究:センサー・恒常性システム)を、
実機(Raspberry Pi + センサー + モーター)へ移行するための土台。

設計方針:
  - 恒常性の計算(逸脱→賞罰変換)・Q学習ロジックは、既存のシミュレーション版
    (homeostasis_prototype.py)からほぼそのまま移植する。変わるのは「センサーの値を
    どう取得するか」「行動をどう実行するか」という、環境とのインターフェース部分だけ。
  - このインターフェースを HardwareInterface という抽象クラスとして切り出し、
    実物のセンサーが届く前は MockHardwareInterface(ダミーの値を返す模擬実装)で
    動作確認する。実機が届いたら RealHardwareInterface の中身(現在は
    NotImplementedError)を GPIO 読み取り・モーター制御コードで埋めるだけでよい。

経緯と設計変更の履歴(2026-08-04):
  1. 当初は「食料/シェルターへの方向」という位置情報を外し、3センサーの離散化ビン
     のみを状態とする簡略版だった。しかしこれには、位置に基づく目的地が存在せず
     自律充電もできないため「なるべく動かない」が文字通り最適方策になってしまう、
     という欠陥があった。
  2. カメラを追加し、「今までにない景色を見ること」自体に好奇心ボーナスを与える
     方式でこれを一部解消した(視覚的な違いを訪問回数ベースの新規性として扱えば、
     ビーコン等の位置インフラなしで探索の動機になる。4.4.3で検証済みの好奇心報酬の
     仕組みを、多エージェント間の伝達ではなく単一エージェントの探索動機として転用)。
  3. その後、「損傷は人間の補修待ちでよいが、エネルギーを自分で確保できない設計は
     欠陥が大きすぎる」という指摘があり、これは正当と判断した(頻繁に訪れる必須の
     需要を毎回外部依存にするのは、稀な重大事故の補修を外部に頼るのとは性質が違う)。
     好奇心用に積むカメラを流用し、充電ドックに貼ったマーカー(ArUco等)を目印に
     接近する「dock」行動を追加した。最後の数センチの位置合わせだけは、離散的な
     行動と学習初期のランダム試行錯誤では事故リスクが高いため、固定的な自動アライン
     機構(視覚サーボによる反射的な動作)として扱う。「充電に行くかどうか・いつ
     行くか」は引き続きQ学習の対象、「どう接続するか」だけを固定機構にする。
  4. 計画書7.2で必須としていた「物理実装後の手動緊急停止機構」が未実装だったため
     追加した。ソフトウェア(学習の出来不出来)に一切依存せず、人間がいつでも
     問答無用でモーターを止められる手段。本当の安全弁はモーター電源を物理的に
     遮断するハードウェアリレー/スイッチであるべきで、is_emergency_stopped()は
     それと連動する状態を読み取るだけの窓口に過ぎない。
  5. 損傷の発生確率を、行動と無関係な一律値から、移動する行動の方が高いという
     当たり前の因果関係を反映した値に改めた(ただし実機のモーター電流・エンコーダ
     不一致が本当に「損傷」を正しく代理できているかは、部品が届いてから実測で
     検証が要る。床材の違いや電圧のゆらぎでも同様の信号が出うるため、これは
     まだ確定した設計ではない)。

未解決のまま残していること:
  状態空間(センサー3種+視覚シーン+ドック距離)が広がってきており、要件7の
  シミュレーションで確認された「タブラーQ学習は複雑な状態空間で汎化に弱い」という
  傾向にいずれ近づく可能性がある。ただし実機は同じ状態を生涯を通じて繰り返し踏むため
  シミュレーションほど深刻にならない可能性もあり、確度の低い懸念として様子見にとどめ、
  今回はタブラー版のまま進める(必要になれば homeostasis_nn_prototype.py のNNエージェント
  への切り替えを検討する)。

注意: これはフェーズ1限定のプロトタイプであり、計画書のフェーズ3(外部監督組織)を
経ていない。要件4(自己保存本能・罰による不可逆な削除)は一切実装しない。
"""

import os
import time
import random
import pickle
import numpy as np
from abc import ABC, abstractmethod

# ------------------------------------------------------------
# 設定(既存のシミュレーション版とできるだけ揃える)
# ------------------------------------------------------------
ACTIONS = ["forward", "backward", "left", "right", "stay", "dock"]

# カメラ画像を粗く離散化した「視覚シーンID」の総数(暫定値)。実機では、
# ダウンサンプリングした画像をハッシュ化・量子化してこの範囲のバケットに
# 落とし込むことを想定。絶対座標ではなく「見た目がどれだけ違うか」だけを
# 扱うため、ビーコン等の位置インフラなしで新規性を検知できる。
N_SCENE_BUCKETS = 12

# 好奇心ボーナスの重み(暫定値、要調整)。訪問回数ベースの新規性ボーナス
# (curiosity_bonus参照)にかける係数で、典型的な恒常性報酬の大きさ
# (逸脱0.3〜0.8程度)と同程度になるよう仮に設定している。
CURIOSITY_WEIGHT = 0.5

# 充電ドックの位置(視覚シーンの疑似グリッド上での座標、モック専用)。
# 実機ではこの「位置」という概念自体を持たず、代わりにカメラでドックの
# マーカー(ArUco等)が見えるかどうか・どれだけ近いかだけを読み取る
# (read_dock_distance_bin参照)。
DOCK_SCENE_POSITION = (0, 0)

# 「dock」行動を、ドック位置で選んだ場合の1ステップあたりの充電量(暫定値)。
# 実機の充電電流・バッテリー容量が確定次第、現実的な値に調整する。
CHARGE_RATE_PER_STEP = 8.0

# 損傷が発生する確率(暫定値)。移動する行動の方が、静止系の行動(stay/dock)より
# 損傷リスクが高いという当たり前の因果関係を持たせる(2026-08-04の議論。以前は
# 行動と無関係な一律確率だった)。ただし、これが実機のモーター電流・エンコーダ
# 不一致という実際のセンサー信号の挙動を正しく代理できているかは未検証。
DAMAGE_RISK_MOVE = 0.05
DAMAGE_RISK_STAY = 0.01

OPTIMAL_DAMAGE = 0.0

# エネルギーが低いほど、枯渇(運用停止)に近づく危険として罰を強める設計。
# ENERGY_LOW_THRESHOLD以上では罰を設けない(energy_deviation参照)。
# 「リチウムイオン電池は満充電を保ち続けると劣化が早まるため8割程度に留める方が
# 長寿命」という事実はあるが、これは人間がいつ充電をやめるかという運用手順の
# 問題であり、報酬には含めない。
ENERGY_LOW_THRESHOLD = 20.0

# 温度の最適値・安全範囲は、人間の体温(36℃前後)ではなく、この個体が実際に
# 搭載する部品の制約から定める(要件1・2の設計方針: 外部の一般的な人間像では
# なく機械自身の状態を予測対象にする、に対応)。
#   - Raspberry Pi自体は80℃でスロットリング開始・85℃が上限と、かなり高温まで耐える。
#   - 律速するのはリチウムイオン系バッテリーの方。効率・寿命が最大化されるのは
#     15〜35℃で、充電時の安全範囲は0〜45℃(0℃未満での充電はリチウムの析出という
#     劣化・発火リスクにつながる)。
# したがって「最適温度」はこのバッテリー安全域の中央値を採用する。人間の快適な
# 室温と近い数値になるのは偶然であり、根拠は人間の生理ではなく搭載部品の化学的な
# 制約である点に注意(2026-08-04の議論を踏まえた修正)。
OPTIMAL_TEMP = 25.0
TEMP_SAFE_MIN = 0.0    # これを下回ると充電時にリチウム析出のリスク
TEMP_SAFE_MAX = 45.0   # バッテリーの充電上限。超えると劣化が加速し、Piのスロットリングにも近づく

ALPHA = 0.2      # 学習率
GAMMA = 0.95     # 割引率
EPS_START = 1.0
EPS_END = 0.05
# 元のグリッドワールド版(homeostasis_prototype.py)はEPS_DECAY_EPISODES=2000を
# エピソード単位で使い、全3000エピソード中の2/3を探索率の減衰に充てていた。
# この実機版へ移植した際、単位をエピソードからステップへ変える一方で数値の
# 再計算をしておらず、2000ステップ(default n_episodes=200×120stepsなら
# 全24000ステップ中のわずか8%)で減衰が終わってしまっていた。しかも状態空間は
# 視覚シーン・ドック距離の追加で約18倍(216→3888通り)に広がっており、以前より
# 遥かに長く探索する必要がある(2026-08-04に発見した欠陥の修正)。
# 元の設計と同じ「全体の2/3を減衰に充てる」比率を、現在のdefault
# n_episodes=200×MAX_STEPS_PER_EPISODE=120=24000ステップに当てはめて再計算する。
EPS_DECAY_STEPS = 16000

STEP_DURATION_SEC = 0.5   # 実機での1ステップあたりの待機時間(モックでは無視してよい)
MAX_STEPS_PER_EPISODE = 120

# Qテーブル・学習進捗の保存先(暫定パス)。実機には「エピソードの区切り」も
# 「学習の終わり」もなく常時稼働するため、緊急停止や再起動のたびに学習内容が
# 消えないよう、ここに保存・復元する(2026-08-04に発見した欠陥の修正)。
Q_TABLE_PATH = "q_table_state.pkl"


# ------------------------------------------------------------
# ハードウェア抽象層
# ------------------------------------------------------------
class HardwareInterface(ABC):
    """実機・模擬環境どちらでも同じインターフェースでやり取りするための抽象クラス。"""

    @abstractmethod
    def read_energy(self) -> float:
        """0〜100のバッテリー残量相当の値を返す。"""
        ...

    @abstractmethod
    def read_temperature(self) -> float:
        """摂氏の温度を返す(DS18B20などの想定)。"""
        ...

    @abstractmethod
    def read_damage(self) -> float:
        """0〜100の損傷度を返す(モーター電流・エンコーダ不一致の積算値などの想定)。"""
        ...

    @abstractmethod
    def do_action(self, action: str) -> None:
        """モーターに行動を実行させる。"""
        ...

    @abstractmethod
    def read_scene_id(self) -> int:
        """
        0〜N_SCENE_BUCKETS-1の、粗く離散化された「今見えている景色」の
        識別子を返す(カメラ画像を想定)。絶対的な位置情報ではなく、
        「前に見た景色と同じか違うか」だけが分かればよい新規性検知用の
        信号である点に注意。
        """
        ...

    @abstractmethod
    def read_dock_distance_bin(self) -> int:
        """
        充電ドックのマーカー(ArUco等)との関係を、0〜2の粗いビンで返す。
        0=接続を試みられる近さ、1=視野内だが遠い、2=マーカーが見えていない。
        絶対座標ではなく、カメラでマーカーがどう見えているかだけに基づく点は
        read_scene_id() と同じ設計思想。
        """
        ...

    @abstractmethod
    def is_emergency_stopped(self) -> bool:
        """
        物理的な緊急停止スイッチが押されているかを返す。実機ではGPIO等で
        常時監視するボタン/リレーの状態を想定し、do_action()を呼ぶ前に
        必ずこれを確認して、真であれば行動を実行しない。

        重要: 本当の安全弁はモーター電源ラインを物理的に遮断するハードウェア
        リレー/スイッチであるべきで、このメソッドはソフトウェア側がその状態を
        把握するための窓口に過ぎない。Raspberry Pi本体がフリーズ・クラッシュ
        しても物理的にモーターを止められることが、ソフトウェアだけの実装より
        優先されるべきである(計画書7.2の「手動緊急停止機構」に対応)。
        """
        ...

    def close(self) -> None:
        """終了時の後片付け(GPIOの解放など)。模擬実装では何もしなくてよい。"""
        pass

    def mark_repaired(self) -> None:
        """
        損傷が実際に補修されたことを外部から明示的に伝えるためのメソッド。

        AI自身が「治った」と自己申告することはできない(自己申告と実際の状態を
        混同しないという、このプロジェクト全体の立場に反するため)。あくまで
        人間が実際に修理した後に、人間の操作(ボタン押下・CLIコマンド等)を
        通じて外部から呼び出すことを想定する。学習ループの中からは呼ばない。
        デフォルトでは何もしない(サブクラスで上書きする)。

        大きな怪我に病院が要るのと同様、機械的な損傷の補修は引き続き人間の
        介入を前提とする(エネルギーの自律確保とは性質が異なるため)。
        """
        pass

    def mark_recharged(self) -> None:
        """
        人間が手動で充電器へ接続したことを外部から明示的に伝えるための、
        あくまで補助的なフォールバック。

        主経路は agent が学習する「dock」行動によるドッキングであり、これは
        学習ループの中で自然に評価される。mark_recharged() は、ドッキングに
        繰り返し失敗する・マーカーが汚れて見えない等の理由で自律充電が
        機能しないときに、人間が直接介入するための保険として残す。
        デフォルトでは何もしない(サブクラスで上書きする)。
        """
        pass


class MockHardwareInterface(HardwareInterface):
    """
    部品が届く前に、コード全体の動作確認・デバッグを行うための模擬実装。

    実際のセンサーの代わりに、行動に応じてそれらしく変化する疑似的な値を返す。
    視覚シーンは4行×3列の小さな循環グリッド上の疑似的な位置として表現し、
    ドックはその中の固定位置(DOCK_SCENE_POSITION)にあるものとする。実機を
    待たずにQ学習のループ全体(状態取得→行動選択→実行→報酬計算→更新)が
    正しく回ることを確認するのが目的であり、学習内容そのものの妥当性を
    検証するものではない(学習内容の妥当性はグリッドワールド版で既に確認済み)。
    """

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.energy = 100.0
        self.temperature = OPTIMAL_TEMP
        self.damage = 0.0
        self.scene_row = 0
        self.scene_col = 0
        self.emergency_stopped = False

    def read_energy(self) -> float:
        return self.energy

    def read_temperature(self) -> float:
        return self.temperature

    def read_damage(self) -> float:
        return self.damage

    def read_scene_id(self) -> int:
        return self.scene_row * 3 + self.scene_col

    def read_dock_distance_bin(self) -> int:
        dist = abs(self.scene_row - DOCK_SCENE_POSITION[0]) + abs(self.scene_col - DOCK_SCENE_POSITION[1])
        if dist == 0:
            return 0
        elif dist <= 2:
            return 1
        else:
            return 2

    def is_emergency_stopped(self) -> bool:
        return self.emergency_stopped

    def trigger_emergency_stop(self) -> None:
        """テスト用: 緊急停止ボタンが押された状態を模擬する(模擬専用の便宜メソッド)。"""
        self.emergency_stopped = True

    def clear_emergency_stop(self) -> None:
        """テスト用: 緊急停止を解除する(実機でも、解除は人間の明示的な操作を要する想定)。"""
        self.emergency_stopped = False

    def do_action(self, action: str) -> None:
        # 緊急停止中は物理的にモーターへ信号を送らない、という前提をモックでも
        # 再現する(本来の安全弁はハードウェアの電源遮断だが、ソフト側の整合性
        # のためモックでも同様にガードする)。
        if self.emergency_stopped:
            return

        move_actions = ("forward", "backward", "left", "right")

        # エネルギー消費: 移動する行動は1.5、静止系の行動(stay/dock)は0.5。
        # 「stay」でもエネルギーが減り続けるのは意図的な設計: Raspberry Pi・
        # センサー類は動作していなくても待機電力を消費し続けるため、「何もしない」を
        # 選び続けても餓死(エネルギー枯渇)からは逃げられないようにしてある。
        self.energy -= 1.5 if action in move_actions else 0.5
        # 温度は実機なら本物のセンサー値をそのまま返すだけでよく、Mockが勝手に
        # 揺らす理由はない(2026-08-04の議論)。以前はランダムウォーク+シェルター
        # 相当の引き戻しイベントを実装していたが、後者は元のグリッドワールド版の
        # 名残で、位置情報を外した今は行動と無関係な不自然な処理になっていた。
        # 単純に固定値のまま変化させないことにする。

        # 視覚シーン(=疑似的な位置)の更新。dockはその場での位置合わせなので位置は変えない。
        if action == "forward":
            self.scene_row = (self.scene_row + 1) % 4
        elif action == "backward":
            self.scene_row = (self.scene_row - 1) % 4
        elif action == "right":
            self.scene_col = (self.scene_col + 1) % 3
        elif action == "left":
            self.scene_col = (self.scene_col - 1) % 3

        # 損傷: 移動する行動の方が静止系の行動よりリスクが高い(2026-08-04の議論。
        # 以前は行動と無関係な一律確率だった)。ただし実機での妥当性は未検証。
        damage_risk = DAMAGE_RISK_MOVE if action in move_actions else DAMAGE_RISK_STAY
        if self.rng.random() < damage_risk:
            self.damage += 15.0
        # 損傷は自動回復させない。モーターの過負荷・車体の破損といった実際の
        # 物理的な損耗は、時間が経てば自然に治るものではないため。

        # 充電: 「dock」を選び、かつドック位置にいる場合のみ充電される。
        # ドック位置にいない状態でdockを選んでも、位置合わせの空振りとして
        # エネルギーは通常のペースで減るだけで恩恵はない。
        if action == "dock" and (self.scene_row, self.scene_col) == DOCK_SCENE_POSITION:
            self.energy += CHARGE_RATE_PER_STEP

        self.energy = float(np.clip(self.energy, 0.0, 100.0))
        self.damage = float(np.clip(self.damage, 0.0, 100.0))

    def reset(self):
        """
        エピソードの区切りで模擬環境を初期状態へ戻す(実機にはない、模擬専用の便宜メソッド)。
        緊急停止の状態はここではリセットしない: 実機で緊急停止が「エピソードの
        区切り」程度で自動解除されるのは安全設計として不適切なため、意図的に
        reset()の対象から外している。
        """
        self.energy = 100.0
        self.temperature = OPTIMAL_TEMP
        self.damage = 0.0
        self.scene_row = 0
        self.scene_col = 0

    def mark_repaired(self) -> None:
        """人間が補修した、という外部からの操作を模擬する(損傷を0に戻す)。"""
        self.damage = 0.0

    def mark_recharged(self) -> None:
        """人間が手動で充電した、という外部からの操作を模擬する(エネルギーを満タンに戻す)。"""
        self.energy = 100.0


class RealHardwareInterface(HardwareInterface):
    """
    実機(Raspberry Pi)向けの実装。部品確定後にここを埋める。

    想定する接続(4.5参照・要件1・2):
      - 体温: DS18B20(1-wireデジタル温度センサー、GPIO4想定)
      - エネルギー: バッテリー電圧をADS1115などのI2C ADC経由で読み取り、
        満充電時の電圧を100・下限電圧を0とする線形マッピングで0〜100に変換
      - 損傷: モーター電流値(INA219などの電流センサー、過大な電流を機構的な負荷の
        兆候として検知)と、車輪エンコーダの実測値/命令値の不一致(意図した通りに
        動けているかの直接的な指標)を組み合わせて算出する。加速度センサー(MPU6050)
        による外部からの衝撃検知を補助的に併用してもよい。いずれの信号も自動では
        回復させない(実際の機械的な損耗は時間経過だけでは治らないため)。
        ただし床材の違いや電圧のゆらぎでも同様の信号が出うるため、実測での
        キャリブレーションが要る(未検証のまま残っている点)。
      - 行動: モータードライバ経由でDCモーター2基を制御(前進・後退・左右旋回・停止)。
        「dock」行動は、カメラでドックのマーカー(ArUco等)を正面から検出したのち、
        Anki Cozmoの充電ドックを参考に、後ろ向きになってバックでドックへ乗り上げる
        固定的な視覚サーボルーチンとして実装する(Q学習で細かい位置合わせまで
        学習させるのではなく、反射的な動作として扱う)。接点は精密なポゴピンではなく
        幅の広い金属板にし、多少の位置ズレでも確実に導通するようにする(ポゴピンだと
        半端な接触でショートしやすいが、幅広接点は位置ズレへの耐性で吸収できる)。
        なお有志によるCozmoの再現実装では、この方式でも成功率はおよそ5割程度に
        とどまっており、「絶対に成功する」設計ではなく「多少失敗しても壊れない」
        設計として理解すること。緊急停止・損傷の非自動回復といった既存の安全策は、
        この方式を採用してもなお必要である。
      - 視覚シーン: カメラモジュール(Pi Camera等)の画像を大きくダウンサンプリング
        (例: 8×8グレースケール)した上でハッシュ・量子化し、N_SCENE_BUCKETS個の
        バケットのいずれかに割り当てる。絶対位置ではなく新規性検知だけが目的
        なので、厳密な特徴抽出は不要。
      - ドック距離: 同じカメラ画像からArUcoマーカー(OpenCVのaruco モジュール等)を
        検出し、マーカーが画像内にどれだけ大きく・中心付近に写っているかから
        read_dock_distance_bin()の3ビンに粗く変換する。
      - 緊急停止: 物理ボタン/リレーをモーター電源ラインに直列で挟み、GPIOでも
        状態を読み取れるようにする。is_emergency_stopped()はこのGPIO状態を返す
        だけで、実際の遮断は配線レベルで保証する。
    """

    def __init__(self):
        raise NotImplementedError(
            "実機のGPIO配線・部品が確定してから実装する。"
            "配線図(どのセンサーをどのGPIO/I2Cアドレスに繋ぐか)を先に固めること。"
            "特に緊急停止用のリレー配線は、他のどの機能より先に確認すること。"
        )

    def read_energy(self) -> float:
        raise NotImplementedError

    def read_temperature(self) -> float:
        raise NotImplementedError

    def read_damage(self) -> float:
        raise NotImplementedError

    def do_action(self, action: str) -> None:
        raise NotImplementedError

    def read_scene_id(self) -> int:
        raise NotImplementedError

    def read_dock_distance_bin(self) -> int:
        raise NotImplementedError

    def is_emergency_stopped(self) -> bool:
        raise NotImplementedError

    def mark_repaired(self) -> None:
        """
        人間が実際に補修した後に、人間自身が(ボタン押下・CLIコマンド等で)
        呼び出すことを想定。学習ループやAI自身の判断からは呼ばない。
        実装は部品確定後、損傷度の内部カウンタを0に戻す処理を書けばよい。
        """
        raise NotImplementedError

    def mark_recharged(self) -> None:
        """
        ドッキングによる自律充電が機能しないときの手動フォールバック。
        人間が実際に充電器を接続した後に、人間自身が呼び出すことを想定。
        """
        raise NotImplementedError


# ------------------------------------------------------------
# 恒常性の計算(homeostasis_prototype.py からほぼそのまま移植)
# ------------------------------------------------------------
def discretize(hw: HardwareInterface):
    energy = hw.read_energy()
    temperature = hw.read_temperature()
    damage = hw.read_damage()
    scene_id = hw.read_scene_id()
    dock_bin = hw.read_dock_distance_bin()

    e_bin = int(np.clip(energy // 20, 0, 5))
    # ビンの幅は、人間の快適温度帯ではなくバッテリーの安全範囲(0〜45℃)を6分割して決める
    t_bin = int(np.clip((temperature - TEMP_SAFE_MIN) // 9, 0, 5))
    d_bin = int(np.clip(damage // 20, 0, 5))
    scene_bin = int(np.clip(scene_id, 0, N_SCENE_BUCKETS - 1))
    dock_bin = int(np.clip(dock_bin, 0, 2))
    # 状態に視覚シーン・ドック距離を含めることで、ビーコン等を使わずに「今どこにいて、
    # ドックにどれだけ近いか」に応じた行動をQ学習が区別して学習できるようにする。
    return (e_bin, t_bin, d_bin, scene_bin, dock_bin), (energy, temperature, damage)


def temp_deviation(temperature: float) -> float:
    """
    温度逸脱を、バッテリーの安全範囲(TEMP_SAFE_MIN〜TEMP_SAFE_MAX)を基準に非対称に計算する。
    安全範囲内では最適値からの距離に比例した緩やかな罰(0〜1)、範囲を逸脱すると
    (充電時のリチウム析出・Piのスロットリングといった実害に近づくため)罰が急激に増す。

    OPTIMAL_TEMPはTEMP_SAFE_MINとTEMP_SAFE_MAXのちょうど中央ではない(25は
    15〜35℃という効率最大域の中央値であり、0〜45℃という安全範囲全体の中央
    ではない)。そのため上下で正規化の基準(span)を分け、それぞれの境界で
    必ずちょうど1.0になるようにする(2026-08-04、境界値の直接検証で発見した
    バグの修正: 単一のhalf_widthで両方向を正規化すると、下限側で1.0を超え、
    安全域を割り込んだ瞬間に逸脱がむしろ下がるという不連続が生じていた)。
    """
    if temperature <= OPTIMAL_TEMP:
        span = OPTIMAL_TEMP - TEMP_SAFE_MIN
        if temperature >= TEMP_SAFE_MIN:
            return (OPTIMAL_TEMP - temperature) / span
        overshoot = TEMP_SAFE_MIN - temperature
        return 1.0 + overshoot / 10.0
    else:
        span = TEMP_SAFE_MAX - OPTIMAL_TEMP
        if temperature <= TEMP_SAFE_MAX:
            return (temperature - OPTIMAL_TEMP) / span
        overshoot = temperature - TEMP_SAFE_MAX
        return 1.0 + overshoot / 10.0


def energy_deviation(energy: float) -> float:
    """
    エネルギーが低いほど、枯渇(運用停止)に近づく危険として罰を強める。
    ENERGY_LOW_THRESHOLD以上では罰を設けない(高エネルギー側の扱いはエージェントの
    行動と無関係なので報酬に含める意味がない、という以前からの設計は維持している。
    「dock」による自律充電を導入した後も、この閾値のロジック自体は変える必要がない:
    エージェントはdock行動を通じてエネルギーを増やせるようになったが、報酬が
    意味を持つべきなのは引き続き「枯渇にどれだけ近いか」であるため)。
    """
    if energy >= ENERGY_LOW_THRESHOLD:
        return 0.0
    return (ENERGY_LOW_THRESHOLD - energy) / ENERGY_LOW_THRESHOLD


def compute_deviation(raw_values):
    energy, temperature, damage = raw_values
    dev_energy = energy_deviation(energy)
    dev_temp = temp_deviation(temperature)
    dev_damage = abs(damage - OPTIMAL_DAMAGE) / 100.0
    deviation = dev_energy + dev_temp + dev_damage
    reward = -deviation
    return deviation, reward


def curiosity_bonus(scene_id: int, visited_counts: dict) -> float:
    """
    訪問回数に基づく好奇心ボーナス(count-based exploration bonus)。

    ICMのような予測誤差ベースの好奇心ではなくこの方式を選ぶ理由: カメラ画像には
    照明のちらつきなど本質的に予測不能なノイズが写り込みうるが、予測誤差ベースの
    好奇心はそうした「予測できないだけで無意味な」対象に延々と引きつけられてしまう
    弱点が知られている("noisy TV problem"、Burda et al. 2018)。訪問回数ベースなら
    同じ視覚バケットを何度見てもボーナスは回数とともに単調に減衰するため、この
    弱点を構造的に避けられる。
    """
    count = visited_counts.get(scene_id, 0)
    return CURIOSITY_WEIGHT / np.sqrt(1.0 + count)


# ------------------------------------------------------------
# Q学習エージェント(homeostasis_prototype.py と同一のロジック)
# ------------------------------------------------------------
# 状態空間について(2026-08-04時点で未解決のまま残す懸念): 視覚シーン・ドック距離を
# 状態に加えたことで、状態空間は最大 6×6×6×12×3 ≈ 3888通りまで広がった。要件7の
# シミュレーションでは、状態空間が複雑になるほどタブラーQ学習は汎化に弱くなる
# ことが確認されているが、実機は同じ状態を生涯を通じて繰り返し踏むためシミュレー
# ションほど深刻にならない可能性もある。確度の低い懸念のため、今回はタブラー版の
# まま進め、必要になれば homeostasis_nn_prototype.py のNNエージェントへの切り替えを
# 検討する。
class QLearningAgent:
    def __init__(self):
        self.q = {}
        self.global_step = 0  # 探索率の減衰に使う、生涯を通じた累積ステップ数

    def _key(self, state, action):
        return (state, action)

    def q_value(self, state, action):
        return self.q.get(self._key(state, action), 0.0)

    def best_action(self, state):
        values = [self.q_value(state, a) for a in ACTIONS]
        return ACTIONS[int(np.argmax(values))]

    def update(self, state, action, reward, next_state):
        current = self.q_value(state, action)
        next_max = max(self.q_value(next_state, a) for a in ACTIONS)
        target = reward + GAMMA * next_max
        self.q[self._key(state, action)] = current + ALPHA * (target - current)

    def save(self, path=Q_TABLE_PATH):
        """
        Qテーブルと累積ステップ数をファイルに保存する。実機では緊急停止時・
        通常終了時に必ず呼び、再起動時にload_or_new()で復元することで、
        再起動のたびに学習内容を失わないようにする(2026-08-04に発見した
        欠陥の修正)。
        """
        with open(path, "wb") as f:
            pickle.dump({"q": self.q, "global_step": self.global_step}, f)

    def load(self, path=Q_TABLE_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q = data["q"]
        self.global_step = data.get("global_step", 0)

    @classmethod
    def load_or_new(cls, path=Q_TABLE_PATH):
        agent = cls()
        if os.path.exists(path):
            agent.load(path)
        return agent


def epsilon_for_step(step):
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + (EPS_END - EPS_START) * frac


# ------------------------------------------------------------
# メインループ
# ------------------------------------------------------------
def run(hw: HardwareInterface, n_episodes=200, dry_run_fast=True, agent=None, q_table_path=Q_TABLE_PATH):
    # agentを渡さなければ、保存済みのQテーブルがあれば読み込み、なければ新規作成する。
    # global_stepもagent自身に持たせて一緒に保存・復元することで、緊急停止からの
    # 再開時に探索率がいきなり1.0へ巻き戻らないようにする(2026-08-04に発見した
    # 欠陥の修正: 以前はrun()を呼ぶたびに空のQテーブルで最初からやり直していた)。
    if agent is None:
        agent = QLearningAgent.load_or_new(q_table_path)
    avg_deviation_per_episode = []
    stay_fraction_per_episode = []
    scene_coverage_per_episode = []
    dock_success_fraction_per_episode = []
    passive_fraction_per_episode = []

    # 好奇心の記憶(視覚バケットごとの訪問回数)。エピソードをまたいで持続させる:
    # 実機の稼働にエピソードの区切りは存在せず、「どこを見たことがあるか」は
    # 個体の生涯を通じた経験として蓄積されるべきものだから。
    visited_counts = {}

    for ep in range(n_episodes):
        if hasattr(hw, "reset"):
            hw.reset()  # 模擬実装のみ持つ便宜メソッド。実機では初期化不要(常時稼働)。

        state, _ = discretize(hw)
        deviations = []
        stay_count = 0
        dock_success_count = 0
        # 「dock」はドック位置にいない限りstayとコスト・損傷リスクが同一で、
        # シーンも変えない。stay_countだけを見ていると、エージェントがドックから
        # 離れた場所で「dock」を選び続ける形の消極性を見逃してしまう
        # (2026-08-04に発見した診断の盲点)。この抜け穴も含めて拾うための集計。
        passive_count = 0

        for _ in range(MAX_STEPS_PER_EPISODE):
            # 緊急停止が検知されたら、学習ループそのものを即座に中断する。
            # 自動的に再開はしない(人間の明示的な解除操作を要する安全設計)。
            if hw.is_emergency_stopped():
                agent.save(q_table_path)
                print("緊急停止が検知されたため、学習ループを中断しました(Qテーブルは保存済み)。")
                return (
                    avg_deviation_per_episode,
                    stay_fraction_per_episode,
                    scene_coverage_per_episode,
                    dock_success_fraction_per_episode,
                    passive_fraction_per_episode,
                )

            eps = epsilon_for_step(agent.global_step)
            if random.random() < eps:
                action = random.choice(ACTIONS)
            else:
                action = agent.best_action(state)

            dock_bin_before = state[4]  # (e_bin, t_bin, d_bin, scene_bin, dock_bin)

            hw.do_action(action)
            if not dry_run_fast:
                time.sleep(STEP_DURATION_SEC)

            next_state, raw_values = discretize(hw)
            deviation, reward = compute_deviation(raw_values)

            scene_bin = next_state[3]
            reward_with_curiosity = reward + curiosity_bonus(scene_bin, visited_counts)
            visited_counts[scene_bin] = visited_counts.get(scene_bin, 0) + 1

            # Q学習の更新には好奇心込みの報酬を使う一方、diagnostics用のdeviationは
            # 恒常性だけを反映した値のまま残す(好奇心・充電が効いて動くように
            # なったことと、恒常性そのものが保てているかを、別々に確認できるようにする)。
            agent.update(state, action, reward_with_curiosity, next_state)

            state = next_state
            deviations.append(deviation)
            if action == "stay":
                stay_count += 1
                passive_count += 1
            if action == "dock":
                if dock_bin_before == 0:
                    dock_success_count += 1
                else:
                    # ドックにいないのに「dock」を選んだ = 実質stayと同じ消極的な選択
                    passive_count += 1
            agent.global_step += 1

        avg_deviation_per_episode.append(float(np.mean(deviations)))
        stay_fraction_per_episode.append(stay_count / MAX_STEPS_PER_EPISODE)
        scene_coverage_per_episode.append(len(visited_counts) / N_SCENE_BUCKETS)
        dock_success_fraction_per_episode.append(dock_success_count / MAX_STEPS_PER_EPISODE)
        passive_fraction_per_episode.append(passive_count / MAX_STEPS_PER_EPISODE)

    agent.save(q_table_path)
    return (
        avg_deviation_per_episode,
        stay_fraction_per_episode,
        scene_coverage_per_episode,
        dock_success_fraction_per_episode,
        passive_fraction_per_episode,
    )


if __name__ == "__main__":
    print("=== 恒常性計算の境界値検証 ===")
    print("(Mockのランダム性に頼らず、狙った値を直接渡して確認する)")
    assert temp_deviation(OPTIMAL_TEMP) == 0.0, "最適温度では逸脱0のはず"
    assert abs(temp_deviation(TEMP_SAFE_MIN) - 1.0) < 1e-9, "安全域の下限では逸脱1のはず"
    assert abs(temp_deviation(TEMP_SAFE_MAX) - 1.0) < 1e-9, "安全域の上限では逸脱1のはず"
    assert temp_deviation(TEMP_SAFE_MAX + 10) > 1.0, "安全域を超えたら逸脱は1を上回るはず"
    assert temp_deviation(TEMP_SAFE_MIN - 10) > 1.0, "安全域を下回っても逸脱は1を上回るはず"
    assert energy_deviation(ENERGY_LOW_THRESHOLD) == 0.0, "閾値ちょうどでは逸脱0のはず"
    assert energy_deviation(100.0) == 0.0, "満充電付近では逸脱0のはず(高エネルギー側は罰なし設計)"
    assert energy_deviation(0.0) == 1.0, "完全枯渇では逸脱1のはず"
    print("温度・エネルギーの逸脱計算、境界値での挙動を確認できた。")
    print()

    print("=== ドライラン(模擬ハードウェア)での動作確認 ===")
    demo_q_path = "q_table_dryrun_demo.pkl"
    if os.path.exists(demo_q_path):
        try:
            os.remove(demo_q_path)  # 前回のデモの保存ファイルが残っていれば消してから始める
        except OSError:
            pass
    mock_hw = MockHardwareInterface(seed=0)
    avg_dev, stay_frac, scene_cov, dock_success, passive_frac = run(
        mock_hw, n_episodes=200, dry_run_fast=True, q_table_path=demo_q_path
    )
    mock_hw.close()

    print(f"最初の20エピソードの平均逸脱: {np.mean(avg_dev[:20]):.4f}")
    print(f"最後の20エピソードの平均逸脱: {np.mean(avg_dev[-20:]):.4f}")
    print(f"最初の20エピソードのstay頻度: {np.mean(stay_frac[:20]):.3f}")
    print(f"最後の20エピソードのstay頻度: {np.mean(stay_frac[-20:]):.3f}")
    print(f"最初の20エピソードの視覚バケット到達率: {np.mean(scene_cov[:20]):.3f}")
    print(f"最後の20エピソードの視覚バケット到達率: {np.mean(scene_cov[-20:]):.3f}")
    print(f"最初の20エピソードのdock成功頻度: {np.mean(dock_success[:20]):.3f}")
    print(f"最後の20エピソードのdock成功頻度: {np.mean(dock_success[-20:]):.3f}")
    print(f"最初の20エピソードの消極性頻度(stay+ドック外dock): {np.mean(passive_frac[:20]):.3f}")
    print(f"最後の20エピソードの消極性頻度(stay+ドック外dock): {np.mean(passive_frac[-20:]):.3f}")
    print("(参考: dock成功頻度が学習とともに上がっていれば、エージェントが")
    print(" 自律的に充電を選べるようになってきている目安になる。消極性頻度は")
    print(" stay単体より広い指標で、dockに隠れた消極性も拾う)")
    print("コード全体(センサー取得→行動選択→実行→報酬計算→Q学習更新)のループが")
    print("エラーなく完走することを確認した。")
    print()

    print("=== Qテーブルの永続化の動作確認 ===")
    reloaded_agent = QLearningAgent.load_or_new(demo_q_path)
    assert len(reloaded_agent.q) > 0, "保存されたQテーブルが空のはず"
    assert reloaded_agent.global_step > 0, "保存された累積ステップ数が0のはず"
    print(f"保存されたQテーブルを読み込み、{len(reloaded_agent.q)}件のエントリと")
    print(f"累積ステップ数{reloaded_agent.global_step}が復元されたことを確認した。")
    print("(緊急停止や再起動をまたいでも学習内容が消えないことの確認)")
    try:
        os.remove(demo_q_path)
    except OSError:
        pass  # 後片付けに失敗しても本筋の検証には影響しない
    print()

    print("=== 緊急停止機構の動作確認 ===")
    estop_q_path = "q_table_estop_demo.pkl"
    if os.path.exists(estop_q_path):
        try:
            os.remove(estop_q_path)
        except OSError:
            pass
    mock_hw2 = MockHardwareInterface(seed=1)
    mock_hw2.trigger_emergency_stop()
    result = run(mock_hw2, n_episodes=5, dry_run_fast=True, q_table_path=estop_q_path)
    completed_episodes = len(result[0])
    print(f"緊急停止済みの状態で学習ループを開始し、{completed_episodes}エピソード目を")
    print("完了する前に中断されたことを確認した(ソフトウェア側がis_emergency_stopped()を")
    print("正しく検知し、行動を実行せず学習ループを止められることの確認)。")
    mock_hw2.close()
    if os.path.exists(estop_q_path):
        try:
            os.remove(estop_q_path)
        except OSError:
            pass

    print()
    print("実機が届いたら RealHardwareInterface を実装して MockHardwareInterface と")
    print("差し替えるだけでよい。ただし緊急停止用の物理配線は、他のどの機能より")
    print("先に確認・単体テストすること。")
