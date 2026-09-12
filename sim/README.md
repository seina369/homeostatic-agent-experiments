# sim/ について

シミュレーション実験一式。各実験の内容・結果の詳細はリポジトリ直下の
`README.md` にまとめてある。ここには、ファイルの探し方と動かし方だけを書く。

## フォルダ構成

`.py` スクリプトは**全て `sim/` 直下に置いたまま動かさない**。多くのスクリプトが
`homeostasis_prototype.py`・`instinct_bias_prototype.py`・`homeostasis_nn_prototype.py`・
`monitor_maturity_prototype.py` などを要件の枠を越えて共通の土台として
`import` しており(合計78箇所・32ファイルが該当)、要件ごとのフォルダへ
分けるとこれらのimportが壊れるため。

結果ファイル(`.png` `.json` `.pkl` `.gif`)だけを、対応する要件のフォルダへ
移してある:

- `req01-04_homeostasis_instinct_legacy/` — 要件1〜4(恒常性センサー・本能バイアス・自己保存本能・レガシー本能)の結果
- `req05_integrated_information/` — 要件5(統合情報理論・Φ)の結果
- `req06_community_signaling/` — 要件6(複数個体の共同体形成・信号創発)の結果
- `req07_self_monitoring/` — 要件7(高階自己モニタリング層)の結果

要件に紐付かないもの(`agent_behavior_animation.py`・`agent_behavior_comparison.gif`)
だけは `sim/` 直下に script と結果を両方置いたままにしてある。

## 動かし方

結果ファイルを読み書きするスクリプトは、実行時のカレントディレクトリを
基準にファイル名を解決する(スクリプト自身の場所は基準にしない)。
そのため、対応する要件フォルダに `cd` してから、`..` 経由でスクリプトを
実行する:

```bash
cd sim/req05_integrated_information
python3 ../iit_phi_prototype.py
```

こうすると、スクリプトは `sim/` 内の共通モジュールを問題なく `import` しつつ
(スクリプト自身の場所からsys.pathを通しているため)、結果ファイルの
読み書きは今いる `req0X` フォルダに対して行われる。

`req01-04` のスクリプトを動かすなら `cd sim/req01-04_homeostasis_instinct_legacy`、
`req06` なら `cd sim/req06_community_signaling`、という要領。

## スクリプト名の接頭辞と要件の対応

| 接頭辞 | 要件 | 備考 |
|---|---|---|
| `homeostasis_prototype.py` | 1〜4(基盤) | センサー恒常性の基本環境。他の多くのスクリプトから共通importされる |
| `homeostasis_nn_*` | 7(非タブラー版) | ニューラルネット移行後のU字型・汎化・grokking検証一式。req07の共通importにもなっている |
| `instinct_bias_*` | 3 | 本能レベルの初期バイアス。ほぼ全スクリプトから共通importされる |
| `self_preservation_*` | 4前半 | 自己保存本能(不可逆な死亡条件) |
| `legacy_*` / `multigen_records*` | 4後半 | レガシー本能(次世代への引き継ぎ)、多世代連鎖、その診断・再解析 |
| `monitor_*` | 7 | 高階自己モニタリング層。`monitor_maturity_prototype.py`は他のmonitor系からも共通import |
| `community_*` / `deviant_*` / `multiagent_*` / `noreward_*` / `recip_*` / `hybrid_*` / `cont_comm_*` / `iterated_*` / `gatecur_*` / `independent_control_summary.json` / `transmission_reanalysis.json` | 6 | 複数個体の共同体形成・信号創発。`community_signal_v2_prototype.py`が主要な共通土台 |
| `iit_*` | 5 | 統合情報理論(Φ)とその代理指標 |
| `nn_weight_connectivity_*` / `nn_activity_pci_*` | 5 | 実在NNへのΦ代理指標の適用 |
| `nn_partA/B/C_summary` / `nn_grok_*` / `nn_intero_*` / `nn_pobs_*` | 7 | `homeostasis_nn_*`系の結果 |
| `nn_selfpres_*` / `nn_multigen_*` / `nn_legacy_*` | 4 | NN版の自己保存・多世代・レガシー系の結果 |
| `nn_comm_*` | 6 | NN版の共同体信号系の結果 |

`nn_` で始まるファイルは上記の通り要件が割れているので、接頭辞だけで
判断せず、対になっているスクリプト名(comparison画像やsummary jsonの
ファイル名の元になったスクリプト)を確認すること。

## 既知の制約: 一部の再解析スクリプトは単体で再実行できない

`legacy_teach_timing_reanalysis.py`・`legacy_elder_and_bias_reanalysis.py`・
`nn_activity_pci_prototype.py` などは、学習済みネットワークの生の状態
(`nn_legacy_split_state_*.pkl`・`nn_legacy_base_*.pkl`・`nn_part*_state_seed*.pkl`等)
を読み込む設計になっているが、これらの中間状態ファイルはサイズが大きく
一度もリポジトリにcommitされていない(gitの履歴を確認済み)。そのため、
これらのスクリプトを実際に動かすには、まず元の学習スクリプト
(`legacy_instinct_nn_splithead_prototype.py`等)を動かして状態を保存し直す
必要がある。これは今回のフォルダ整理より前からの制約であり、整理によって
新たに壊れたものではない(import解決・結果ファイルの読み書き経路は
上記の動かし方で問題なく機能することを確認済み)。
