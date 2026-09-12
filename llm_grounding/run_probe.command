#!/bin/bash
# ダブルクリックで実行するMLX速度実測プローブ。
# 1. mlx, mlx-lm をインストール(externally-managed-environmentなら.venvへ切替)
# 2. run_mlx_speed_probe.py --policy mlx --episodes 20 を実行
# 3. 結果を mlx_speed_probe_results.json に保存
# 終了後もウィンドウは閉じず、何かキーを押すまで表示したままにする。

cd "$(dirname "$0")" || exit 1

echo "=== MLX速度実測プローブ ==="
echo "作業ディレクトリ: $(pwd)"
echo ""

PYTHON=python3
PIP="python3 -m pip"

echo "--- mlx, mlx-lm のインストールを試みます(システムのpython3) ---"
INSTALL_LOG="/tmp/run_probe_pip_install.log"
if $PIP install mlx mlx-lm > "$INSTALL_LOG" 2>&1; then
    echo "インストール成功(システム環境)"
else
    if grep -qi "externally-managed-environment" "$INSTALL_LOG"; then
        echo "externally-managed-environment エラーを検出。.venv を作成して切り替えます。"
        python3 -m venv .venv
        if [ ! -f ".venv/bin/python3" ]; then
            echo "エラー: .venv の作成に失敗しました。"
            echo ""
            echo "何かキーを押すと終了します..."
            read -n 1 -s
            exit 1
        fi
        PYTHON="$(pwd)/.venv/bin/python3"
        PIP="$(pwd)/.venv/bin/python3 -m pip"
        $PIP install --upgrade pip > /tmp/run_probe_pip_upgrade.log 2>&1
        if $PIP install mlx mlx-lm > "$INSTALL_LOG" 2>&1; then
            echo ".venv 内へのインストール成功"
        else
            echo "エラー: .venv内でのインストールにも失敗しました。ログ:"
            cat "$INSTALL_LOG"
            echo ""
            echo "何かキーを押すと終了します..."
            read -n 1 -s
            exit 1
        fi
    else
        echo "エラー: pip install が想定外の理由で失敗しました。ログ:"
        cat "$INSTALL_LOG"
        echo ""
        echo "何かキーを押すと終了します..."
        read -n 1 -s
        exit 1
    fi
fi

echo ""
echo "使用するpython: $PYTHON"
echo ""
echo "--- 速度実測プローブを実行します(20エピソード、初回はモデルダウンロードで時間がかかります) ---"
echo ""

$PYTHON run_mlx_speed_probe.py --policy mlx --episodes 20 --out mlx_speed_probe_results.json
STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "=== 完了。mlx_speed_probe_results.json に保存しました ==="
else
    echo "=== エラー終了(終了コード $STATUS)。上のログを確認してください ==="
fi
echo ""
echo "何かキーを押すとこのウィンドウを閉じます..."
read -n 1 -s
