#!/bin/bash
# ダブルクリックで実行する: git push origin main
# ローカルのcommitをGitHub(公開リポジトリ)へ反映する。
# このスクリプト自体は確認を挟まないので、push対象は実行前に確認しておくこと。

cd "$(dirname "$0")" || exit 1

echo "=== git push (origin main) ==="
echo "作業ディレクトリ: $(pwd)"
echo ""
echo "--- 現在のブランチとpush対象のコミット ---"
git status -sb
echo ""
git log --oneline origin/main..HEAD
echo ""
echo "上記のコミットをGitHub(origin/main、公開リポジトリ)にpushします。"
echo ""

git push origin main
STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "=== push成功 ==="
else
    echo "=== push失敗(終了コード $STATUS)。上のログを確認してください ==="
fi
echo ""
echo "何かキーを押すとこのウィンドウを閉じます..."
read -n 1 -s
