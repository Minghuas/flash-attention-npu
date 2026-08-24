#!/bin/bash
# 清理残留的 test_splitb_s3.py 相关进程（挂死测试会占用 NPU 卡，影响后续运行——devlog #21）
# 用法：bash debug/kill_test.sh          （默认清理 test_splitb_s3.py）
#       bash debug/kill_test.sh 关键字    （清理其它测试脚本，如 test_flash_attn_npu_v2.py）
KEYWORD="${1:-splitb}"

echo "== 查找含 '$KEYWORD' 的进程 =="
PIDS=$(pgrep -f "$KEYWORD")
if [ -z "$PIDS" ]; then
    echo "无残留进程"
else
    echo "发现: $PIDS"
    ps -fp $PIDS | head -8
    echo "== kill -9 =="
    kill -9 $PIDS 2>/dev/null
    sleep 1
    REMAIN=$(pgrep -f "$KEYWORD")
    if [ -z "$REMAIN" ]; then
        echo "已全部清理"
    else
        echo "仍残留: $REMAIN（可能有子进程，逐个强杀）"
        pkill -9 -f "$KEYWORD"
        sleep 1
        pgrep -f "$KEYWORD" >/dev/null && echo "警告：仍有残留，请检查 ps -ef | grep $KEYWORD" || echo "已全部清理"
    fi
fi

# 顺带提示各 NPU 卡上的残留占用（python 主进程 kill 后设备一般自动释放）
echo "== 各卡进程概览（host 集团命令，无输出即干净）=="
npu-smi info 2>/dev/null | grep -E "Process|python" | head -10 || true
