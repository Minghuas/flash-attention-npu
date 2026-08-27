#!/bin/bash
# 清理占用 NPU 的进程（支持按关键字过滤或全量清理）
# 用法:
#   bash debug/kill_test.sh               # 默认清理含 "splitb" 的进程
#   bash debug/kill_test.sh -k flash      # 清理含 "flash" 的进程
#   bash debug/kill_test.sh -a            # 清理所有占用 NPU 的进程（危险！）
#   bash debug/kill_test.sh -h            # 显示帮助

set -euo pipefail  # 加强错误检测

SCRIPT_NAME=$(basename "$0")
HELP_MSG="用法: $SCRIPT_NAME [选项]
选项:
  -k, --keyword KEYWORD   按关键字清理进程（默认: splitb）
  -a, --all               清理所有占用 NPU 的进程（危险操作，需二次确认）
  -f, --force             配合 -a 跳过确认（慎用）
  -h, --help              显示此帮助信息
示例:
  $SCRIPT_NAME -k attn    杀死命令行含 'attn' 的进程
  $SCRIPT_NAME -a         杀死所有 NPU 进程（需确认）"

# 默认参数
KEYWORD="splitb"
MODE="keyword"   # 可选 keyword / all
FORCE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -k|--keyword)
            KEYWORD="$2"
            MODE="keyword"
            shift 2
            ;;
        -a|--all)
            MODE="all"
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "$HELP_MSG"
            exit 0
            ;;
        -*)
            echo "未知选项: $1" >&2
            echo "$HELP_MSG" >&2
            exit 1
            ;;
        *)  # 向后兼容：直接传关键字（不带 -k）
            KEYWORD="$1"
            MODE="keyword"
            shift
            ;;
    esac
done

# ---------- 函数：安全杀死进程 ----------
kill_with_grace() {
    local pid_list=("$@")
    if [[ ${#pid_list[@]} -eq 0 ]]; then
        return 0
    fi

    echo ">>> 准备终止进程: ${pid_list[*]}"
    # 先 SIGTERM
    kill -15 "${pid_list[@]}" 2>/dev/null || true
    sleep 2

    # 检查剩余进程
    local remaining=()
    for pid in "${pid_list[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            remaining+=("$pid")
        fi
    done

    if [[ ${#remaining[@]} -gt 0 ]]; then
        echo ">>> 部分进程未响应 SIGTERM，强制 kill -9: ${remaining[*]}"
        kill -9 "${remaining[@]}" 2>/dev/null || true
        sleep 1
        # 再次检查
        local still_alive=()
        for pid in "${remaining[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                still_alive+=("$pid")
            fi
        done
        if [[ ${#still_alive[@]} -gt 0 ]]; then
            echo "⚠️ 警告：以下进程仍然存活（可能为内核线程或僵尸）: ${still_alive[*]}" >&2
            echo "   请手动检查: ps -fp ${still_alive[*]}" >&2
            return 1
        fi
    fi
    echo ">>> 成功终止 ${#pid_list[@]} 个进程"
    return 0
}

# ---------- 模式1：按关键字清理 ----------
clean_by_keyword() {
    echo "== 查找含关键字 '$KEYWORD' 的进程 =="
    local pids
    pids=$(pgrep -f "$KEYWORD" | grep -v "$$" || true)  # 排除脚本自身

    if [[ -z "$pids" ]]; then
        echo "无匹配进程"
        return 0
    fi

    echo "发现 PID: $pids"
    ps -fp $pids 2>/dev/null | head -10 || true

    # 转为数组
    mapfile -t pid_arr <<< "$pids"
    kill_with_grace "${pid_arr[@]}"

    # 额外清理子进程（防止孤儿）
    echo ">>> 清理可能残留的子进程..."
    pkill -P $pids 2>/dev/null || true
    sleep 1
    local remain
    remain=$(pgrep -f "$KEYWORD" | grep -v "$$" || true)
    if [[ -n "$remain" ]]; then
        echo "⚠️ 仍有残留进程: $remain，尝试强制清理..."
        kill -9 $remain 2>/dev/null || true
    fi
}

# ---------- 模式2：全量清理 NPU 进程 ----------
clean_all_npu() {
    # 检查 npu-smi 是否可用
    if ! command -v npu-smi &>/dev/null; then
        echo "错误: npu-smi 命令未找到，无法获取 NPU 进程列表" >&2
        return 1
    fi

    echo "== 正在获取所有占用 NPU 的进程 =="
    # 解析 npu-smi info 输出，提取 PID（兼容不同格式）
    # 典型输出列: "Process ID" 或 "PID"，通常位于第二列
    # 使用 awk 匹配包含数字的行，并提取第一个数字字段
    local raw_output
    raw_output=$(npu-smi info 2>/dev/null | grep -E "Process|python|PID" | grep -oE '[0-9]{4,}' || true)

    if [[ -z "$raw_output" ]]; then
        echo "未检测到任何占用 NPU 的进程（或解析失败）"
        echo "提示: 请手动执行 'npu-smi info' 查看输出格式，若解析异常可调整本脚本的过滤规则"
        return 0
    fi

    # 去重，排除脚本自身 PID
    local pids
    pids=$(echo "$raw_output" | sort -u | grep -v "^$$$" | tr '\n' ' ')
    if [[ -z "$pids" ]]; then
        echo "无有效进程（已排除自身）"
        return 0
    fi

    echo "找到的 NPU 进程 PID: $pids"
    ps -fp $pids 2>/dev/null | head -20 || true

    # 全量清理属于危险操作，需二次确认
    if [[ "$FORCE" != true ]]; then
        read -p "⚠️  即将杀死以上所有 NPU 进程，继续？(y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "操作已取消"
            return 0
        fi
    fi

    mapfile -t pid_arr <<< "$pids"
    kill_with_grace "${pid_arr[@]}"

    # 额外清理这些进程可能产生的子进程（通过 pkill -P 递归）
    echo ">>> 清理子进程..."
    for pid in "${pid_arr[@]}"; do
        pkill -P "$pid" 2>/dev/null || true
    done
    sleep 1

    # 最后验证是否还有残留
    local remain
    remain=$(npu-smi info 2>/dev/null | grep -E "Process|python|PID" | grep -oE '[0-9]{4,}' | sort -u | grep -v "^$$$" || true)
    if [[ -n "$remain" ]]; then
        echo "⚠️ 警告：仍有进程占用 NPU: $remain" >&2
        echo "   可能为系统服务或新启动的进程，请手动检查" >&2
    else
        echo "✅ 所有 NPU 进程已清理干净"
    fi
}

# ---------- 主逻辑 ----------
case "$MODE" in
    keyword)
        clean_by_keyword
        ;;
    all)
        clean_all_npu
        ;;
    *)
        echo "内部错误：未知模式" >&2
        exit 1
        ;;
esac

# 顺带显示当前 NPU 状态（可选）
echo "== 当前 NPU 卡状态（若有残留将显示）=="
npu-smi info 2>/dev/null | grep -E "Process|python" | head -10 || echo "无占用信息（或 npu-smi 未输出）"