#!/bin/bash
# FA kernel 性能剖析一键脚本：msopprof 双模式 × 双算子（fa_test.py）
#
# 用法：
#   bash perf/profile/run_test.sh                              # prof + 我们的 SplitB
#   bash perf/profile/run_test.sh --test-torch                # prof + torch_npu baseline
#   bash perf/profile/run_test.sh sim                         # sim + SplitB
#   bash perf/profile/run_test.sh sim --test-torch            # sim + torch_npu
#   bash perf/profile/run_test.sh --batch 64 --seqlen 96      # 透传 fa_test.py 参数
#
# --kernel-name 自动适配：
#   我们的算子 → *SplitB*（匹配 FAInferSplitB<half, ...>）
#   torch_npu  → *FlashAttention*（实测 910B4: FlashAttentionScore_<hash>_mix_aic；
#                glob 大小写敏感，*attention* 小写匹配不到）
#
# 输出：perf/results/profile/{prof|sim}/（我们）或 {prof|sim}_torch/（torch 基线）

set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -P "${SCRIPT_DIR}/../.." && pwd)"


MODE="prof"
if [ "${1:-}" = "sim" ] || [ "${1:-}" = "prof" ]; then
    MODE="$1"
    shift
fi

OUT_DIR="${REPO_ROOT}/perf/results/profile/${MODE}"
PYTHON="${PYTHON:-python}"
MSOPPROF="$(command -v msopprof 2>/dev/null || echo /usr/local/Ascend/cann-9.0.0/bin/msopprof)"

# --test-torch 透传 + kernel-name/输出目录自动适配
FA_ARGS=("$@")
TEST_TORCH=false
for a in "$@"; do
    if [ "$a" = "--test-torch" ]; then
        TEST_TORCH=true
    fi
done

# torch 基线结果单独归档，避免与我们的算子混在一起
if [ "$TEST_TORCH" = true ]; then
    OUT_DIR="${REPO_ROOT}/perf/results/profile/${MODE}_torch"
fi

if [ -n "${KERNEL_NAME:-}" ]; then
    KN="${KERNEL_NAME}"
elif [ "$TEST_TORCH" = true ]; then
    KN="*FlashAttention*"
else
    KN="*SplitB*"
    # KN="*FAInferKernel*"
fi

# 注意：不删除 OUT_DIR——msopprof 每次运行会在其下自建子目录
mkdir -p "${OUT_DIR}"

echo "[run_test] mode=${MODE} test_torch=${TEST_TORCH} kernel=${KN} → ${OUT_DIR}"
echo "[run_test] fa_test args: ${FA_ARGS[*]:-(none)}"

if [ "${MODE}" = "sim" ]; then
    "${MSOPPROF}" simulator \
        --kernel-name="${KN}" \
        --soc-version=Ascend910B4 \
        --timeout=10 \
        --output="${OUT_DIR}" \
        "${PYTHON}" "${SCRIPT_DIR}/fa_test.py" "${FA_ARGS[@]}"
else
    if [ -n "${AIC_METRICS:-}" ]; then
        "${MSOPPROF}" \
            --kernel-name="${KN}" \
            --aic-metrics="${AIC_METRICS}" \
            --output="${OUT_DIR}" \
            "${PYTHON}" "${SCRIPT_DIR}/fa_test.py" "${FA_ARGS[@]}"
    else
        echo "[run_test] aic-metrics: Roofline,Occupancy,MemoryDetail"
        "${MSOPPROF}" \
            --kernel-name="${KN}" \
            --aic-metrics=Roofline,Occupancy,MemoryDetail \
            --output="${OUT_DIR}" \
            "${PYTHON}" "${SCRIPT_DIR}/fa_test.py" "${FA_ARGS[@]}"
    fi
fi

echo "[run_test] 完成 → ${OUT_DIR}"
