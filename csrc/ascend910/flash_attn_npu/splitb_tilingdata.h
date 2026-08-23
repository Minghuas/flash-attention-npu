/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB（大 Batch 小 SeqLen 前向模板）的 tiling 数据结构。
 * 照搬自 ops-transformer FlashAttentionScoreTilingB 的 tiling 字段（v3 catlass 路线）：
 *   - InputParams / CoreParams / MultiCoreParams 三段
 *   - v3 变更（相对范式 A 归档版）：删 TCubeTiling×2 与 SoftMaxTiling——catlass BlockMmad
 *     自管 L1/L0，online_softmax.hpp 用 UB 内建参数，均无需 host 侧 cube/softmax tiling
 *   - 核间切分 aic 基数（FAInfer 同款），非参考的 aiv 基数（D7 因地制宜项）
 * 设计文档：perf/design/splitb_integration.md（v3）
 */

#ifndef FLASH_ATTN_NPU_SPLITB_TILINGDATA_H
#define FLASH_ATTN_NPU_SPLITB_TILINGDATA_H

#include <cstdint>

namespace SplitB {

// ============ InputParams（照搬参考，裁剪） ============
class SplitBInputParams {
public:
    // 基础形状：5D 语义 [B, N2, G, S1, S2]，N2=kv头数，G=q头/kv头（GQA 组数）
    int64_t bSize = 0;
    int64_t n2Size = 0;
    int64_t gSize = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t alignedS2 = 0;
    int64_t dSize = 0;
    uint32_t blockSize = 0;       // paged KV 页大小（FAInfer tiling 同名字段；非 paged 路径仅占位穿透）
    float scaleValue = 0.0f;
    float softcapValue = 0.0f;      // 我方特性：Vec1 softmax 前施加（HAS_SOFTCAP 编译期门控）
    int64_t windowSizeLeft = 0;     // SWA 窗口（FAInfer 语义，-1 = 无界）
    int64_t windowSizeRight = 0;
    uint8_t isCausalFlag = 0;       // 调试/分支辅助（mask 形态由模板参数 MASK_TYPE 决定）
    uint8_t debugFlag = 0;          // 设备 printf 探针开关（FLASH_ATTN_SPLITB_DEBUG=1，devlog #42）
    uint8_t dumpFlag = 0;           // 设备 DumpTensor 探针开关（FLASH_ATTN_SPLITB_DUMP=1，devlog #44）
    uint8_t softmaxOnly = 0;        // 只跑段1+2（FLASH_ATTN_SPLITB_SOFTMAX_ONLY=1，devlog #44.25）
    uint8_t remain[3] = {0};

    int64_t get_bSize() const { return bSize; }
    int64_t get_n2Size() const { return n2Size; }
    int64_t get_gSize() const { return gSize; }
    int64_t get_s1Size() const { return s1Size; }
    int64_t get_s2Size() const { return s2Size; }
    int64_t get_alignedS2() const { return alignedS2; }
    int64_t get_dSize() const { return dSize; }
    uint32_t get_blockSize() const { return blockSize; }
    float get_scaleValue() const { return scaleValue; }
    float get_softcapValue() const { return softcapValue; }
    int64_t get_windowSizeLeft() const { return windowSizeLeft; }
    int64_t get_windowSizeRight() const { return windowSizeRight; }
    uint8_t get_isCausalFlag() const { return isCausalFlag; }
    uint8_t get_debugFlag() const { return debugFlag; }
    uint8_t get_dumpFlag() const { return dumpFlag; }
    uint8_t get_softmaxOnly() const { return softmaxOnly; }

    void set_bSize(int64_t v) { bSize = v; }
    void set_n2Size(int64_t v) { n2Size = v; }
    void set_gSize(int64_t v) { gSize = v; }
    void set_s1Size(int64_t v) { s1Size = v; }
    void set_s2Size(int64_t v) { s2Size = v; }
    void set_alignedS2(int64_t v) { alignedS2 = v; }
    void set_dSize(int64_t v) { dSize = v; }
    void set_blockSize(uint32_t v) { blockSize = v; }
    void set_scaleValue(float v) { scaleValue = v; }
    void set_softcapValue(float v) { softcapValue = v; }
    void set_windowSizeLeft(int64_t v) { windowSizeLeft = v; }
    void set_windowSizeRight(int64_t v) { windowSizeRight = v; }
    void set_isCausalFlag(uint8_t v) { isCausalFlag = v; }
    void set_debugFlag(uint8_t v) { debugFlag = v; }
    void set_dumpFlag(uint8_t v) { dumpFlag = v; }
    void set_softmaxOnly(uint8_t v) { softmaxOnly = v; }
};

// ============ CoreParams（照搬参考，裁剪） ============
class SplitBCoreParams {
public:
    int32_t s1BaseSize = 0;        // Vec1 softmax 行块（参考公式：8K元素/s2 对齐16，≤256）
    int32_t s1BaseTailSize = 0;
    int64_t s1OuterSize = 0;
    int32_t s1Vec2BaseSize = 0;    // Vec2 行块
    int32_t s1Vec2BaseTailSize = 0;
    int64_t s1Vec2OuterSize = 0;
    int32_t s2BaseSize = 0;        // = alignedS2（S2 不切分，核心）
    int32_t s2BaseTailSize = 0;
    int64_t s2OuterSize = 0;       // 恒 1
    int32_t dBaseSize = 0;         // = alignedD（D 不切分）
    int32_t dBaseTailSize = 0;
    int64_t dOuterSize = 0;        // 恒 1
    int32_t bBaseSize = 0;         // 恒 1：每个 boIdx = 1 个 batch 的全部头
    int32_t bBaseTailSize = 0;
    int64_t bOuterSize = 0;        // = B
    int32_t remain0 = 0;
    int64_t remain1 = 0;

    int32_t get_s1BaseSize() const { return s1BaseSize; }
    int32_t get_s1BaseTailSize() const { return s1BaseTailSize; }
    int64_t get_s1OuterSize() const { return s1OuterSize; }
    int32_t get_s1Vec2BaseSize() const { return s1Vec2BaseSize; }
    int32_t get_s1Vec2BaseTailSize() const { return s1Vec2BaseTailSize; }
    int64_t get_s1Vec2OuterSize() const { return s1Vec2OuterSize; }
    int32_t get_s2BaseSize() const { return s2BaseSize; }
    int32_t get_s2BaseTailSize() const { return s2BaseTailSize; }
    int64_t get_s2OuterSize() const { return s2OuterSize; }
    int32_t get_dBaseSize() const { return dBaseSize; }
    int32_t get_dBaseTailSize() const { return dBaseTailSize; }
    int64_t get_dOuterSize() const { return dOuterSize; }
    int32_t get_bBaseSize() const { return bBaseSize; }
    int32_t get_bBaseTailSize() const { return bBaseTailSize; }
    int64_t get_bOuterSize() const { return bOuterSize; }

    void set_s1BaseSize(int32_t v) { s1BaseSize = v; }
    void set_s1BaseTailSize(int32_t v) { s1BaseTailSize = v; }
    void set_s1OuterSize(int64_t v) { s1OuterSize = v; }
    void set_s1Vec2BaseSize(int32_t v) { s1Vec2BaseSize = v; }
    void set_s1Vec2BaseTailSize(int32_t v) { s1Vec2BaseTailSize = v; }
    void set_s1Vec2OuterSize(int64_t v) { s1Vec2OuterSize = v; }
    void set_s2BaseSize(int32_t v) { s2BaseSize = v; }
    void set_s2BaseTailSize(int32_t v) { s2BaseTailSize = v; }
    void set_s2OuterSize(int64_t v) { s2OuterSize = v; }
    void set_dBaseSize(int32_t v) { dBaseSize = v; }
    void set_dBaseTailSize(int32_t v) { dBaseTailSize = v; }
    void set_dOuterSize(int64_t v) { dOuterSize = v; }
    void set_bBaseSize(int32_t v) { bBaseSize = v; }
    void set_bBaseTailSize(int32_t v) { bBaseTailSize = v; }
    void set_bOuterSize(int64_t v) { bOuterSize = v; }
};

// ============ MultiCoreParams（aic 基数，FAInfer 同款；参考为 aiv 基数） ============
class SplitBMultiCoreParams {
public:
    int32_t coreNum = 0;          // 实际使用的核数（≤ aicNum）
    int32_t reserve = 0;
    int64_t totalSize = 0;        // = bOuterSize = B
    int64_t splitFactorSize = 0;  // 每核分到的 batch 数
    int64_t splitFactorTailSize = 0;

    int32_t get_coreNum() const { return coreNum; }
    int64_t get_totalSize() const { return totalSize; }
    int64_t get_splitFactorSize() const { return splitFactorSize; }
    int64_t get_splitFactorTailSize() const { return splitFactorTailSize; }

    void set_coreNum(int32_t v) { coreNum = v; }
    void set_totalSize(int64_t v) { totalSize = v; }
    void set_splitFactorSize(int64_t v) { splitFactorSize = v; }
    void set_splitFactorTailSize(int64_t v) { splitFactorTailSize = v; }
};

// ============ 总结构（host 填充 → GM 传入 → kernel 只读） ============
class SplitBTilingData {
public:
    SplitBInputParams inputParams;
    SplitBCoreParams coreParams;
    SplitBMultiCoreParams multiCoreParams;
};

} // namespace SplitB

#endif // FLASH_ATTN_NPU_SPLITB_TILINGDATA_H
