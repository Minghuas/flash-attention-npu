/**
 * Copyright (c) 2026, perf-shortSeqLargeBatch project.
 *
 * SplitB（大 Batch 小 SeqLen 前向模板）的 tiling 数据结构。
 * 照搬自 ops-transformer flash_attention_score_tiling.h 的
 * FlashAttentionScoreGeneralTilingData，裁剪到 v1 需要的字段：
 *   - InputParams / CoreParams / MultiCoreParams 三段 + bmm1/bmm2 的 TCubeTiling
 *     + SoftMaxTiling（SoftmaxFlashV2 用）
 *   - 去掉 pse/dropmask/sparse/rope 等 v1 未启用特性的字段（HAS_ALIBI 等接入时再补）
 * 设计文档：perf/design/splitb_integration.md；参考解读：
 * perf/analysis/reference_splitb_deep_dive.md。
 */

#ifndef FLASH_ATTN_NPU_SPLITB_TILINGDATA_H
#define FLASH_ATTN_NPU_SPLITB_TILINGDATA_H

#include <cstdint>
// TCubeTiling / SoftMaxTiling 来自 CANN highlevel_api（setup.py 已加 include 路径）
#include "kernel_tiling/kernel_tiling.h"

namespace SplitB {

// TCubeTiling / SoftMaxTiling 位于 AscendC::tiling（kernel_tiling.h），域内别名
using AscendC::tiling::TCubeTiling;
using AscendC::tiling::SoftMaxTiling;

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
    float scaleValue = 0.0f;
    float softcapValue = 0.0f;      // 我方特性：参考实现无 softcap，Vec1 softmax 前施加
    // attenMask（我方 causal/SWA 翻译为参考的 mask 形态后走参考的 mask 处理代码）
    uint32_t attenMaskS2Size = 0;   // mask 的行宽（NO_COMPRESS SS 形态下 = alignedS2）
    uint8_t attenMaskShapeType = 0; // 0:(B,N2,G,S1,S2) 1:(B,1,1,S1,S2) 2:(1,1,1,S1,S2) —— v1 用 2
    uint8_t attenMaskCompressMode = 0; // 参考 AttenMaskCompressMode；v1: 0=NO_COMPRESS
    uint8_t layoutType = 0;         // v1 固定 BSND（参考 LayOutTypeEnum::LAYOUT_BSH）
    uint8_t isCausalFlag = 0;       // 我方语义：causal=true 时 mask 偏移走右下因果推导
    uint8_t remain[5] = {0};

    int64_t get_bSize() const { return bSize; }
    int64_t get_n2Size() const { return n2Size; }
    int64_t get_gSize() const { return gSize; }
    int64_t get_s1Size() const { return s1Size; }
    int64_t get_s2Size() const { return s2Size; }
    int64_t get_alignedS2() const { return alignedS2; }
    int64_t get_dSize() const { return dSize; }
    float get_scaleValue() const { return scaleValue; }
    float get_softcapValue() const { return softcapValue; }
    uint32_t get_attenMaskS2Size() const { return attenMaskS2Size; }
    uint8_t get_attenMaskShapeType() const { return attenMaskShapeType; }
    uint8_t get_attenMaskCompressMode() const { return attenMaskCompressMode; }
    uint8_t get_isCausalFlag() const { return isCausalFlag; }
    uint8_t get_layoutType() const { return layoutType; }

    void set_bSize(int64_t v) { bSize = v; }
    void set_n2Size(int64_t v) { n2Size = v; }
    void set_gSize(int64_t v) { gSize = v; }
    void set_s1Size(int64_t v) { s1Size = v; }
    void set_s2Size(int64_t v) { s2Size = v; }
    void set_alignedS2(int64_t v) { alignedS2 = v; }
    void set_dSize(int64_t v) { dSize = v; }
    void set_scaleValue(float v) { scaleValue = v; }
    void set_softcapValue(float v) { softcapValue = v; }
    void set_attenMaskS2Size(uint32_t v) { attenMaskS2Size = v; }
    void set_attenMaskShapeType(uint8_t v) { attenMaskShapeType = v; }
    void set_attenMaskCompressMode(uint8_t v) { attenMaskCompressMode = v; }
    void set_isCausalFlag(uint8_t v) { isCausalFlag = v; }
    void set_layoutType(uint8_t v) { layoutType = v; }
};

// ============ CoreParams（照搬参考，裁剪） ============
class SplitBCoreParams {
public:
    int32_t s1BaseSize = 0;        // Vec1 softmax 行块
    int32_t s1BaseTailSize = 0;
    int64_t s1OuterSize = 0;
    int32_t s1Vec2BaseSize = 0;    // Vec2 行块
    int32_t s1Vec2BaseTailSize = 0;
    int64_t s1Vec2OuterSize = 0;
    int32_t s2BaseSize = 0;        // = alignedS2（S2 不切分）
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

// ============ MultiCoreParams（照搬参考，裁剪） ============
class SplitBMultiCoreParams {
public:
    int32_t coreNum = 0;          // 实际使用的切片数（AIV 单位，≤ aivNum）
    int32_t reserve = 0;
    int64_t totalSize = 0;        // = bOuterSize = B
    int64_t splitFactorSize = 0;  // 每个切片分到的 batch 数
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
    TCubeTiling bmm1TilingData;             // QK^T 的 cube tiling（host 用 MatmulApiTiling 生成）
    TCubeTiling bmm2TilingData;             // PV 的 cube tiling
    SoftMaxTiling softmaxFlashTilingData;   // SoftmaxFlashV2 tiling（host 用 TilingFunc 生成）
};

} // namespace SplitB

#endif // FLASH_ATTN_NPU_SPLITB_TILINGDATA_H
