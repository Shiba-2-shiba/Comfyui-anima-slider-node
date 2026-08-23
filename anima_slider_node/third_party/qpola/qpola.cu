// =========================================================================
// QPOLA カーネルコア v1.0.4 260724 by muooon https://github.com/muooon/QPOLA
// =========================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

// 各種量子化･低ビット型用のヘッダー
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ハードウェアに準ず [ROCm：warp32～64 iPEX：warp16～64]各変数名なども(要修正)
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// =========================================================================
// 型ごとの固有処理を吸収するヘルパー(型トレイト)
// =========================================================================

// 1. メモリからの読込み時：各量子化型からfp32へ変換
__device__ __forceinline__ float to_float(float val)           { return val; }
__device__ __forceinline__ float to_float(__half val)          { return __half2float(val); }
__device__ __forceinline__ float to_float(__nv_bfloat16 val)   { return __bfloat162float(val); }
__device__ __forceinline__ float to_float(signed char val)     { return static_cast<float>(val); }
__device__ __forceinline__ float to_float(__nv_fp8_e4m3 val)   { return static_cast<float>(val); }
__device__ __forceinline__ float to_float(__nv_fp8_e5m2 val)   { return static_cast<float>(val); }
// ＜将来的実装案＞ 将来アンコメントする際：組み込み型によっては static_cast<float> ではなく
// __fp42float(val) or __nv_fp4_e2m12float(val) のような専用変換関数になる可能性があります
/*
__device__ __forceinline__ float to_float(__nv_fp4_e2m1 val)   { return static_cast<float>(val); }
*/

// 2. 型ごとの固有属性を一括管理する型トレイト (Traits)
template <typename T> struct TypeTraits;

template <> struct TypeTraits<float> {
    // fp32 基準･最も精密／LSB(最下位ビット)
    static __device__ __forceinline__ float clamp_max() { return 3.4e38f; }
    static __device__ __forceinline__ float lsb_step()  { return 0.0f; }
    static __device__ __forceinline__ float lim_g_hat() { return 16.0f; }
};

template <> struct TypeTraits<__half> {
    // fp16 有効値 [-65504.0, 65504.0]／LSB(最下位ビット)
    static __device__ __forceinline__ float clamp_max() { return 65504.0f; }
    static __device__ __forceinline__ float lsb_step()  { return 0.0f; }
    static __device__ __forceinline__ float lim_g_hat() { return 8.0f; }
};

template <> struct TypeTraits<__nv_bfloat16> {
    // bf16 有効値 [-65504.0, 65504.0]／LSB(最下位ビット)
    static __device__ __forceinline__ float clamp_max() { return 65504.0f; }
    static __device__ __forceinline__ float lsb_step()  { return 0.0f; }
    static __device__ __forceinline__ float lim_g_hat() { return 8.0f; }
};

template <> struct TypeTraits<signed char> {
    // int8 有効値 [127f] オーバーフロー防止／LSB(最下位ビット)
    static __device__ __forceinline__ float clamp_max() { return 127.0f; }
    static __device__ __forceinline__ float lsb_step()  { return 1.0f; }
    static __device__ __forceinline__ float lim_g_hat() { return 2.0f; }
};

template <> struct TypeTraits<__nv_fp8_e4m3> {
    // fp8 e4m3 有効値 448.0f 厳重ガード／LSB(最下位ビット)
    static __device__ __forceinline__ float clamp_max() { return 448.0f; }
    static __device__ __forceinline__ float lsb_step()  { return 0.0625f; }
    static __device__ __forceinline__ float lim_g_hat() { return 4.0f; }
};

template <> struct TypeTraits<__nv_fp8_e5m2> {
    // fp8 e5m2 有効値 57344.0f 厳重ガード／LSB(最下位ビット)
    static __device__ __forceinline__ float clamp_max() { return 57344.0f; }
    static __device__ __forceinline__ float lsb_step()  { return 0.25f; }
    static __device__ __forceinline__ float lim_g_hat() { return 4.0f; }
};

/*
// fp4 用 TypeTraits ＜将来的実装案＞
template <> struct TypeTraits<__nv_fp4_e2m1> {
    // FP4 の最も高精度な等間隔ゾーンの上限（2.0f）に合わせる
    static __device__ __forceinline__ float clamp_max() { return 6.0f; }
    static __device__ __forceinline__ float lsb_step()  { return 0.5f; } // 最小刻み
    static __device__ __forceinline__ float lim_g_hat() { return 2.0f; }
};
*/

// 3. メモリへの書き戻し時：fp32からターゲット型へキャスト (TypeTraits連動)

// 超軽量･GPU命令数最小の乱数 (0.0f ～ 1.0f 未満)
__device__ __forceinline__ float fast_rand_01() {
    // clock() を使わずスレッドIDのビットシャッフルだけで1サイクル生成
    uint32_t x = (blockIdx.x * blockDim.x + threadIdx.x) * 1664525u + 1013904223u;
    return (x & 0x00FFFFFF) * (1.0f / 16777216.0f);
}

// 低精度量子化型のための汎用確率的丸め (Stochastic Rounding)
template <typename T>
__device__ __forceinline__ float apply_quant_jitter(float val, float conflict) {
    // LSB の 25% (0.25f) を基本振幅とし、conflict (0.0～2.0) で最大 +40% 微増
    float lsb = TypeTraits<T>::lsb_step();
    // LSBの範囲でConflict(アライメント衝突)に応じた微小ジッターを加算
    float jitter_scale = lsb * 0.25f * (1.0f + 0.2f * conflict);
    // (-0.5 〜 +0.5) の乱数を乗算してジッター化
    return val + (fast_rand_01() - 0.5f) * jitter_scale;
}

// 汎用宣言 (fp32、fp16、bf16 従来のまま変更なし)
template <typename T>
__device__ __forceinline__ T from_float(float val, float conflict = 0.0f);

// --- fp32 ---
template <> __device__ __forceinline__ float from_float<float>(float val, float conflict) {
    return val;
}
// --- fp16 ---
template <> __device__ __forceinline__ __half from_float<__half>(float val, float conflict) {
    float max_v = TypeTraits<__half>::clamp_max();
    return __float2half(fmaxf(-max_v, fminf(max_v, val)));
}
// --- bf16 ---
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val, float conflict) {
    float max_v = TypeTraits<__nv_bfloat16>::clamp_max();
    return __float2bfloat16(fmaxf(-max_v, fminf(max_v, val)));
}
// --- int8 ---
template <> __device__ __forceinline__ signed char from_float<signed char>(float val, float conflict) {
    float max_v = TypeTraits<signed char>::clamp_max();
    float clamped = fmaxf(-max_v, fminf(max_v, val));
    return static_cast<signed char>(apply_quant_jitter<signed char>(clamped, conflict));
}
// --- fp8 (e4m3) ---
template <> __device__ __forceinline__ __nv_fp8_e4m3 from_float<__nv_fp8_e4m3>(float val, float conflict) {
    float max_v = TypeTraits<__nv_fp8_e4m3>::clamp_max();
    float clamped = fmaxf(-max_v, fminf(max_v, val));
    return __nv_fp8_e4m3(apply_quant_jitter<__nv_fp8_e4m3>(clamped, conflict));
}
// --- fp8 (e5m2) ---
template <> __device__ __forceinline__ __nv_fp8_e5m2 from_float<__nv_fp8_e5m2>(float val, float conflict) {
    float max_v = TypeTraits<__nv_fp8_e5m2>::clamp_max();
    float clamped = fmaxf(-max_v, fminf(max_v, val));
    return __nv_fp8_e5m2(apply_quant_jitter<__nv_fp8_e5m2>(clamped, conflict));
}

/*
// --- nv_fp4 (e2m1) --- ＜将来的実装案＞
template <> __device__ __forceinline__ __nv_fp4_e2m1 from_float<__nv_fp4_e2m1>(float val, float conflict) {
    float max_v = TypeTraits<__nv_fp4_e2m1>::clamp_max();
    float clamped = fmaxf(-max_v, fminf(max_v, val));
    return __nv_fp4_e2m1(apply_quant_jitter<__nv_fp4_e2m1>(clamped, conflict));
}
*/

// =========================================================================
// QPOLA コアロジック (__device__) 関数にしてカーネルからの呼出し許可
// =========================================================================
template <typename T>
__device__ __forceinline__ void qpola_kernel_impl(
    T* p,
    const T* g,
    float base_lr,
    float eps,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // 1. 境界内かどうかの判定と初期読み込み
    bool is_active = (idx < n);
    float active_count = is_active ? 1.0f : 0.0f;

    // プロトコル化(Coalesced Access / Contiguous)
    float g_val = is_active ? to_float(g[idx]) : 0.0f;
    float p_val = is_active ? to_float(p[idx]) : 0.0f;

    // NaN / Inf ガード
    if (isnan(g_val) || isinf(g_val)) g_val = 0.0f;
    if (isnan(p_val) || isinf(p_val)) p_val = 0.0f;

    // 2. 勾配の方向(符号)と局所スケールの抽出
    float g_sign = (g_val > 0.0f) ? 1.0f : ((g_val < 0.0f) ? -1.0f : 0.0f);

    float warp_direction_sum = g_sign;
    float warp_g_abs_sum     = fabsf(g_val);

    // ワープシャッフルによる 32 スレッド並列集計(重み p は集計しない)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        active_count        += __shfl_down_sync(0xffffffff, active_count, offset);
        warp_direction_sum += __shfl_down_sync(0xffffffff, warp_direction_sum, offset);
        warp_g_abs_sum     += __shfl_down_sync(0xffffffff, warp_g_abs_sum, offset);
    }

    // ブロードキャスト
    float warp_active_count   = __shfl_sync(0xffffffff, active_count, 0);
    float micro_direction_sum = __shfl_sync(0xffffffff, warp_direction_sum, 0);
    float warp_g_scale_sum    = __shfl_sync(0xffffffff, warp_g_abs_sum, 0);

    float div_count = fmaxf(warp_active_count, 1.0f);
    float micro_direction_mean = micro_direction_sum / div_count;
    float warp_g_scale         = warp_g_scale_sum / div_count; // 周囲の勾配平均スケール

    // 3. マクロ集計(共有メモリによる 256 スレッドブロック集計)
    __shared__ float s_block_direction[BLOCK_SIZE / WARP_SIZE];
    __shared__ float s_block_active[BLOCK_SIZE / WARP_SIZE];

    if (lane == 0) {
        s_block_direction[warp_id] = micro_direction_sum;
        s_block_active[warp_id]    = warp_active_count;
    }

    __syncthreads();

    float block_direction_sum = 0.0f;
    float block_active_sum    = 0.0f;
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE / WARP_SIZE; ++i) {
        block_direction_sum += s_block_direction[i];
        block_active_sum    += s_block_active[i];
    }

    float macro_direction_mean = block_direction_sum / fmaxf(block_active_sum, 1.0f);

    // 範囲外スレッドの保護(同期待ち完了後)
    if (!is_active) return;

    // 4. 空間アライメント(一致度) / Conflict(衝突度) 算出
    float micro_align = g_sign * micro_direction_mean;
    float macro_align = g_sign * macro_direction_mean;

    float diff_micro = fmaxf(0.0f, 1.0f - micro_align);
    float diff_macro = fmaxf(0.0f, 1.0f - macro_align);
    float conflict   = (diff_micro + diff_macro) * 0.5f; // 0.0(完全一致) ～ 2.0(完全反転)

    // 5. 減衰係数の算出(途中で変数の意味を変えず1step計算)
    constexpr float min_factor = 0.01f;
    constexpr float decay_rate = (1.0f - min_factor) * 0.5f;

    // Conflict に応じ 1.0 〜 min_factor の間で素直にクランプ
    float raw_adaptation = 1.0f - conflict * decay_rate;
    float adaptation_factor = fminf(fmaxf(raw_adaptation, min_factor), 1.0f);

    // 6. 勾配の無次元化 と 純粋な無次元更新
    // g_hat：単位の削ぎ落とされた平均 1.0 前後の正規化勾配
    // 突起値(Outlier) 32.0 等の跳ね上がり時に最大値 lim(2.0～16.0）で滑らかに飽和(サチュレート)
    float lim = TypeTraits<T>::lim_g_hat();
    float g_hat_raw = g_val / (warp_g_scale + eps);
    float g_hat = tanhf(g_hat_raw / lim) * lim; // 無次元化勾配を型ごとの上限で安全に飽和

    // 引き算構造により p = 0.0 (LoRA) も 1step から LR の歩幅で自然脱出
    // 勾配 g_val が 0 になれば自然に 0 へ定着(0スパース維持)
    float next_p = p_val - (base_lr * g_hat * adaptation_factor);

    // 書き戻し時の安全性保持
    if (isnan(next_p) || isinf(next_p)) next_p = p_val;
    p[idx] = from_float<T>(next_p, conflict);
}

// =========================================================================
// Python (ctypes) 側からデータ型に応じて明示的に呼び分ける外部エントリーポイント
// =========================================================================
extern "C" {

// マクロでエントリーポイントの定義をパターン化
#define DEFINE_QPOLA_ENTRY(SUFFIX, TYPE) \
__global__ void qpola_kernel_##SUFFIX(TYPE* p, const TYPE* g, float base_lr, float eps, int n) { \
    qpola_kernel_impl<TYPE>(p, g, base_lr, eps, n); \
}

// 1行ずつスッキリ宣言！
DEFINE_QPOLA_ENTRY(fp32,     float)
DEFINE_QPOLA_ENTRY(fp16,     __half)
DEFINE_QPOLA_ENTRY(bf16,     __nv_bfloat16)
DEFINE_QPOLA_ENTRY(int8,     signed char)
DEFINE_QPOLA_ENTRY(fp8_e4m3, __nv_fp8_e4m3)
DEFINE_QPOLA_ENTRY(fp8_e5m2, __nv_fp8_e5m2)

#undef DEFINE_QPOLA_ENTRY

} // extern "C"
