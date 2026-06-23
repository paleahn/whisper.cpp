#pragma once
#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * x, T * y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
#if GGML_CUDA_HAS_BF16
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;
#else
typedef void (*to_bf16_cuda_t)(const void *, void *, int64_t, cudaStream_t);
#endif // GGML_CUDA_HAS_BF16

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
#if GGML_CUDA_HAS_BF16
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;
#else
typedef void (*to_bf16_nc_cuda_t)(const void *, void *,
    int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, cudaStream_t);
#endif // GGML_CUDA_HAS_BF16

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

template<typename dst_t, typename src_t>
struct ggml_cuda_cast_impl {
    __host__ __device__ static inline dst_t cast(src_t x) {
        return float(x);
    }
};

template<typename T>
struct ggml_cuda_cast_impl<T, T> {
    __host__ __device__ static inline T cast(T x) {
        return x;
    }
};

template<typename src_t>
struct ggml_cuda_cast_impl<int32_t, src_t> {
    __host__ __device__ static inline int32_t cast(src_t x) {
        return int32_t(x);
    }
};

template<>
struct ggml_cuda_cast_impl<int32_t, int32_t> {
    __host__ __device__ static inline int32_t cast(int32_t x) {
        return x;
    }
};

template<>
struct ggml_cuda_cast_impl<half2, float2> {
    __host__ __device__ static inline half2 cast(float2 x) {
        return __float22half2_rn(x);
    }
};

#if GGML_CUDA_HAS_BF16
template<typename src_t>
struct ggml_cuda_cast_impl<nv_bfloat16, src_t> {
    __host__ __device__ static inline nv_bfloat16 cast(src_t x) {
        return __float2bfloat16(float(x));
    }
};

template<typename dst_t>
struct ggml_cuda_cast_impl<dst_t, nv_bfloat16> {
    __host__ __device__ static inline dst_t cast(nv_bfloat16 x) {
        return __bfloat162float(x);
    }
};

template<>
struct ggml_cuda_cast_impl<int32_t, nv_bfloat16> {
    __host__ __device__ static inline int32_t cast(nv_bfloat16 x) {
        return int32_t(__bfloat162float(x));
    }
};

template<>
struct ggml_cuda_cast_impl<nv_bfloat16, nv_bfloat16> {
    __host__ __device__ static inline nv_bfloat16 cast(nv_bfloat16 x) {
        return x;
    }
};

template<>
struct ggml_cuda_cast_impl<float2, nv_bfloat162> {
    __host__ __device__ static inline float2 cast(nv_bfloat162 x) {
#ifdef GGML_USE_HIP
        return make_float2(__bfloat162float(__low2bfloat16(x)), __bfloat162float(__high2bfloat16(x)));
#else
#if __CUDA_ARCH__ >= 800
        return __bfloat1622float2(x);
#else
        return make_float2(__bfloat162float(x.x), __bfloat162float(x.y));
#endif // __CUDA_ARCH__ >= 800
#endif // GGML_USE_HIP
    }
};

template<>
struct ggml_cuda_cast_impl<nv_bfloat162, float2> {
    __host__ __device__ static inline nv_bfloat162 cast(float2 x) {
        // bypass compile error on cuda 12.0.1
#ifdef GGML_USE_HIP
        return __float22bfloat162_rn(x);
#else
        return {x.x, x.y};
#endif // GGML_USE_HIP
    }
};
#endif // GGML_CUDA_HAS_BF16

template<typename dst_t, typename src_t>
 __host__ __device__ inline dst_t ggml_cuda_cast(src_t x) {
    return ggml_cuda_cast_impl<dst_t, src_t>::cast(x);
}
