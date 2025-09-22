#include "exp_golomb.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/device/device_scan.cuh>
namespace eg::gpu {
static inline void ck(cudaError_t e, const char* m) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(m) + ": " + cudaGetErrorString(e));
}
__device__ __forceinline__ uint32_t ue_len_dev(uint32_t v) {
  uint32_t x = v + 1u;
  uint32_t lz = 31u - __clz(x);
  return 2u * lz + 1u;
}
__global__ void k_len(const uint32_t* in, uint32_t* lens, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    lens[i] = ue_len_dev(in[i]);
}
__device__ __forceinline__ int lower_bound_end_gt(uint64_t target, const uint64_t* offs,
                                                  const uint32_t* lens, int N) {
  int lo = 0;
  int hi = N;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    uint64_t end = offs[mid] + static_cast<uint64_t>(lens[mid]);
    if (end <= target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}
__device__ __forceinline__ int lower_bound_start_ge(uint64_t target, const uint64_t* offs,
                                                    int N) {
  int lo = 0;
  int hi = N;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (offs[mid] < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}
__global__ void k_pack_words(const uint32_t* in, const uint32_t* lens, const uint64_t* offs,
                             uint64_t total_bits, uint64_t* words, int N, int nwords) {
  int wid = blockIdx.x * blockDim.x + threadIdx.x;
  if (wid >= nwords)
    return;
  uint64_t word_start = static_cast<uint64_t>(wid) * 64ULL;
  uint64_t word_end = word_start + 64ULL;
  if (word_start >= total_bits)
    return;
  if (word_end > total_bits)
    word_end = total_bits;
  int first_symbol = lower_bound_end_gt(word_start, offs, lens, N);
  int last_symbol = lower_bound_start_ge(word_end, offs, N);
  uint64_t word = 0ULL;
  for (int idx = first_symbol; idx < last_symbol; idx++) {
    uint64_t sym_start = offs[idx];
    uint64_t sym_end = sym_start + static_cast<uint64_t>(lens[idx]);
    if (sym_end <= word_start || sym_start >= word_end)
      continue;
    uint64_t chunk_begin = sym_start < word_start ? word_start : sym_start;
    uint64_t chunk_end = sym_end < word_end ? sym_end : word_end;
    uint32_t take = static_cast<uint32_t>(chunk_end - chunk_begin);
    if (!take)
      continue;
    uint32_t skip = static_cast<uint32_t>(chunk_begin - sym_start);
    uint32_t total = lens[idx];
    uint32_t lz = (total - 1u) >> 1;
    uint32_t x = in[idx] + 1u;
    uint32_t info_mask = lz ? ((1u << lz) - 1u) : 0u;
    uint64_t code = (1ULL << lz) | static_cast<uint64_t>(x & info_mask);
    uint64_t shift = static_cast<uint64_t>(total - skip - take);
    uint64_t mask = (take == 64u) ? ~0ULL : ((1ULL << take) - 1ULL);
    uint64_t bits = (code >> shift) & mask;
    uint32_t word_offset = static_cast<uint32_t>(chunk_begin - word_start);
    uint32_t shift_into_word = 64u - (word_offset + take);
    word |= bits << shift_into_word;
  }
  words[wid] = word;
}
eg::Bitstream encode_ue(eg::Span<const uint32_t> codeNums) {
  size_t N = codeNums.size();
  if (!N)
    return {};
  for (auto v : codeNums) {
    if (v == std::numeric_limits<uint32_t>::max())
      throw std::out_of_range("Exp-Golomb UE encode requires v < UINT32_MAX");
  }
  thrust::device_vector<uint32_t> d_in(codeNums.begin(), codeNums.end()), d_len(N);
  dim3 b(256), g((unsigned)((N + b.x - 1) / b.x));
  k_len<<<g, b>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_len.data()), N);
  ck(cudaGetLastError(), "k_len");
  thrust::device_vector<uint64_t> d_off(N);
  size_t temp_bytes = 0;
  ck(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, thrust::raw_pointer_cast(d_len.data()),
                                   thrust::raw_pointer_cast(d_off.data()), N),
     "scan temp size");
  thrust::device_vector<unsigned char> d_temp(temp_bytes ? temp_bytes : 1);
  void* temp_ptr = temp_bytes ? static_cast<void*>(thrust::raw_pointer_cast(d_temp.data())) : nullptr;
  ck(cub::DeviceScan::ExclusiveSum(temp_ptr, temp_bytes, thrust::raw_pointer_cast(d_len.data()),
                                   thrust::raw_pointer_cast(d_off.data()), N),
     "exclusive scan");
  uint64_t last_off = 0;
  uint32_t last_len = 0;
  ck(cudaMemcpy(&last_off, thrust::raw_pointer_cast(d_off.data() + (N - 1)), sizeof(uint64_t),
                cudaMemcpyDeviceToHost), "copy last offset");
  ck(cudaMemcpy(&last_len, thrust::raw_pointer_cast(d_len.data() + (N - 1)), sizeof(uint32_t),
                cudaMemcpyDeviceToHost), "copy last length");
  uint64_t total_bits = last_off + static_cast<uint64_t>(last_len);
  size_t nwords = (size_t)((total_bits + 63) / 64);
  thrust::device_vector<uint64_t> d_words(nwords, 0ULL);
  dim3 bw(256), gw((unsigned)((nwords + bw.x - 1) / bw.x));
  k_pack_words<<<gw, bw>>>(thrust::raw_pointer_cast(d_in.data()),
                           thrust::raw_pointer_cast(d_len.data()),
                           thrust::raw_pointer_cast(d_off.data()), total_bits,
                           thrust::raw_pointer_cast(d_words.data()), static_cast<int>(N),
                           static_cast<int>(nwords));
  ck(cudaGetLastError(), "k_pack_words");
  ck(cudaDeviceSynchronize(), "sync");
  eg::Bitstream bs;
  bs.bit_length = total_bits;
  thrust::host_vector<uint64_t> h = d_words;
  bs.words.assign(h.begin(), h.end());
  return bs;
}
__device__ __forceinline__ uint32_t get_bit(const uint64_t* w, uint64_t bitlen, uint64_t pos) {
  if (pos >= bitlen)
    return 0u;
  uint64_t i = pos >> 6;
  uint32_t b = pos & 63;
  return (w[i] >> (63 - b)) & 1u;
}
__device__ __forceinline__ uint64_t get_bits(const uint64_t* w, uint64_t bitlen, uint64_t pos,
                                             uint32_t n) {
  uint64_t v = 0;
  for (uint32_t k = 0; k < n; k++)
    v = (v << 1) | get_bit(w, bitlen, pos + k);
  return v;
}
__global__ void k_decode_warp(const uint64_t* words, const uint64_t* stream_offs,
                              uint64_t total_bits, uint32_t* out, size_t Ns, size_t nstreams,
                              int* status) {
  uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (warp >= nstreams)
    return;
  uint32_t lane = threadIdx.x % 32;
  uint64_t bitpos = stream_offs[warp];
  if (lane == 0)
    status[warp] = 0;
  for (size_t i = 0; i < Ns; i++) {
    uint32_t lz = 0;
    int fail = 0;
    if (lane == 0) {
      while (true) {
        if (bitpos >= total_bits) {
          fail = 1;
          break;
        }
        if (get_bit(words, total_bits, bitpos) != 0u) {
          bitpos++;
          break;
        }
        lz++;
        bitpos++;
      }
    }
    lz = __shfl_sync(0xffffffff, lz, 0);
    fail = __shfl_sync(0xffffffff, fail, 0);
    if (fail) {
      if (lane == 0)
        status[warp] = 1;
      return;
    }
    uint64_t mypos = __shfl_sync(0xffffffff, bitpos, 0);
    if (lane == 0 && (mypos + lz > total_bits))
      fail = 1;
    fail = __shfl_sync(0xffffffff, fail, 0);
    if (fail) {
      if (lane == 0)
        status[warp] = 1;
      return;
    }
    uint32_t info = 0;
    if (lane == 0) {
      info = lz ? (uint32_t)get_bits(words, total_bits, mypos, lz) : 0u;
      bitpos = mypos + lz;
    }
    info = __shfl_sync(0xffffffff, info, 0);
    uint32_t x = (1u << lz) | info;
    if (lane == 0) {
      out[warp * Ns + i] = x - 1u;
    }
  }
}
std::vector<uint32_t> decode_ue_many(eg::Span<const uint64_t> words,
                                     eg::Span<const uint64_t> offs, size_t Ns) {
  size_t nstreams = offs.size();
  if (!nstreams || !Ns)
    return {};
  thrust::device_vector<uint64_t> d_words(words.begin(), words.end()),
      d_offs(offs.begin(), offs.end());
  thrust::device_vector<uint32_t> d_out(nstreams * Ns, 0u);
  thrust::device_vector<int> d_status(nstreams, 0);
  uint64_t total_bits = (uint64_t)words.size() * 64ULL;
  dim3 b(128), g((unsigned)((nstreams * 32 + b.x - 1) / b.x));
  k_decode_warp<<<g, b>>>(thrust::raw_pointer_cast(d_words.data()),
                          thrust::raw_pointer_cast(d_offs.data()), total_bits,
                          thrust::raw_pointer_cast(d_out.data()), Ns, nstreams,
                          thrust::raw_pointer_cast(d_status.data()));
  ck(cudaGetLastError(), "k_decode_warp");
  ck(cudaDeviceSynchronize(), "sync");
  thrust::host_vector<int> h_status = d_status;
  for (size_t i = 0; i < nstreams; i++) {
    if (h_status[i] != 0)
      throw std::runtime_error("Exp-Golomb decode encountered a truncated stream");
  }
  std::vector<uint32_t> out(nstreams * Ns);
  thrust::copy(d_out.begin(), d_out.end(), out.begin());
  return out;
}
}  // namespace eg::gpu
