#include "exp_golomb.hpp"

#include <algorithm>
#include <climits>
#include <limits>
#include <stdexcept>
namespace eg::cpu {
namespace {
inline uint32_t countl_zero32(uint32_t x) {
#if defined(__GNUC__)
  return x ? static_cast<uint32_t>(__builtin_clz(x)) : 32u;
#else
  uint32_t n = 0;
  while ((x & (1u << 31)) == 0u && n < 32u) {
    x <<= 1;
    n++;
  }
  return n;
#endif
}
}  // namespace
struct BitWriter {
  std::vector<uint64_t> words;
  uint64_t bitpos = 0;
  void reserve_bits(uint64_t bits) {
    size_t need = (bits + 63) / 64;
    if (need > words.size())
      words.resize(need, 0);
  }
  void put_bits(uint64_t value, uint32_t nbits) {
    if (!nbits)
      return;
    reserve_bits(bitpos + nbits);
    uint64_t widx = bitpos >> 6;
    uint32_t boff = bitpos & 63;
    uint32_t first = std::min<uint32_t>(nbits, 64 - boff);
    uint64_t hi_mask = (first == 64) ? ~0ULL : ((1ULL << first) - 1);
    uint64_t hi_bits = (value >> (nbits - first)) & hi_mask;
    words[widx] |= hi_bits << (64 - boff - first);
    if (first < nbits) {
      uint32_t rem = nbits - first;
      uint64_t lo_mask = (rem == 64) ? ~0ULL : ((1ULL << rem) - 1);
      uint64_t lo_bits = value & lo_mask;
      words[widx + 1] |= lo_bits << (64 - rem);
    }
    bitpos += nbits;
  }
  eg::Bitstream finish() const {
    eg::Bitstream bs;
    bs.words = words;
    bs.bit_length = bitpos;
    return bs;
  }
};
static inline uint32_t ue_length(uint32_t v) {
  uint32_t x = v + 1u;
  if (x == 0u)
    throw std::out_of_range("Exp-Golomb UE encode requires v < UINT32_MAX");
  uint32_t lz = 31u - countl_zero32(x);
  return 2u * lz + 1u;
}
eg::Bitstream encode_ue(eg::Span<const uint32_t> codeNums) {
  uint64_t total = 0;
  for (auto v : codeNums) {
    if (v == std::numeric_limits<uint32_t>::max())
      throw std::out_of_range("Exp-Golomb UE encode requires v < UINT32_MAX");
    total += ue_length(v);
  }
  BitWriter bw;
  bw.reserve_bits(total);
  for (auto v : codeNums) {
    uint32_t x = v + 1u;
    uint32_t lz = 31u - countl_zero32(x);
    if (lz)
      bw.put_bits(0, lz);
    bw.put_bits(1, 1);
    if (lz) {
      uint32_t info = x & ((1u << lz) - 1u);
      bw.put_bits(info, lz);
    }
  }
  return bw.finish();
}
std::vector<uint32_t> decode_ue(const Bitstream& bs, size_t N) {
  std::vector<uint32_t> out;
  out.reserve(N);
  auto get_bit = [&](uint64_t i) -> uint32_t {
    uint64_t w = i >> 6;
    if (i >= bs.bit_length || w >= bs.words.size())
      throw std::out_of_range("Exp-Golomb decode read past end of stream");
    uint32_t b = i & 63;
    return (bs.words[w] >> (63 - b)) & 1u;
  };
  auto get_bits = [&](uint64_t i, uint32_t n) -> uint64_t {
    uint64_t v = 0;
    for (uint32_t k = 0; k < n; k++)
      v = (v << 1) | get_bit(i + k);
    return v;
  };
  uint64_t pos = 0;
  for (size_t i = 0; i < N; i++) {
    uint32_t lz = 0;
    while (true) {
      if (pos >= bs.bit_length)
        throw std::runtime_error("Exp-Golomb decode ran past end of stream");
      if (get_bit(pos) != 0u)
        break;
      lz++;
      pos++;
    }
    pos++;
    if (pos + lz > bs.bit_length)
      throw std::runtime_error("Exp-Golomb decode truncated in info bits");
    uint32_t info = lz ? (uint32_t)get_bits(pos, lz) : 0u;
    pos += lz;
    uint32_t x = (1u << lz) | info;
    out.push_back(x - 1u);
  }
  return out;
}
}  // namespace eg::cpu
