#pragma once
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <vector>
namespace eg {
template <typename T>
class Span {
 public:
  using element_type = T;
  using value_type = typename std::remove_cv<T>::type;
  using size_type = std::size_t;
  using pointer = T*;
  using iterator = T*;

  Span() : ptr_(nullptr), size_(0) {}
  Span(pointer ptr, size_type count) : ptr_(ptr), size_(count) {}
  template <typename Alloc>
  Span(std::vector<value_type, Alloc>& vec) : ptr_(vec.data()), size_(vec.size()) {}
  template <typename Alloc>
  Span(const std::vector<value_type, Alloc>& vec) : ptr_(vec.data()), size_(vec.size()) {}
  template <typename U = T, typename = typename std::enable_if<std::is_const<U>::value>::type>
  Span(std::initializer_list<value_type> init)
      : ptr_(init.begin()), size_(init.size()) {}

  pointer data() const { return ptr_; }
  size_type size() const { return size_; }
  bool empty() const { return size_ == 0; }
  iterator begin() const { return ptr_; }
  iterator end() const { return ptr_ + size_; }
  T& operator[](size_type idx) const { return ptr_[idx]; }

 private:
  pointer ptr_;
  size_type size_;
};
struct Bitstream {
  std::vector<uint64_t> words;
  uint64_t bit_length = 0;
};
namespace cpu {
// Exp-Golomb UE encoding expects every value to be strictly less than UINT32_MAX.
// Callers must also ensure that every decode has enough remaining bits; the decoder
// will throw std::runtime_error if a stream terminates early.
Bitstream encode_ue(Span<const uint32_t> codeNums);
std::vector<uint32_t> decode_ue(const Bitstream& bs, size_t N);
inline uint32_t map_se_to_ue(int32_t v) {
  if (v == std::numeric_limits<int32_t>::min())
    throw std::out_of_range("Exp-Golomb SE mapping requires v > INT32_MIN");
  return (v <= 0) ? (uint32_t)(-2LL * (int64_t)v) : (uint32_t)(2LL * (int64_t)v - 1);
}
inline int32_t map_ue_to_se(uint32_t u) {
  long long k = (u + 1) / 2;
  return (u % 2 == 0) ? (int32_t)(-k) : (int32_t)(k);
}
}  // namespace cpu
namespace gpu {
Bitstream encode_ue(Span<const uint32_t> codeNums);
std::vector<uint32_t> decode_ue_many(Span<const uint64_t> bitstreams_words,
                                     Span<const uint64_t> stream_offsets_bits, size_t Ns);
}  // namespace gpu
}  // namespace eg
