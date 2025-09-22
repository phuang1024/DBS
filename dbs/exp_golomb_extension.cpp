#include <torch/extension.h>

#include "exp_golomb.hpp"

#include <cstring>
#include <vector>

namespace {

std::vector<uint32_t> tensor_to_u32_vec(const torch::Tensor& tensor) {
  auto t = tensor.to(torch::kInt).cpu().contiguous();
  std::vector<uint32_t> data(t.numel());
  auto* src = t.data_ptr<int32_t>();
  for (int64_t i = 0; i < t.numel(); ++i) {
    if (src[i] < 0)
      throw std::out_of_range("Exp-Golomb UE encode requires non-negative values");
    data[static_cast<size_t>(i)] = static_cast<uint32_t>(src[i]);
  }
  return data;
}

torch::Tensor u64_vec_to_tensor(const std::vector<uint64_t>& data) {
  torch::Tensor out = torch::empty({static_cast<long>(data.size())}, torch::dtype(torch::kUInt64));
  std::memcpy(out.data_ptr<uint64_t>(), data.data(), data.size() * sizeof(uint64_t));
  return out;
}

std::tuple<torch::Tensor, int64_t> encode_ue_cpu(torch::Tensor values) {
  auto host = tensor_to_u32_vec(values);
  eg::Bitstream bs = eg::cpu::encode_ue({host.data(), host.size()});
  return std::make_tuple(u64_vec_to_tensor(bs.words), static_cast<int64_t>(bs.bit_length));
}

torch::Tensor decode_ue_cpu(torch::Tensor words_tensor, int64_t bit_length, int64_t num_symbols) {
  torch::Tensor words_cpu = words_tensor.to(torch::kCPU).contiguous();
  std::vector<uint64_t> words(words_cpu.numel());
  std::memcpy(words.data(), words_cpu.data_ptr<uint64_t>(), words.size() * sizeof(uint64_t));
  eg::Bitstream bs;
  bs.words = std::move(words);
  bs.bit_length = static_cast<uint64_t>(bit_length);
  auto decoded = eg::cpu::decode_ue(bs, static_cast<size_t>(num_symbols));
  auto out = torch::empty({static_cast<long>(decoded.size())}, torch::dtype(torch::kInt));
  std::memcpy(out.data_ptr<int32_t>(), decoded.data(), decoded.size() * sizeof(int32_t));
  return out;
}

std::tuple<torch::Tensor, int64_t> encode_ue_gpu(torch::Tensor values) {
  auto host = tensor_to_u32_vec(values);
  eg::Bitstream bs = eg::gpu::encode_ue({host.data(), host.size()});
  return std::make_tuple(u64_vec_to_tensor(bs.words), static_cast<int64_t>(bs.bit_length));
}

torch::Tensor decode_ue_many_gpu(torch::Tensor words_tensor, torch::Tensor offsets_tensor, int64_t symbols_per_stream) {
  auto words_cpu = words_tensor.to(torch::kCPU).contiguous();
  auto offsets_cpu = offsets_tensor.to(torch::kCPU).contiguous();

  std::vector<uint64_t> words(words_cpu.numel());
  std::memcpy(words.data(), words_cpu.data_ptr<uint64_t>(), words.size() * sizeof(uint64_t));

  std::vector<uint64_t> offsets(offsets_cpu.numel());
  std::memcpy(offsets.data(), offsets_cpu.data_ptr<uint64_t>(), offsets.size() * sizeof(uint64_t));

  auto decoded = eg::gpu::decode_ue_many({words.data(), words.size()}, {offsets.data(), offsets.size()}, static_cast<size_t>(symbols_per_stream));
  auto out = torch::empty({static_cast<long>(decoded.size())}, torch::dtype(torch::kInt));
  std::memcpy(out.data_ptr<int32_t>(), decoded.data(), decoded.size() * sizeof(int32_t));
  return out.view({static_cast<long>(offsets.size()), static_cast<long>(symbols_per_stream)});
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("encode_ue_cpu", &encode_ue_cpu, "Encode unsigned Exp-Golomb (CPU)");
  m.def("decode_ue_cpu", &decode_ue_cpu, "Decode unsigned Exp-Golomb (CPU)",
        py::arg("words"), py::arg("bit_length"), py::arg("num_symbols"));
  m.def("encode_ue_gpu", &encode_ue_gpu, "Encode unsigned Exp-Golomb (GPU wrapper)");
  m.def("decode_ue_many_gpu", &decode_ue_many_gpu, "Decode many streams (GPU)",
        py::arg("words"), py::arg("offset_bits"), py::arg("symbols_per_stream"));
}
