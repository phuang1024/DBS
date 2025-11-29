/**
 * Implementation of batched encoding.
 * Combine 8 int8 values into a single uint64.
 */

#include <torch/extension.h>
#include "eg_coding.hpp"


torch::Tensor batched_encode_tensor(torch::Tensor data) {
    BitWriter writer;

    auto accessor = data.accessor<int8_t, 1>();
    for (int i = 0; i < data.size(0); i += 8) {
        uint64_t batch_value = 0;
        for (int j = 0; j < 8; j++) {
            if (i + j >= data.size(0)) {
                break;
            }
            int8_t value = accessor[i + j];
            uint8_t pos_value = (value > 0) ? (2 * value - 1) : (-2 * value);
            batch_value |= (uint64_t)pos_value << (8 * j);
        }
        batch_value = bit_sig_perm(batch_value);
        encode_eg(batch_value, writer);
    }

    auto options = torch::TensorOptions().dtype(torch::kUInt64);
    torch::Tensor result = torch::from_blob(writer.buffer.data(), {(int64_t)writer.buffer.size()}, options).clone();
    return result;
}


torch::Tensor batched_decode_tensor(torch::Tensor data) {
    BitReader reader(data.data_ptr<uint64_t>(), data.size(0));

    std::vector<int8_t> values;
    while (true) {
        uint64_t batch_value;
        if (!decode_eg(reader, batch_value)) {
            break;
        }
        batch_value = bit_sig_perm(batch_value);

        for (int j = 0; j < 8; j++) {
            uint8_t pos_value = (batch_value >> (8 * j)) & (uint64_t)255;
            int8_t value = (pos_value & 1) ? ((pos_value + 1) / 2) : (-(pos_value / 2));
            values.push_back(value);
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kInt8);
    return torch::from_blob(values.data(), {(int64_t)values.size()}, options).clone();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_tensor", &batched_encode_tensor, "");
    m.def("decode_tensor", &batched_decode_tensor, "");
}
