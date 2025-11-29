/**
 * Implementation of standard sequential EG coding.
 * Encode each int8 value as is.
 */

#include <torch/extension.h>
#include "eg_coding.hpp"


torch::Tensor std_encode_tensor(torch::Tensor data) {
    BitWriter writer;

    auto accessor = data.accessor<int8_t, 1>();
    int i = 0;
    while (i < data.size(0)) {
        int64_t value = (int64_t)accessor[i];
        uint64_t pos_value = (value > 0) ? (2 * value - 1) : (-2 * value);
        encode_eg(pos_value, writer);

        if (value == 0) {
            // Count run length of zeros
            int run_length = 1;
            while (i + run_length < data.size(0) && accessor[i + run_length] == value) {
                run_length++;
            }
            encode_eg(run_length, writer);
            i += run_length;
        } else {
            i++;
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kUInt64);
    torch::Tensor result = torch::from_blob(writer.buffer.data(), {(int64_t)writer.buffer.size()}, options).clone();
    return result;
}


torch::Tensor std_decode_tensor(torch::Tensor data) {
    BitReader reader(data.data_ptr<uint64_t>(), data.size(0));

    std::vector<int8_t> values;
    while (true) {
        uint64_t pos_value;
        if (!decode_eg(reader, pos_value)) {
            break;
        }

        int64_t value = (pos_value & 1) ? ((pos_value + 1) / 2) : (-(pos_value / 2));
        if (value == 0) {
            // Decode run length of zeros
            uint64_t run_length;
            if (!decode_eg(reader, run_length)) {
                break;
            }
            for (uint64_t i = 0; i < run_length; ++i) {
                values.push_back(value);
            }
        } else {
            values.push_back((int8_t)value);
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kInt8);
    return torch::from_blob(values.data(), {(int64_t)values.size()}, options).clone();
}


/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_tensor", &std_encode_tensor, "");
    m.def("decode_tensor", &std_decode_tensor, "");
}
*/
