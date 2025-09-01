#include <cstdint>
#include <iostream>
#include <vector>
#include <torch/extension.h>


// Bit reading order is LSB first
class BitStreamWriter {
public:
    std::vector<uint8_t> buffer;
    uint8_t curr_byte;
    int bit_pos;

    BitStreamWriter() {
        curr_byte = 0;
        bit_pos = 0;
    }

    // Write the lowest `bits` bits of `value` to the stream
    void write(uint32_t value, int bits) {
        for (int i = 0; i < bits; ++i) {
            if (bit_pos == 8) {
                buffer.push_back(curr_byte);
                curr_byte = 0;
                bit_pos = 0;
            }
            curr_byte |= ((value >> (bits - i - 1)) & 1) << bit_pos;
            bit_pos++;
        }
    }

    void finish() {
        if (bit_pos > 0) {
            buffer.push_back(curr_byte);
        }
    }
};


/**
 * data: uint32 tensor of shape (N,) containing the values to encode
 * return: uint8 tensor of shape (M,) containing the encoded bitstream
 */
torch::Tensor encode(torch::Tensor data) {
    BitStreamWriter writer;

    auto accessor = data.accessor<uint32_t, 1>();
    for (int i = 0; i < data.size(0); ++i) {
        //std::cerr << "VALUE " << accessor[i] << "\n";
        uint32_t value = accessor[i] + 1;
        int num_bits = 32 - __builtin_clz(value);
        //std::cerr << "WRITE " << 0 << " for bits " << (num_bits - 1) << "\n";
        writer.write(0, num_bits - 1);
        //std::cerr << "WRITE " << value << " for bits " << num_bits << std::endl;
        writer.write(value, num_bits);
    }
    writer.finish();

    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    torch::Tensor result = torch::from_blob(writer.buffer.data(), {(int64_t)writer.buffer.size()}, options).clone();
    return result;
}


class BitStreamReader {
public:
    const uint8_t* buffer;
    size_t buffer_size;
    size_t byte_pos;
    int bit_pos;

    BitStreamReader(const uint8_t* buf, size_t size) : buffer(buf), buffer_size(size), byte_pos(0), bit_pos(0) {}

    // return: Whether read was successful
    bool read(int bits, uint32_t& r_value) {
        r_value = 0;
        for (int i = 0; i < bits; ++i) {
            if (byte_pos >= buffer_size) {
                return false;
            }
            r_value |= ((buffer[byte_pos] >> bit_pos) & 1) << (bits - i - 1);
            bit_pos++;
            if (bit_pos == 8) {
                bit_pos = 0;
                byte_pos++;
            }
        }
        return true;
    }

    /*
    void step_back() {
        if (bit_pos == 0) {
            if (byte_pos == 0) {
                throw std::runtime_error("Cannot step back");
            }
            byte_pos--;
            bit_pos = 7;
        } else {
            bit_pos--;
        }
    }
    */
};


/**
 * data: uint8 tensor of shape (N,) containing the bitstream
 * return: uint32 tensor of shape (M,) containing the decoded values
 */
torch::Tensor decode(torch::Tensor data) {
    BitStreamReader reader(data.data_ptr<uint8_t>(), data.numel());

    std::vector<uint32_t> values;
    while (true) {
        uint32_t value;

        // Count leading zeros
        int num_bits = 0;
        while (true) {
            if (!reader.read(1, value)) {
                goto done;
            }
            if (value == 1) {
                break;
            }
            num_bits++;
        }

        // Read the value
        if (!reader.read(num_bits, value)) {
            goto done;
        }
        // Add the leading 1 bit
        value |= (1 << num_bits);
        values.push_back(value - 1);
    }

done:
    auto options = torch::TensorOptions().dtype(torch::kUInt32);
    return torch::from_blob(values.data(), {(int64_t)values.size()}, options).clone();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode", &encode, "Encode a tensor of uint32 values into a bitstream with Exponential-Golomb coding.");
    m.def("decode", &decode, "");
}
