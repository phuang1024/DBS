/**
 * Implementation of Exponential Golomb coding.
 *
 * Negative number support:
 * If x > 0, encode 2*x - 1
 * Else, encode -2*x
 * If decoding an odd number, return (value + 1) / 2
 * Else, return -(value / 2)
 *
 * Run length coding:
 * Only zeros are run-length encoded.
 * After every zero value, the run length of zeros is encoded using identical EG coding.
 *   Including the zero that was encoded.
 *   E.g. a run of 5 zeros is encoded as: 0, 5
 * When decoding, if a zero is decoded, the next value is decoded as the run length of zeros.
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <torch/extension.h>


/**
 * Bit writer.
 * Bit writing order is LSB first.
 */
class BitStreamWriter {
public:
    std::vector<uint32_t> buffer;
    uint32_t curr_byte;
    int bit_pos;

    BitStreamWriter():
        curr_byte(0), bit_pos(0) {}

    /**
     * Write the N least significant bits of a value to the stream.
     * The MSB of `value` is written first.
     * @param value The value to write.
     * @param bits The number N of bits to write.
     */
    void write(uint32_t value, int bits) {
        for (int i = 0; i < bits; ++i) {
            if (bit_pos == 32) {
                buffer.push_back(curr_byte);
                curr_byte = 0;
                bit_pos = 0;
            }
            curr_byte |= ((value >> (bits - i - 1)) & 1) << bit_pos;
            bit_pos++;
        }
    }

    /**
     * Write remaining bits to the stream, padding with zeros if necessary.
     */
    void finish() {
        if (bit_pos > 0) {
            buffer.push_back(curr_byte);
        }
    }
};


/**
 * Encode a uint32 value using EG, and write to bit stream.
 */
void encode_value(uint32_t value, BitStreamWriter& writer) {
    value += 1;
    int num_bits = 32 - __builtin_clz(value);
    writer.write(0, num_bits - 1);
    writer.write(value, num_bits);
}


/**
 * Bit reader.
 * Bit reading order is LSB first.
 */
class BitStreamReader {
public:
    const uint32_t* buffer;
    size_t buffer_size;
    size_t byte_pos;
    int bit_pos;

    BitStreamReader(const uint32_t* buf, size_t size):
        buffer(buf), buffer_size(size), byte_pos(0), bit_pos(0) {}

    /**
     * Read N bits from the stream into r_value.
     * The MSB of r_value is filled first.
     * @return Whether read was successful
     */
    bool read(int bits, uint32_t& r_value) {
        r_value = 0;
        for (int i = 0; i < bits; ++i) {
            if (byte_pos >= buffer_size) {
                return false;
            }
            r_value |= ((buffer[byte_pos] >> bit_pos) & 1) << (bits - i - 1);
            bit_pos++;
            if (bit_pos == 32) {
                bit_pos = 0;
                byte_pos++;
            }
        }
        return true;
    }
};


/**
 * Decode a uint32 value using EG from bit stream.
 * return: Whether decode was successful
 */
bool decode_value(BitStreamReader& reader, uint32_t& r_value) {
    // Count leading zeros
    int num_bits = 0;
    uint32_t bit;
    while (true) {
        if (!reader.read(1, bit)) {
            return false;
        }
        if (bit == 1) {
            break;
        }
        num_bits++;
    }

    // Read the value
    if (!reader.read(num_bits, r_value)) {
        return false;
    }
    // Add the leading 1 bit
    r_value |= (1 << num_bits);
    r_value -= 1;
    return true;
}


/**
 * Encode a list of values, using negative number support and run length coding.
 * @param data int32 tensor of shape (N,) containing the values to encode
 * @return uint32 tensor of shape (M,) containing the encoded bitstream
 */
torch::Tensor encode(torch::Tensor data) {
    BitStreamWriter writer;

    auto accessor = data.accessor<int32_t, 1>();
    int i = 0;
    while (i < data.size(0)) {
        int32_t value = accessor[i];
        uint32_t pos_value = (value > 0) ? (2 * value - 1) : (-2 * value);
        encode_value(pos_value, writer);

        if (value == 0) {
            // Count run length of zeros
            int run_length = 1;
            while (i + run_length < data.size(0) && accessor[i + run_length] == 0) {
                run_length++;
            }
            encode_value(run_length, writer);
            i += run_length;
        } else {
            i++;
        }
    }
    writer.finish();

    auto options = torch::TensorOptions().dtype(torch::kUInt32);
    torch::Tensor result = torch::from_blob(writer.buffer.data(), {(int64_t)writer.buffer.size()}, options).clone();
    return result;
}


/**
 * Decode a bitstream into a list of values, using negative number support and run length coding.
 * @param data uint32 tensor of shape (N,) containing the bitstream
 * @return uint32 tensor of shape (M,) containing the decoded values
 */
torch::Tensor decode(torch::Tensor data) {
    BitStreamReader reader(data.data_ptr<uint32_t>(), data.numel());

    std::vector<uint32_t> values;
    while (true) {
        uint32_t pos_value;
        if (!decode_value(reader, pos_value)) {
            break;
        }

        int32_t value = (pos_value & 1) ? ((pos_value + 1) / 2) : (-(pos_value / 2));
        if (value == 0) {
            // Decode run length of zeros
            uint32_t run_length;
            if (!decode_value(reader, run_length)) {
                break;
            }
            for (uint32_t i = 0; i < run_length; ++i) {
                values.push_back(0);
            }
        } else {
            values.push_back(value);
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kInt32);
    return torch::from_blob(values.data(), {(int64_t)values.size()}, options).clone();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode", &encode, "Encode a tensor of uint32 values into a bitstream with Exponential-Golomb coding.");
    m.def("decode", &decode, "");
}
