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
 *
 * Encoding and decoding is split into a hierarchy of three functions:
 * BitStreamWriter and BitStreamReader:
 *   Write and read bits to/from a stream of uint64 values.
 * encode_value and decode_value:
 *   Encode and decode a single uint64 value using EG coding.
 * encode and decode:
 *   Encode and decode a list of int64 values, using negative number support and run length coding.
 *
 * The first uint64 in the output stream is the number of bits the encoded stream contains.
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
    std::vector<uint64_t> buffer;
    uint64_t curr_byte;
    int bit_pos;

    BitStreamWriter():
            curr_byte(0), bit_pos(0) {
        // Number of bits indicator.
        buffer.push_back(0);
    }

    /**
     * Write the N least significant bits of a value to the stream.
     * The MSB of `value` is written first.
     * @param value The value to write.
     * @param bits The number N of bits to write.
     */
    void write(uint64_t value, int bits) {
        for (int i = 0; i < bits; ++i) {
            if (bit_pos == 64) {
                buffer.push_back(curr_byte);
                curr_byte = 0;
                bit_pos = 0;
            }
            std::cerr << "write bit pos " << bit_pos << std::endl;
            curr_byte |= ((value >> (bits - i - 1)) & 1) << bit_pos;
            bit_pos++;
        }
        buffer[0] += bits;
    }

    /**
     * Write remaining bits to the stream, padding with zeros if necessary.
     */
    void finish() {
        std::cerr << "finish bit pos " << bit_pos << std::endl;
        if (bit_pos > 0) {
            buffer.push_back(curr_byte);
        }
    }
};


/**
 * Encode a uint64 value using EG, and write to bit stream.
 */
void encode_value(uint64_t value, BitStreamWriter& writer) {
    value += 1;
    int num_bits = 64 - __builtin_clz(value);
    writer.write(0, num_bits - 1);
    writer.write(value, num_bits);
}


/**
 * Bit reader.
 * Bit reading order is LSB first.
 */
class BitStreamReader {
public:
    const uint64_t* buffer;
    size_t buffer_size;
    int last_byte_bits;

    size_t byte_pos;
    int bit_pos;

    BitStreamReader(const uint64_t* buf):
            buffer(buf), byte_pos(1), bit_pos(0) {
        uint64_t num_bits = buffer[0];
        buffer_size = (num_bits + 63) / 64;
        last_byte_bits = num_bits % 64;
    }

    /**
     * Read N bits from the stream into r_value.
     * The MSB of r_value is filled first.
     * @return Whether read was successful
     */
    bool read(int bits, uint64_t& r_value) {
        r_value = 0;
        for (int i = 0; i < bits; ++i) {
            if (byte_pos - 1 >= buffer_size && bit_pos >= last_byte_bits) {
                std::cerr << "read1" << std::endl;
                return false;
            }
            r_value |= ((buffer[byte_pos] >> bit_pos) & 1) << (bits - i - 1);
            bit_pos++;
            if (bit_pos == 64) {
                bit_pos = 0;
                byte_pos++;
            }
        }
        return true;
    }
};


/**
 * Decode a uint64 value using EG from bit stream.
 * return: Whether decode was successful
 */
bool decode_value(BitStreamReader& reader, uint64_t& r_value) {
    // Count leading zeros
    int num_bits = 0;
    uint64_t bit;
    while (true) {
        if (!reader.read(1, bit)) {
            std::cerr << "decvalue1" << std::endl;
            return false;
        }
        if (bit == 1) {
            break;
        }
        num_bits++;
    }

    // Read the value
    if (!reader.read(num_bits, r_value)) {
        std::cerr << "decvalue2" << std::endl;
        return false;
    }
    // Add the leading 1 bit
    r_value |= (1 << num_bits);
    r_value -= 1;
    return true;
}


/**
 * Encode a list of values, using negative number support and run length coding.
 * @param data int64 tensor of shape (N,) containing the values to encode
 * @return uint64 tensor of shape (M,) containing the encoded bitstream
 */
torch::Tensor encode(torch::Tensor data) {
    BitStreamWriter writer;

    auto accessor = data.accessor<int64_t, 1>();
    int i = 0;
    while (i < data.size(0)) {
        int64_t value = accessor[i];
        uint64_t pos_value = (value > 0) ? (2 * value - 1) : (-2 * value);
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

    auto options = torch::TensorOptions().dtype(torch::kUInt64);
    torch::Tensor result = torch::from_blob(writer.buffer.data(), {(int64_t)writer.buffer.size()}, options).clone();
    return result;
}


/**
 * Decode a bitstream into a list of values, using negative number support and run length coding.
 * @param data uint64 tensor of shape (N,) containing the bitstream
 * @return int64 tensor of shape (M,) containing the decoded values
 */
torch::Tensor decode(torch::Tensor data) {
    BitStreamReader reader(data.data_ptr<uint64_t>());

    std::vector<uint64_t> values;
    while (true) {
        uint64_t pos_value;
        if (!decode_value(reader, pos_value)) {
            std::cerr << "dec1" << std::endl;
            break;
        }

        int64_t value = (pos_value & 1) ? ((pos_value + 1) / 2) : (-(pos_value / 2));
        if (value == 0) {
            // Decode run length of zeros
            uint64_t run_length;
            if (!decode_value(reader, run_length)) {
                std::cerr << "dec2" << std::endl;
                break;
            }
            for (uint64_t i = 0; i < run_length; ++i) {
                values.push_back(0);
            }
        } else {
            values.push_back(value);
        }
        std::cerr << "finished read " << value << std::endl;
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64);
    return torch::from_blob(values.data(), {(int64_t)values.size()}, options).clone();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode", &encode, "");
    m.def("decode", &decode, "");
}
