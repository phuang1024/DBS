/**
 * Exponential golomb coding implementation.
 *
 * Code words are uint64_t.
 * The MSB of values to encode are written first (a requirement of EG coding).
 * The MSB of code words are used first, to allow for efficiency.
 */

#include <bit>
#include <cstdint>
#include <vector>
#include <torch/extension.h>


class BitWriter {
public:
    // Current byte is stored in the last element of the buffer.
    std::vector<uint64_t> buffer;
    // Position of next bit to write. From 63 (MSB) to 0 (LSB).
    int bit_pos;

    BitWriter() : bit_pos(63) {
        buffer.push_back(0);
    }

    void write(uint64_t value, int bits) {
        // Index of bit in value to write next.
        int i = bits - 1;

        // Batch write bits.
        while (i >= 0) {
            if (bit_pos < 0) {
                buffer.push_back(0);
                bit_pos = 63;
            }

            int bits_to_write = std::min(i + 1, bit_pos + 1);
            uint64_t chunk = (value >> (i - bits_to_write + 1)) & ((1ULL << bits_to_write) - 1);
            buffer.back() |= (chunk << (bit_pos - bits_to_write + 1));
            bit_pos -= bits_to_write;
            i -= bits_to_write;
        }
    }

    /**
     * Write N zero bits.
     * Only advances the bit counter.
     * Requires 0 <= bits < 64.
     */
    void write_zeros(int bits) {
        bit_pos -= bits;
        if (bit_pos < 0) {
            buffer.push_back(0);
            bit_pos = 64 + bit_pos;
        }
    }
};


/**
 * Bit reader.
 * Bit reading order is LSB first.
 */
class BitReader {
public:
    const uint64_t* buffer;
    size_t buffer_size;

    size_t byte_pos;
    int bit_pos;

    BitReader(const uint64_t* buf, size_t buf_size):
            buffer(buf), byte_pos(0), bit_pos(63), buffer_size(buf_size) {
    }

    /**
     * Read N bits from the stream into r_value.
     * The MSB of r_value is filled first.
     * @return Whether read was successful
     */
    bool read(int bits, uint64_t& r_value) {
        r_value = 0;

        // Index of bit in r_value to write next.
        int i = bits - 1;

        // Batch read bits.
        while (i >= 0) {
            if (bit_pos < 0) {
                bit_pos = 63;
                byte_pos++;
            }
            if (byte_pos >= buffer_size) {
                return false;
            }

            int bits_to_read = std::min(i + 1, bit_pos + 1);
            uint64_t chunk = (buffer[byte_pos] >> (bit_pos - bits_to_read + 1)) & ((1ULL << bits_to_read) - 1);
            r_value |= (chunk << (i - bits_to_read + 1));
            bit_pos -= bits_to_read;
            i -= bits_to_read;
        }

        return true;
    }

    /**
     * Count the number of zeros until the next one bit.
     * Advances pointer to the next one bit; i.e. the next bit read will be the one.
     */
    bool read_zeros(int& r_count) {
        r_count = 0;

        while (true) {
            if (bit_pos < 0) {
                bit_pos = 63;
                byte_pos++;
            }
            if (byte_pos >= buffer_size) {
                return false;
            }

            uint64_t byte = (bit_pos == 63) ? buffer[byte_pos] : buffer[byte_pos] & ((1ULL << (bit_pos + 1)) - 1);
            int count = std::countl_zero(byte) - 63 + bit_pos;
            r_count += count;
            bit_pos -= count;
            if (bit_pos >= 0) {
                // Found a one bit
                return true;
            }
        }
    }
};


/**
 * Encode a uint64 value using EG to bit stream.
 */
void encode_value(uint64_t value, BitWriter& writer) {
    value += 1;
    int num_bits = 64 - std::countl_zero(value);
    writer.write_zeros(num_bits - 1);
    writer.write(value, num_bits);
}


/**
 * Encode a list of values, using negative number support and run length coding.
 * @param data int64 tensor of shape (N,) containing the values to encode
 * @return uint64 tensor of shape (M,) containing the encoded bitstream
 */
torch::Tensor encode_tensor(torch::Tensor data) {
    BitWriter writer;

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

    auto options = torch::TensorOptions().dtype(torch::kUInt64);
    torch::Tensor result = torch::from_blob(writer.buffer.data(), {(int64_t)writer.buffer.size()}, options).clone();
    return result;
}


/**
 * Decode a uint64 value using EG from bit stream.
 * return: Whether decode was successful
 */
bool decode_value(BitReader& reader, uint64_t& r_value) {
    // Count leading zeros
    int num_bits = 0;
    reader.read_zeros(num_bits);

    // Read the value
    if (!reader.read(num_bits + 1, r_value)) {
        return false;
    }
    r_value -= 1;

    return true;
}


/**
 * Decode a bitstream into a list of values, using negative number support and run length coding.
 * @param data uint64 tensor of shape (N,) containing the bitstream
 * @return int64 tensor of shape (M,) containing the decoded values
 */
torch::Tensor decode_tensor(torch::Tensor data) {
    BitReader reader(data.data_ptr<uint64_t>(), data.size(0));

    std::vector<uint64_t> values;
    while (true) {
        uint64_t pos_value;
        if (!decode_value(reader, pos_value)) {
            break;
        }

        int64_t value = (pos_value & 1) ? ((pos_value + 1) / 2) : (-(pos_value / 2));
        if (value == 0) {
            // Decode run length of zeros
            uint64_t run_length;
            if (!decode_value(reader, run_length)) {
                break;
            }
            for (uint64_t i = 0; i < run_length; ++i) {
                values.push_back(0);
            }
        } else {
            values.push_back(value);
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kInt64);
    return torch::from_blob(values.data(), {(int64_t)values.size()}, options).clone();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("encode_tensor", &encode_tensor, "");
    m.def("decode_tensor", &decode_tensor, "");
}
