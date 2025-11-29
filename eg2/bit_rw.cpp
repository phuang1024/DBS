/**
 * Implementation of bit reader and writer.
 */

#include <bit>
#include <cstdint>
#include "eg_coding.hpp"


BitWriter::BitWriter() {
    bit_pos = 63;
    buffer.push_back(0);
}

void BitWriter::write(uint64_t value, int bits) {
    // Index of bit in value to write next.
    int i = bits - 1;

    // Batch write bits.
    while (i >= 0) {
        if (bit_pos < 0) {
            buffer.push_back(0);
            bit_pos = 63;
        }

        int bits_to_write = std::min(i + 1, bit_pos + 1);

        uint64_t chunk = (value >> (i - bits_to_write + 1));
        // Only need to mask if not writing full 64 bits.
        if (bits_to_write < 64) {
            chunk &= ((1ULL << bits_to_write) - 1);
        }

        buffer.back() |= (chunk << (bit_pos - bits_to_write + 1));

        bit_pos -= bits_to_write;
        i -= bits_to_write;
    }
}

void BitWriter::write_zeros(int bits) {
    bit_pos -= bits;
    while (bit_pos < 0) {
        buffer.push_back(0);
        bit_pos += 64;
    }
}


BitReader::BitReader(const uint64_t* buf, size_t buf_size):
        buffer(buf), byte_pos(0), bit_pos(63), buffer_size(buf_size) {
}

bool BitReader::read(int bits, uint64_t& r_value) {
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

        uint64_t chunk = (buffer[byte_pos] >> (bit_pos - bits_to_read + 1));
        if (bits_to_read < 64) {
            chunk &= ((1ULL << bits_to_read) - 1);
        }

        r_value |= (chunk << (i - bits_to_read + 1));

        bit_pos -= bits_to_read;
        i -= bits_to_read;
    }

    return true;
}

bool BitReader::read_zeros(int& r_count) {
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
