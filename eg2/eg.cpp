/**
 * Implementation of Exp-Golomb encode and decode.
 */

#include <bit>
#include <cstdint>
#include "eg_coding.hpp"


void encode_eg(uint64_t value, BitWriter& writer) {
    value += 1;
    int num_bits = 64 - std::countl_zero(value);
    writer.write_zeros(num_bits - 1);
    writer.write(value, num_bits);
}


bool decode_eg(BitReader& reader, uint64_t& r_value) {
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


uint64_t bit_sig_perm(uint64_t value) {
    uint64_t result = 0;
    for (int bit = 0; bit < 8; bit++) {
        for (int byte = 0; byte < 8; byte++) {
            uint64_t bit_value = (value >> (byte * 8 + bit)) & 1ULL;
            result |= (bit_value << (bit * 8 + byte));
        }
    }
    return result;
}
