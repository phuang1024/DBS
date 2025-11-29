/**
 * Exponential Golomb coding implementation for torch tensors.
 *
 * This module provides:
 * - Bit stream r/w classes.
 * - Exp-Golomb encode and decode functions.
 * - Various methods to encode tensors.
 *
 * Coding turns int8 values into uint64 words.
 * int8 are turned into uint8 via standard EG mapping.
 *
 * The MSB of values to encode are written first (a requirement of EG coding).
 * The MSB of code words are used first, to allow for efficiency.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <torch/extension.h>


class BitWriter {
public:
    // Current byte is stored in the last element of the buffer.
    std::vector<uint64_t> buffer;
    // Position of next bit to write. From 63 (MSB) to 0 (LSB).
    int bit_pos;

    BitWriter();

    void write(uint64_t value, int bits);

    /**
     * Write N zero bits.
     * Only advances the bit counter.
     * Requires 0 <= bits < 64.
     */
    void write_zeros(int bits);
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

    BitReader(const uint64_t* buf, size_t buf_size);

    /**
     * Read N bits from the stream into r_value.
     * The MSB of r_value is filled first.
     * @return Whether read was successful
     */
    bool read(int bits, uint64_t& r_value);

    /**
     * Count the number of zeros until the next one bit.
     * Advances pointer to the next one bit; i.e. the next bit read will be the one.
     */
    bool read_zeros(int& r_count);
};


/**
 * Encode a uint64 value using EG to bit stream.
 */
void encode_eg(uint64_t value, BitWriter& writer);

/**
 * Decode a uint64 value using EG from bit stream.
 * return: Whether decode was successful
 */
bool decode_eg(BitReader& reader, uint64_t& r_value);

/**
 * Bit permutation for optimal compression:
 * The highest 8 bits of the output
 *   is each of the highest 8 bits of each 8-bit value.
 * Then the next 8 bits of the output
 *   are each of the next 8 bits of each input value.
 *
 * This operation is it's own inverse.
 */
uint64_t bit_sig_perm(uint64_t value);


/**
 * Standard encoding method.
 * Encodes parameters one by one.
 * Uses run length coding for zeros.
 */
torch::Tensor std_encode_tensor(torch::Tensor data);

torch::Tensor std_decode_tensor(torch::Tensor data);


/**
 * Batched encoding method.
 * Combine every 8 int8 values into a single uint64 value to encode.
 */
torch::Tensor batched_encode_tensor(torch::Tensor data);

/**
 * Batched decoding method.
 * The return tensor length will be a multiple of 8,
 *   which is the original tensor zero padded.
 */
torch::Tensor batched_decode_tensor(torch::Tensor data);


/**
 * Sign encoding method.
 * Each value is encoded as two bits:
 * - 00: zero.
 * - 01: positive.
 * - 10: negative.
 * Therefore, 32 values are packed into a single uint64.
 */
torch::Tensor sign_encode_tensor(torch::Tensor data);

/**
 * The return tensor length will be a multiple of 32,
 *   which is the original tensor zero padded.
 */
torch::Tensor sign_decode_tensor(torch::Tensor data);
