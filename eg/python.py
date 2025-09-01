import torch


class ExpGolombCode:
    def __init__(self, k=0):
        self.k = k

    def encode(self, nums):
        codes = [None] * len(nums)
        for i, num in enumerate(nums):
            code = num + (1 << self.k)
            codes[i] = "0" * (int(code).bit_length() - self.k - 1) + bin(code)[2:]
        return codes

    def decode(self, codes):
        nums = torch.zeros(len(codes), dtype=torch.long)
        for i, code in enumerate(codes):
            num = int("0b" + code, base=2)
            nums[i] = num - (1 << self.k)
        return nums

    def streamEncode(self, nums):
        codes = self.encode(nums)
        return "".join(codes)

    def streamDecode(self, streamStr):
        codes = []
        start = 0
        while start < len(streamStr):
            cnt = 0
            while streamStr[start + cnt] == "0":
                cnt += 1
            end = start + 2 * cnt + self.k + 1
            codes.append(streamStr[start:end])
            start = end
        nums = self.decode(codes)
        return nums

EGCode = ExpGolombCode(k=0)


def encode(nums):
    return EGCode.streamEncode(nums.tolist())


def decode(data):
    return EGCode.streamDecode(data).to(torch.uint32)
