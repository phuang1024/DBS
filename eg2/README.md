This directory implements EG coding for compression
in the all_reduce operation of PyTorch distributed.

See https://docs.google.com/document/d/1ZgLJoC_sL7M9nvwSkRmESeN065Z6dCVlwwUM1_BLn88/edit?tab=t.0#heading=h.qlypzpz99akk

This is different from previous implementations (`eg`, `dbs`) because
this modifies the all_reduce operation and compresses and decompresses with each transfer.
