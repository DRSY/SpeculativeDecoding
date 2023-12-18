This repository contains a referential standalone pytorch implementation of auto-regressive decoding and speculative decoding. All in [main.py](./main.py).
## Features
+ Support common sampling strategies such as greedy search, temperature sampling and top-p sampling.
+ Verified output distribution integrity(check this by setting temperature to 1e-8).
+ Verified speed up gain using various target/draft model pairs.

## Acknowledgement
Our implementation is based on the version of the following paper from DeepMind:
```latex
@article{chen2023accelerating,
  title={Accelerating large language model decoding with speculative sampling},
  author={Chen, Charlie and Borgeaud, Sebastian and Irving, Geoffrey and Lespiau, Jean-Baptiste and Sifre, Laurent and Jumper, John},
  journal={arXiv preprint arXiv:2302.01318},
  year={2023}
}
```
