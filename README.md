This repository contains a referential standalone pytorch implementation of auto-regressive decoding and speculative decoding. All in [main.py](./main.py).
## Features
+ KV cache management
+ Support common sampling strategies such as greedy search, temperature sampling and top-p sampling.
+ Verified output distribution integrity(check this by setting temperature to 1e-8).
+ Verified speed up gain using various target/draft model pairs.
+ Colored visualization

## Example Usage
```bash
python main.py \
        --prompt "Below is a piece of Python code to efficienctly compute the n-th Fibonacci number using cache(a lookup table):" \
        --num_draft_tokens 4 \
        --temperature 1e-8 \
        --top_p 1.0 \
        --max_new_tokens 80
```

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
