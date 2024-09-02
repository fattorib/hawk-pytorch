# Hawk - PyTorch 

This is a PyTorch implementation of Hawk from [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427). It uses a [custom Triton kernel](https://github.com/fattorib/fast_sequential_scan) for the sequential scan and supports `torch.compile``. Because of this, it is the fastest implementation of Hawk available for GPU. 

# Install

```bash
git clone https://github.com/fattorib/hawk-pytorch
cd hawk-pytorch
pip install -e .
```

# Usage

```python
import torch 
from hawk import HawkModel, HawkConfig

config = HawkConfig(vocab_size=32000, 
                    hidden_size=512, 
                    intermediate_size=1024, 
                    recurrent_size=512, 
                    num_hidden_layers=8, 
                    num_blocks = 16,
                    post_norm=False)


model = HawkModel(config, use_cache=False)

model.to('cuda')
model = torch.compile(model) # this works!

x = torch.randint(size=(1, 2048), low=1, high=32000, device="cuda:0")
with torch.autocast(device_type = 'cuda', dtype=torch.bfloat16):
    loss = model(x, x)
loss.backward()
```

# Citations / Acknowledgemtnts

```bibtex
@misc{de2024griffinmixinggatedlinear,
      title={Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models}, 
      author={Soham De and Samuel L. Smith and Anushan Fernando and Aleksandar Botev and George Cristian-Muraru and Albert Gu and Ruba Haroun and Leonard Berrada and Yutian Chen and Srivatsan Srinivasan and Guillaume Desjardins and Arnaud Doucet and David Budden and Yee Whye Teh and Razvan Pascanu and Nando De Freitas and Caglar Gulcehre},
      year={2024},
      eprint={2402.19427},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.19427}, 
}
```

- Code in `hawk/external.py` taken from `google-deepmind/recurrentgemma`
