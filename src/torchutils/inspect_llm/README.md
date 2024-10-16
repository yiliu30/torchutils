### Llama-2-7b

```bash

output: tensor([[    1, 24948,   592,  1048,   319, 29902,   322, 23158,   297,   278,
          3030]], device='cuda:0')
Generated text: <s> Tell me about AI and ML in the context
Stage 1:
  - input: torch.Size([1, 6, 4096]), weight: torch.Size([4096, 4096]): 128
  - input: torch.Size([1, 6, 4096]), weight: torch.Size([11008, 4096]): 64
  - input: torch.Size([1, 6, 11008]), weight: torch.Size([4096, 11008]): 32
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([32000, 4096]): 1
Stage 2:
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([4096, 4096]): 128
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([11008, 4096]): 64
  - input: torch.Size([1, 1, 11008]), weight: torch.Size([4096, 11008]): 32
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([32000, 4096]): 1
Stage 3:
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([4096, 4096]): 128
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([11008, 4096]): 64
  - input: torch.Size([1, 1, 11008]), weight: torch.Size([4096, 11008]): 32
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([32000, 4096]): 1
Stage 4:
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([4096, 4096]): 128
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([11008, 4096]): 64
  - input: torch.Size([1, 1, 11008]), weight: torch.Size([4096, 11008]): 32
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([32000, 4096]): 1
Stage 5:
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([4096, 4096]): 128
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([11008, 4096]): 64
  - input: torch.Size([1, 1, 11008]), weight: torch.Size([4096, 11008]): 32
  - input: torch.Size([1, 1, 4096]), weight: torch.Size([32000, 4096]): 1
```

### Llama-2-13b

```bash
output: tensor([[    1, 24948,   592,  1048,   319, 29902,   322,   920,   366,   526,
           773]], device='cuda:0')
Generated text: <s> Tell me about AI and how you are using
Stage 1:
  - input: torch.Size([1, 6, 5120]), weight: torch.Size([5120, 5120]): 160
  - input: torch.Size([1, 6, 5120]), weight: torch.Size([13824, 5120]): 80
  - input: torch.Size([1, 6, 13824]), weight: torch.Size([5120, 13824]): 40
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([32000, 5120]): 1
Stage 2:
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([5120, 5120]): 160
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([13824, 5120]): 80
  - input: torch.Size([1, 1, 13824]), weight: torch.Size([5120, 13824]): 40
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([32000, 5120]): 1
Stage 3:
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([5120, 5120]): 160
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([13824, 5120]): 80
  - input: torch.Size([1, 1, 13824]), weight: torch.Size([5120, 13824]): 40
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([32000, 5120]): 1
Stage 4:
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([5120, 5120]): 160
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([13824, 5120]): 80
  - input: torch.Size([1, 1, 13824]), weight: torch.Size([5120, 13824]): 40
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([32000, 5120]): 1
Stage 5:
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([5120, 5120]): 160
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([13824, 5120]): 80
  - input: torch.Size([1, 1, 13824]), weight: torch.Size([5120, 13824]): 40
  - input: torch.Size([1, 1, 5120]), weight: torch.Size([32000, 5120]): 1
```