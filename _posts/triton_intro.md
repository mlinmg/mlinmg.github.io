---
layout: post
title: "Getting Started with Triton Kernels for GPU Acceleration"
date: 2025-01-15 10:30:00 +0100
categories: [ml, gpu, optimization]
---

Triton is a language for writing custom GPU kernels without directly touching CUDA. It's perfect for optimizing attention operations or other custom ops in PyTorch.

## Why Triton?

- Python syntax, not CUDA C++
- Automatic memory optimization
- Compatible with any GPU (NVIDIA, AMD, Intel)

## Hello World Kernel

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
              BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)
```

## How to Use It

```python
import torch

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# Test
x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')
z = add(x, y)
```

## Next Steps

- Implement custom attention mechanisms
- Optimize batch operations for inference
- Benchmark against native CUDA kernels

Triton really is the sweet spot between productivity and performance.
