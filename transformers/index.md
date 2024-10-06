---
layout: default
title: "Transformers"
permalink: /transformers/
nav_order: 4
---

# Tranformers
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}


## Glossary

The following are some of the important concepts of transformers. They also
have simple examples to understand them in detail.

### Multi-Head Attention

Instead of using a single set of query (Q), key (K), and value (V) matrices,
multi-head attention splits these into multiple sets. Each set, or "head,"
processes the input independently, allowing the model to focus on different
parts of the sequence. Each head performs its own attention calculation,
capturing various details such as syntax, semantics, or long-term dependencies.
The outputs from all heads are concatenated and linearly transformed to produce
the final output.

```LaTeX
MultiHead(Q,K,V)=Concat(head-1,…,head-h)⋅W_o
```

#### Scaled dot product attention

![Alt text](assets/images/transformers/Scaled_Dot_Product.png)

```python
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)
```



## Sample code

```python
import torch
import torch.nn.functional as F
from nnsight import NNsight
import nnsight

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    attention_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)
```

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # Linear projections
        print("Forward pass Q shape :",Q.shape)
        Q = self.W_q(Q)
        print("Forward pass Q shape after W_q:",Q.shape)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k)
        print("Forward pass Q shape after view:",Q.shape)
        Q = Q.transpose(1, 2)
        print("Forward pass Q shape after transpose:",Q.shape)
        print()

        print("Forward pass K shape :",K.shape)
        K = self.W_k(K)
        print("Forward pass K shape after W_k:",K.shape)
        K = K.view(batch_size, -1, self.num_heads, self.d_k)
        print("Forward pass K shape after view:",K.shape)
        K = K.transpose(1, 2)
        print("Forward pass K shape after transpose:",K.shape)
        print()

        print("Forward pass V shape :",V.shape)
        V = self.W_v(V)
        print("Forward pass V shape after W_v:",V.shape)
        V = V.view(batch_size, -1, self.num_heads, self.d_k)
        print("Forward pass V shape after view:",V.shape)
        V = V.transpose(1, 2)
        print("Forward pass V shape after transpose:",V.shape)
        print()

        # Apply attention on all the projected vectors in batch
        scores = scaled_dot_product_attention(Q, K, V)
        print("Forward pass scores shape :",scores.shape)
        print()

        # Concatenate and apply final linear layer
        scores = scores.transpose(1, 2)
        print("Forward pass scores shape after transpose:",scores.shape)
        scores = scores.contiguous()
        print("Forward pass scores shape after contiguous:",scores.shape)
        concat_scores = scores.view(batch_size, -1, self.num_heads * self.d_k)
        print("Forward pass concat_scores shape after view:",concat_scores.shape)
        print()

        concat_scores  = self.W_o(concat_scores)
        print("Forward pass concat_scores shape after W_o:",concat_scores.shape)

        return concat_scores
```

```python
# Example usage:
d_model = 512
num_heads = 8

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

# Dummy input (batch_size=64; sequence_length=10; embedding_dim=512)
Q = K = V = torch.rand(64, 10, d_model)

output = mha(Q, K, V)
```

```
Forward pass Q shape : torch.Size([64, 10, 512])
Forward pass Q shape after W_q: torch.Size([64, 10, 512])
Forward pass Q shape after view: torch.Size([64, 10, 8, 64])
Forward pass Q shape after transpose: torch.Size([64, 8, 10, 64])

Forward pass K shape : torch.Size([64, 10, 512])
Forward pass K shape after W_k: torch.Size([64, 10, 512])
Forward pass K shape after view: torch.Size([64, 10, 8, 64])
Forward pass K shape after transpose: torch.Size([64, 8, 10, 64])

Forward pass V shape : torch.Size([64, 10, 512])
Forward pass V shape after W_v: torch.Size([64, 10, 512])
Forward pass V shape after view: torch.Size([64, 10, 8, 64])
Forward pass V shape after transpose: torch.Size([64, 8, 10, 64])

Forward pass scores shape : torch.Size([64, 8, 10, 64])

Forward pass scores shape after transpose: torch.Size([64, 10, 8, 64])
Forward pass scores shape after contiguous: torch.Size([64, 10, 8, 64])
Forward pass concat_scores shape after view: torch.Size([64, 10, 512])

Forward pass concat_scores shape after W_o: torch.Size([64, 10, 512])
```

```python
m = NNsight(mha)

with m.trace(Q,K,V) as tracer:
  OUT = []
  for x in [m.W_q, m.W_k, m.W_v, m.W_o]:
    input_dim = x.input.shape.save()
    output_dim = x.output.shape.save()
    OUT.append((x,input_dim,output_dim))


for x in OUT:
  print("layer : ",x[0])
  print("input shape :",x[1])
  print("output shape :",x[2])
  print("\n")
```

```
layer :  Linear(in_features=512, out_features=512, bias=True)
input shape : torch.Size([64, 10, 512])
output shape : torch.Size([64, 10, 512])


layer :  Linear(in_features=512, out_features=512, bias=True)
input shape : torch.Size([64, 10, 512])
output shape : torch.Size([64, 10, 512])


layer :  Linear(in_features=512, out_features=512, bias=True)
input shape : torch.Size([64, 10, 512])
output shape : torch.Size([64, 10, 512])


layer :  Linear(in_features=512, out_features=512, bias=True)
input shape : torch.Size([64, 10, 512])
output shape : torch.Size([64, 10, 512])
```
