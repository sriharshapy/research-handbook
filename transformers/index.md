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

$$
\frac{a}{b} = \frac{c}{d}
$$
