# microGNN: Graph-to-MLP Knowledge Distillation

A minimal, educational implementation of **knowledge distillation** from a Graph Convolutional Network (GCN) teacher to an MLP student for node classification. No `torch_geometric`—only PyTorch.

## Motivation

GNNs are memory- and compute-intensive in production: they require the full graph (adjacency matrix) at inference and repeated message-passing. Distilling a trained GCN into a small MLP that uses only **node features** (no graph structure) yields a model that is cheaper to deploy and scales to high-throughput serving while retaining much of the teacher’s accuracy.

## Math

- **Normalized adjacency:**  
  Â = D^{-1/2}(A + I)D^{-1/2}  
  where D is the degree matrix of (A + I).

- **GCN layer:**  
  H^{(l+1)} = σ(Â H^{(l)} W^{(l)})  
  with σ = ReLU (hidden) or identity (output). Implemented with `torch.matmul`.

- **Distillation:** The teacher produces soft logits; the student is trained to match them with KL divergence (with optional temperature T).

## How to run

```bash
pip install -r requirements.txt
python micro_gnn.py
```

You should see Teacher GCN accuracy and Student MLP accuracy (after distillation) printed.

## Project layout

- `micro_gnn.py` — data generation, GCN math, Teacher GCN, Student MLP, training and distillation loop.
