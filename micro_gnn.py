"""
microGNN: Graph-to-MLP Knowledge Distillation
Minimal GCN teacher + MLP student, trained with KD for node classification.
Dependencies: torch only. No torch_geometric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. Data Generation: synthetic financial transaction network (fraud vs normal)
# -----------------------------------------------------------------------------
def generate_toy_graph(n_nodes=80, n_features=8, n_fraud=20, edge_prob_same=0.15, edge_prob_diff=0.04, seed=42):
    """Returns (node_features, adj_matrix_dense, labels). Labels: 0=normal, 1=fraud."""
    torch.manual_seed(seed)
    # Node features: random; labels: first n_fraud are fraud, rest normal
    labels = torch.zeros(n_nodes, dtype=torch.long)
    labels[:n_fraud] = 1
    X = torch.randn(n_nodes, n_features) * 0.5
    # Adjacency: homophily — more edges among same-label nodes
    A = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            p = edge_prob_same if labels[i] == labels[j] else edge_prob_diff
            if torch.rand(1).item() < p:
                A[i, j] = A[j, i] = 1.0
    # Add self-loops for GCN normalization
    A = A + torch.eye(n_nodes)
    return X, A, labels


# -----------------------------------------------------------------------------
# 2. GCN Math: H^{(l+1)} = σ(Â H^{(l)} W^{(l)}), Â = D^{-1/2}(A+I)D^{-1/2}
# -----------------------------------------------------------------------------
def normalize_adj(A):
    """A has self-loops. Return Â = D^{-1/2} A D^{-1/2}."""
    D = A.sum(dim=1)
    D_inv_sqrt = torch.pow(D + 1e-8, -0.5)
    return D_inv_sqrt.view(-1, 1) * A * D_inv_sqrt.view(1, -1)


def gcn_layer(H, A_norm, W):
    """Single GCN layer: H_new = A_norm @ H @ W (no activation here)."""
    return torch.matmul(torch.matmul(A_norm, H), W)


# -----------------------------------------------------------------------------
# 3. Teacher: 2-layer GCN (uses graph structure)
# -----------------------------------------------------------------------------
class TeacherGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(in_dim, hidden_dim) * 0.1)
        self.W2 = nn.Parameter(torch.randn(hidden_dim, out_dim) * 0.1)

    def forward(self, x, A_norm):
        H1 = F.relu(gcn_layer(x, A_norm, self.W1))
        H2 = gcn_layer(H1, A_norm, self.W2)
        return H2  # logits


# -----------------------------------------------------------------------------
# 4. Student: 2-layer MLP (no adjacency — deployment-friendly)
# -----------------------------------------------------------------------------
class StudentMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits


# -----------------------------------------------------------------------------
# 5. Training: Phase 1 = Teacher CE loss; Phase 2 = Distill with KL
# -----------------------------------------------------------------------------
def train_teacher(model, X, A_norm, labels, epochs=150, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X, A_norm)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
    return model


def train_student_distill(student, teacher, X, A_norm, epochs=150, lr=0.01, T=3.0):
    """Distill teacher logits into student using KL divergence (soft targets)."""
    teacher.eval()
    with torch.no_grad():
        soft_targets = F.log_softmax(teacher(X, A_norm) / T, dim=1)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        log_student = F.log_softmax(student(X) / T, dim=1)
        loss = F.kl_div(log_student, torch.exp(soft_targets), reduction="batchmean") * (T * T)
        loss.backward()
        opt.step()
    return student


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


# -----------------------------------------------------------------------------
# 6. Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    X, A, labels = generate_toy_graph()
    A_norm = normalize_adj(A)
    n_classes = 2
    hidden = 16

    teacher = TeacherGCN(X.shape[1], hidden, n_classes)
    train_teacher(teacher, X, A_norm, labels)
    acc_teacher = accuracy(teacher(X, A_norm), labels)
    print(f"Teacher GCN accuracy: {acc_teacher:.4f}")

    student = StudentMLP(X.shape[1], hidden, n_classes)
    train_student_distill(student, teacher, X, A_norm)
    acc_student = accuracy(student(X), labels)
    print(f"Student MLP accuracy (after distillation): {acc_student:.4f}")
