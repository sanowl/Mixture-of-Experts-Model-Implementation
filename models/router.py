import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Router(nn.Module):
    def __init__(self, input_size: int, num_experts: int, k: int, temperature: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(input_size, num_experts)
        self.k = k
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routing_logits = self.gate(x) / self.temperature
        routing_probs = F.softmax(routing_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(routing_probs, self.k, dim=-1)
        top_k_probs_normalized = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        return routing_logits, top_k_probs_normalized, top_k_indices

class Expert(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

