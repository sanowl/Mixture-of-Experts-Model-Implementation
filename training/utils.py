import torch
import torch.nn.functional as F
from typing import List

def load_balancing_loss(routing_logits_list: List[torch.Tensor], num_experts: int, temperature: float = 0.1) -> torch.Tensor:
    losses = []
    for routing_logits in routing_logits_list:
        route_probs = F.softmax(routing_logits / temperature, dim=-1)
        route_frac = route_probs.mean(dim=0)
        entropy = -(route_frac * torch.log(route_frac + 1e-10)).sum()
        losses.append(num_experts * ((route_frac - 1/num_experts)**2).sum() - 0.1 * entropy)
    return sum(losses) / len(routing_logits_list)

def router_z_loss(routing_logits_list: List[torch.Tensor]) -> torch.Tensor:
    return sum(((routing_logits.exp().sum(dim=-1) - 1)**2).mean() for routing_logits in routing_logits_list)

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

