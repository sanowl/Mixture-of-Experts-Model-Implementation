import torch
import torch.nn as nn
from typing import List, Tuple
from models.router import Router, Expert
from config import ModelConfig

class MoELayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_size = config.hidden_size
        self.output_size = config.hidden_size
        self.num_experts = config.num_experts
        self.k = config.k
        
        self.router = Router(self.input_size, config.num_experts, config.k, config.temperature)
        self.experts = nn.ModuleList([Expert(self.input_size, config.hidden_size, self.output_size, config.dropout) for _ in range(config.num_experts)])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        routing_logits, top_k_probs, top_k_indices = self.router(x)
        x = x.view(-1, self.input_size)
        top_k_indices = top_k_indices.view(-1, self.k)
        top_k_probs = top_k_probs.view(-1, self.k)
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        expert_outputs = expert_outputs.transpose(0, 1)
        selected_outputs = torch.gather(expert_outputs, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, self.output_size))
        combined_output = (selected_outputs * top_k_probs.unsqueeze(-1)).sum(dim=1)
        combined_output = combined_output.view(batch_size, seq_len, self.output_size)
        return combined_output, routing_logits

class MoEModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_projection = nn.Linear(config.input_size, config.hidden_size)
        self.layers = nn.ModuleList([MoELayer(config) for _ in range(config.num_layers)])
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(config.hidden_size, num_heads=8, batch_first=True) for _ in range(config.num_layers - 1)])
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.input_projection(x)
        routing_logits_list = []
        for i, (layer, attn) in enumerate(zip(self.layers[:-1], self.attention_layers)):
            residual = x
            x, routing_logits = layer(x)
            x, _ = attn(x, x, x)
            x = self.layer_norm(residual + self.dropout(x))
            routing_logits_list.append(routing_logits)
        x, routing_logits = self.layers[-1](x)
        routing_logits_list.append(routing_logits)
        return x, routing_logits_list

class MoEClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.moe = MoEModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.output_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        moe_output, routing_logits_list = self.moe(x)
        logits = self.classifier(moe_output[:, -1, :])
        return logits, routing_logits_list

