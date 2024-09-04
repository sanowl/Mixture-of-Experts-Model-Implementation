from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    output_size: int
    num_experts: int
    k: int
    num_layers: int
    dropout: float = 0.1
    temperature: float = 0.1

