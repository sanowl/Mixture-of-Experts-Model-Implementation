# Mixture-of-Experts (MoE) Model

This project implements a **Mixture-of-Experts (MoE)** deep learning model using PyTorch. The model activates only a subset of experts at a time, optimizing computation for large models by activating a small percentage of parameters on a per-input basis.

## Features

- **Mixture-of-Experts (MoE) Model**: Uses a router to select a subset of experts for each input.
- **Configurable Model**: Model architecture, including the number of experts, layers, and routing strategies, can be easily modified via the `ModelConfig` class.
- **Attention Mechanism**: Multihead attention layers are incorporated for sequence modeling tasks.
- **Custom Loss Functions**: Includes load balancing loss and router z-loss to ensure efficient use of experts.
- **Early Stopping**: Integrated early stopping to prevent overfitting based on validation performance.
- **TensorBoard Support**: Logs training metrics, including loss and accuracy, for monitoring via TensorBoard.

## Installation

   ```

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Dataset**:
   - The project uses a `SequenceDataset` class that generates random data for testing. Modify the dataset generation in `datasets.py` if you'd like to use real datasets.

2. **Configuration**:
   - You can modify the model parameters in `config.py`. The `ModelConfig` class allows you to set input sizes, hidden sizes, the number of experts, etc.

3. **Running the Training Script**:
   - To train the model, run the `main.py` script:
     ```bash
     python main.py
     ```

4. **Monitoring with TensorBoard**:
   - Training logs are saved under the `runs/` directory. You can start TensorBoard to visualize metrics:
     ```bash
     tensorboard --logdir=runs
     ```

## Model Configuration

The model configuration can be customized via the `ModelConfig` dataclass in `config.py`. Key parameters include:

- `input_size`: The size of the input features.
- `hidden_size`: The size of hidden layers and expert dimensions.
- `output_size`: The number of output classes.
- `num_experts`: The total number of experts in the MoE model.
- `k`: The number of experts activated per input.
- `num_layers`: The number of MoE layers.
- `dropout`: Dropout rate for regularization.
- `temperature`: Temperature parameter for softmax routing.

## Training and Evaluation

- The `train_model` function handles the training process, including logging and evaluation on the validation set.
- Early stopping is implemented to stop training when validation loss stops improving.
- The model is saved to `moe_model.pth` once training is complete.

## Key Components

- **Router**: Selects the top-k experts based on the input.
- **Expert**: Individual fully connected feed-forward networks.
- **MoE Layer**: A combination of a router and several experts.
- **Attention Layer**: Multihead attention for sequence-based tasks.

## Future Work

- Extend the model for use with larger, real-world datasets.
- Improve memory efficiency for large-scale training.
- Experiment with different routing strategies or task-specific MoE models.


