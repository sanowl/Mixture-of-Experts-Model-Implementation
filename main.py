import torch
import logging
from datasets import SequenceDataset
from models.moe import MoEClassifier
from training.train import train_model
from config import ModelConfig
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def main():
    # Hyperparameters
    config = ModelConfig(
        input_size=512,
        hidden_size=1024,
        output_size=10,
        num_experts=16,
        k=4,
        num_layers=4,
        dropout=0.1,
        temperature=0.1
    )
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = SequenceDataset(num_samples=10000, seq_length=20, input_size=config.input_size, num_classes=config.output_size)
    val_dataset = SequenceDataset(num_samples=2000, seq_length=20, input_size=config.input_size, num_classes=config.output_size)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Initialize model, loss function, optimizer, and scheduler
    model = MoEClassifier(config)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Train the model
    try:
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, config)
    except Exception as e:
        logger.exception(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    main()

