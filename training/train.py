import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from training.utils import load_balancing_loss, router_z_loss, EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import logging

logger = logging.getLogger(__name__)

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    config,
    lb_weight: float = 0.01,
    z_weight: float = 0.001
) -> None:
    model.to(device)
    early_stopping = EarlyStopping(patience=5)
    writer = SummaryWriter(log_dir='runs/moe_experiment')

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, routing_logits_list = model(inputs)
            main_loss = criterion(logits, labels)
            lb_loss = load_balancing_loss(routing_logits_list, config.num_experts, config.temperature)
            z_loss = router_z_loss(routing_logits_list)
            total_loss = main_loss + lb_weight * lb_loss + z_weight * z_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_acc += (logits.argmax(dim=1) == labels).float().mean().item()
            progress_bar.set_postfix({'loss': total_loss.item(), 'acc': train_acc / (progress_bar.n + 1)})
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    writer.close()

def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_acc += (logits.argmax(dim=1) == labels).float().mean().item()
    return total_loss / len(data_loader), total_acc / len(data_loader)

