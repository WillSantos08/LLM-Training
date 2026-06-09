"""
Training logic for Cephalon Luna
"""

import torch
import torch.optim as optim
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader
import time


class Trainer:
    """
    Trainer class for the Transformer model
    """
    
    def __init__(self, model, config, device: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            model: Transformer model to train
            config: Configuration object
            device: Device to use (cpu, cuda, mps). Uses config.device if None
        """
        self.model = model
        self.config = config
        self.device = device or config.device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            weight_decay=config.weight_decay
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0
        self.current_step = 0
    
    def train_epoch(self, train_dataloader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_dataloader: DataLoader for training data
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, targets) in enumerate(train_dataloader):
            # Move to device
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            num_batches += 1
            self.current_step += 1
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(
                    f"Epoch {self.current_epoch + 1} | "
                    f"Batch {batch_idx + 1}/{len(train_dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg Loss: {avg_loss:.4f}"
                )
        
        avg_epoch_loss = total_loss / num_batches
        return avg_epoch_loss
    
    def validate(self, val_dataloader: DataLoader) -> float:
        """
        Validate on validation set
        
        Args:
            val_dataloader: DataLoader for validation data
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, targets in val_dataloader:
                # Move to device
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                logits, loss = self.model(input_ids, targets)
                
                # Statistics
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ):
        """
        Train the model
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            num_epochs: Number of epochs. Uses config.num_epochs if None
        """
        num_epochs = num_epochs or self.config.num_epochs
        best_val_loss = float('inf')
        
        print("\n" + "="*70)
        print("CEPHALON LUNA - TRAINING STARTED")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batches per epoch: {len(train_dataloader)}")
        print("="*70 + "\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_dataloader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                self.val_losses.append(val_loss)
            
            # Time
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch
            print("\n" + "-"*70)
            print(f"Epoch {epoch + 1}/{num_epochs} completed")
            print(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f}")
            print(f"Time: {epoch_time:.2f}s")
            print("-"*70 + "\n")
            
            # Save best model
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_model.pth")
                print(f"✅ New best model saved (val_loss: {best_val_loss:.4f})\n")
            
            # Save checkpoint every N epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("="*70 + "\n")
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint
        
        Args:
            filename: Checkpoint filename
        """
        # Ensure checkpoints directory exists
        checkpoints_dir = Path(self.config.checkpoints_path)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Full path to checkpoint file
        checkpoint_path = checkpoints_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.current_epoch = checkpoint['epoch']
        self.current_step = checkpoint['step']
        
        print(f"✅ Checkpoint loaded: {filepath}")
        print(f"   Epoch: {self.current_epoch}, Step: {self.current_step}")
    
    def save_model(self, filename: str):
        """
        Save trained model only (without optimizer state)
        
        Args:
            filename: Model filename
        """
        # Ensure trained_models directory exists
        models_dir = Path(self.config.trained_models_path)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Full path to model file
        model_path = models_dir / filename
        
        torch.save(self.model.state_dict(), model_path)
        print(f"✅ Model saved: {model_path}")
    
    def load_model(self, filepath: str):
        """
        Load trained model
        
        Args:
            filepath: Path to model file
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"✅ Model loaded: {filepath}")