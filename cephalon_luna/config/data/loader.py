"""
DataLoader creation utilities
"""

from typing import Optional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import ConversationDataset, collate_fn_pad


def create_dataloader(
    file_path: str,
    tokenizer_name: str = 'gpt2',
    batch_size: int = 8,
    max_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for conversation data
    
    Args:
        file_path: Path to JSON data file
        tokenizer_name: Name of tokenizer from HuggingFace
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    
    # Load tokenizer
    print(f"📦 Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print(f"📂 Loading data from: {file_path}")
    dataset = ConversationDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    if len(dataset) == 0:
        print("⚠ Dataset is empty!")
        return None
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_pad,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    print(f"✅ DataLoader created with {len(dataset)} examples, batch size {batch_size}")
    
    return dataloader, tokenizer