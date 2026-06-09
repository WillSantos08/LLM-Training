"""
Dataset class for conversation data
"""

import json
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.utils.data import Dataset


class ConversationDataset(Dataset):
    """
    Dataset for conversation pairs (question-answer)
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int,
        padding: str = 'max_length'
    ):
        """
        Args:
            file_path: Path to JSON file with conversation data
            tokenizer: Tokenizer from HuggingFace
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length' or 'longest')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.examples = []
        
        # Load data from JSON
        self._load_data(file_path)
    
    def _load_data(self, file_path: str):
        """
        Load conversation data from JSON file
        
        Expected format:
        [
            {
                "question": "...",
                "answer": "..."
            },
            ...
        ]
        
        Args:
            file_path: Path to JSON file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"⚠ Data file not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each conversation pair
            for item in data:
                question = item.get('question', '')
                answer = item.get('answer', '')
                
                if question and answer:
                    # Format as: "User: {question}\nAssistant: {answer}"
                    full_text = f"User: {question}\nAssistant: {answer}"
                    
                    # Tokenize
                    tokens = self.tokenizer.encode(
                        full_text,
                        max_length=self.max_length,
                        truncation=True
                    )
                    
                    if len(tokens) > 1:
                        self.examples.append(tokens)
            
            print(f"✅ Loaded {len(self.examples)} conversation examples")
        
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single example
        
        Args:
            idx: Index of example
        
        Returns:
            input_ids: Token input (shift by 1 from full text)
            target_ids: Token targets (shift by 1 from full text)
        """
        tokens = self.examples[idx]
        
        # Input is all tokens except last
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        
        # Target is all tokens except first (shifted by 1)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids


def collate_fn_pad(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function with dynamic padding
    
    Args:
        batch: List of (input_ids, target_ids) tuples
    
    Returns:
        Padded input and target tensors
    """
    inputs, targets = zip(*batch)
    
    # Find max length in batch
    max_len = max(len(inp) for inp in inputs)
    
    # Pad sequences
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(inputs, targets):
        pad_len = max_len - len(inp)
        
        # Pad input with 0s
        padded_inp = torch.cat([
            inp,
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        # Pad target with -100 (ignored in loss)
        padded_tgt = torch.cat([
            tgt,
            torch.full((pad_len,), -100, dtype=torch.long)
        ])
        
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)