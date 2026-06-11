import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


class LunaDataset(Dataset):
    """
    Para cada posição i:
        input  = tokens[i : i + context_len]
        target = tokens[i+1 : i + context_len + 1]
    """

    def __init__(self, token_ids: List[int], context_len: int):
        self.data        = torch.tensor(token_ids, dtype=torch.long)
        self.context_len = context_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_len + 1]
        return chunk[:-1], chunk[1:]


def build_loaders(
    token_ids:   List[int],
    context_len: int,
    batch_size:  int,
    val_ratio:   float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """Divide em treino/val e retorna DataLoaders."""

    split    = int(len(token_ids) * (1 - val_ratio))
    train_ds = LunaDataset(token_ids[:split], context_len)
    val_ds   = LunaDataset(token_ids[split:], context_len)

    train_dl = DataLoader(
        train_ds,
        batch_size = batch_size,
        shuffle    = True,
        drop_last  = True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size = batch_size,
        shuffle    = False,
        drop_last  = True,
    )

    print(f"  📦 Train : {len(train_ds):,} sequências")
    print(f"  📦 Val   : {len(val_ds):,} sequências")

    return train_dl, val_dl