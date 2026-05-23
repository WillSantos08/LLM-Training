import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

# Create a tensor
scalar = torch.tensor(7)
vector = torch.tensor([7, 7])
MATRIX = torch.tensor([[7, 7],
                       [7, 7]])

TENSOR = torch.tensor([[[7, 7, 7],
                        [7, 7, 7]],
                        [[7, 7, 7],
                         [7, 7, 7]],
                         [[7, 7, 7],
                          [7, 7, 7]]])