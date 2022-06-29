import torch
import torch.nn as nn
import numpy as np

# only for test, not required actually

x = torch.rand((10, 1, 512))
y = torch.rand((10, 1, 512))

torch.save(x, 'train_image.pt')
torch.save(y, 'train_text.pt')
