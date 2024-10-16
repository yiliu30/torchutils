def freeze_seed(seed=0):
    import random

    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    import numpy as np

    np.random.seed(seed)
