import torch
import numpy as np
import random

def set_seed(seed):
    """모든 난수 생성기 시드 고정"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #torch.backends.cuda.matmul.allow_tf32 = True
        #torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #torch.set_float32_matmul_precision('high')  # PyTorch 2.x
    np.random.seed(seed)
    random.seed(seed)