import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class ClipDatasetOptimized(Dataset):
    """
    미리 로드된 전체 ndarray에서 클립을 생성하는 데이터셋.
    __getitem__에서 텐서 변환을 수행하여 데이터 로딩 병목을 최소화.
    """
    def __init__(self, X_data, y_data, clip_length, clip_stride):
        super().__init__()
        # NumPy 배열을 그대로 유지 (초기화 속도 향상 및 메모리 효율)
        self.X_data = X_data 
        self.y_data = y_data
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        
        self.indices = []
        num_frames = X_data.shape[0]
        for i in range(0, num_frames - self.clip_length + 1, self.clip_stride):
            self.indices.append(i)
            
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start_frame = self.indices[idx]
        end_frame = start_frame + self.clip_length
        
        # 여기서 필요한 만큼만 잘라서 Tensor로 변환
        clip_np = self.X_data[start_frame:end_frame]
        clip = torch.from_numpy(clip_np).float() # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3) # (C, T, H, W)
        
        # 라벨도 여기서 Tensor로 변환
        label_np = self.y_data[end_frame - 1]
        label = torch.from_numpy(label_np).long()
        
        return clip, label

def build_dataloaders_from_arrays(
    X_train,
    y_train,
    X_valid,
    y_valid,
    cfg: dict,
    device: torch.device,
) -> dict[str, DataLoader]:
    clip_len   = int(cfg["input_spec"]["clip_length"])
    clip_stride= int(cfg["input_spec"]["clip_stride"])
    batch_size = int(cfg["train"]["batch_size"])
    num_workers= int(cfg["train"]["num_workers"])
    prefetch    = int(cfg["train"].get("prefetch_factor", 4))

    train_dataset = ClipDatasetOptimized(X_train, y_train, clip_len, clip_stride)
    valid_dataset = ClipDatasetOptimized(X_valid, y_valid, clip_len, clip_stride)

    # 공통 kwargs
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if device.type == "cuda":
        loader_kwargs["pin_memory_device"] = "cuda"
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=prefetch,
        )
    else:
        loader_kwargs["persistent_workers"] = False  # prefetch_factor는 추가하지 않음

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    print(f"✅ 데이터셋 준비 완료: Train {len(train_dataset)}개, Validation {len(valid_dataset)}개 클립")
    return {"train": train_loader, "val": valid_loader}