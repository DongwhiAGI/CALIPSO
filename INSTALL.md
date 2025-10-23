# Environment

* **OS:** Windows 11
* **Python:** 3.10
* **GPU:** NVIDIA RTX 4060

# Quick Install

```bash
conda create -n calipso python=3.10 -y
conda activate calipso
```
```bash
pip install -r requirements.txt
```

## Verify the Virtual Environment

### Check for Dependency Conflicts

```bash
pip check
pip install pipdeptree
pipdeptree --warn fail
```

### Manual Import Smoke Tests

```bash
python scripts/smoke_imports.py
python scripts/smoke_imports.py --check-gpu --check-opengl --check-ffmpeg
python scripts/smoke_imports.py --strict-optional --check-opengl
```

#### (From Anaconda Prompt)

```bash
python -m scripts.smoke_imports
python -m scripts.smoke_imports --check-gpu --check-opengl --check-ffmpeg
python -m scripts.smoke_imports --strict-optional --check-opengl
```

# Verify GPU/CUDA

```python
!nvidia-smi

import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:",
```
