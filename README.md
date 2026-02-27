# Gated-ls-block
A plug-and-play gated large-separable convolution (LSConv) block for HRNet backbone in MMPose. This module replaces the standard BasicBlock with a gated linear unit (GLU) mechanism and custom LSConv, optimized for high-resolution pose estimation tasks.

## Installation
1. Download `gated_block.py` into your MMPose directory: mmpose/models/backbones/
2. Pull or copy the dependency file `lsnet.py` into the **same directory** above.
3. Register the module in `mmpose/models/backbones/__init__.py`:
```python
from .gated_block import GatedLSBlock
__all__ = [..., 'GatedLSBlock']
Usage
In your MMPose config file, replace the block parameter in the HRNet backbone with GatedLSBlock：
model = dict(
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage2=dict(block='GatedLSBlock'),
            stage3=dict(block='GatedLSBlock'),
            stage4=dict(block='GatedLSBlock'),
        )
    )
)
## Dependency
numpy>=1.21
matplotlib>=3.4
scipy>=1.7
opencv-python>=4.5
torch>=1.10
mmpose>=1.0.0 
