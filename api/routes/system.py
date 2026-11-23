"""Local system and model information."""

import torch
import feat
from fastapi import APIRouter

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("")
def system_info():
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "py_feat_version": getattr(feat, "__version__", "unknown"),
    }
