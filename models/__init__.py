from . import total3d, mgnet
from . import loss

method_paths = {
    'TOTAL3D': total3d,
    'MGNet': mgnet
}

__all__ = ['method_paths']