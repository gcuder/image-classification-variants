from .efficient_net import EfficientNet
from .transformer import VisionTransformer

BACKBONES = {
    EfficientNet.__name__: EfficientNet,
    VisionTransformer.__name__: VisionTransformer
}


def backbones(name: str, **kwargs):
    try:
        backbone = BACKBONES[name](**kwargs)
        return backbone
    except KeyError:
        raise ValueError()
