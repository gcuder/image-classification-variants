from .efficient_net import EfficientNet

BACKBONES = {
    EfficientNet.__name__: EfficientNet
}


def backbones(name: str, **kwargs):
    try:
        backbone = BACKBONES[name](**kwargs)
        return backbone
    except KeyError:
        raise ValueError()
