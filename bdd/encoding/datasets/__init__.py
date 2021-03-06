from .base import *
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CityscapesSegmentation
from .bdd100k_drivable import BDD100K_Drivable
from .bdd100k_seg import BDD100K_Seg

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'cityscapes': CityscapesSegmentation,
    'bdd100k_drivable': BDD100K_Drivable,
    'bdd100k_seg': BDD100K_Seg,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
