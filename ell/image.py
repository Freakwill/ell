from .ells import *
from .utils import max0
from PIL import Image

class ImageLike:
    @classmethod
    def open(cls, filename, *args, **kwargs):
        return cls.from_image(Image.open(filename), *args, **kwargs)

    def minmaxmap(self, lb=0, ub=255):
        obj = np.asarray(self)
        mi = obj.min(axis=(0,1))
        ma = obj.max(axis=(0,1))
        return (ub - lb) / (ma - mi) * (self.copy() - mi) + lb

    def exposure(self, k=0.5):
        assert 0<=k<=1
        cpy = self.copy()
        return cpy + k/256 * (256-cpy) * cpy

    def to_image(self, mode=None, resize=False, exposure=False):
        if mode is None:
            mode = self.mode
        if resize:
            obj = self.resize(min_index=np.array((0,0)))
        else:
            obj = self.copy()
        if exposure:
            obj = np.round(obj.exposure())
        else:
            obj = np.round(obj)
        return Image.fromarray(np.asarray(obj).astype('uint8')).convert(mode)

    @property
    def mode(self):
        return self._mode

    def mallat_tensor(self, filter, level=2, tree=None, offset=0):
        """Mallat Algo of tensor wavelets

        Every time, it produces a branch with 4 nodes: low-band,
        horizontal-band, vertical-band, diagonal-band, under low-band
        in the previous level.

        Return a tree with nodes representing the coefs.
        """

        def _mt(x, filter, tree=None, offset=0):
            if offset==level:
                tree = WaveletTree(level=0)
                tree.create_node(identifier=offset, data=x)
                return tree
            else:
                tree = WaveletTree(level=level-offset)
                root = tree.create_node(identifier=offset, data=x)
                L = x.reduce(filter)
                H = x.reduce(filter.tensor(filter.check()))
                V = x.reduce(filter.check().tensor(filter))
                D = x.reduce(filter.check())
                tree.create_node(identifier=(offset+1, 1), data=H, parent=offset)
                tree.create_node(identifier=(offset+1, 2), data=V, parent=offset)
                tree.create_node(identifier=(offset+1, 3), data=D, parent=offset)
                subtree = _mt(L, filter, offset=offset+1)
                tree.paste(offset, subtree)
                return tree
        return _mt(self, filter, tree, offset=0)

from treelib import Node, Tree
class WaveletTree(Tree):
    """Quadtree struct of wavelet coefs
    
    Extends:
        Tree
    """

    def __init__(self, level=0, *args, **kwargs):
        super().__init__(node_class=WaveletNode, *args, **kwargs)
        self._level = level

class WaveletNode(Node):
    @property
    def tag(self):
        if isinstance(self.identifier, int):
            return f"L{self.identifier}"
        else:
            return f"{['H', 'V', 'D'][self.identifier[1]-1]}{self.identifier[0]}"


class ImageRGB(ImageLike, MultiEll2d):
    _mode = 'RGB'

    @staticmethod
    def from_image(image, min_index=np.array([0,0]), max_index=None):
        array = np.asarray(image, dtype=np.float64)
        assert array.ndim == 3, 'Make sure the array representing the image has 3 dim.'
        if max_index is None:
            max_index = min_index + np.array(array.shape[:-1]) - 1
        return ImageRGB(array, min_index=min_index, max_index=max_index)


class ImageGrey(ImageLike, Ell2d):
    _mode = 'L'
