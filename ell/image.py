#!/usr/bin/env python3

from .ells import *
from .utils import max0
from .errors import *
from PIL import Image

class ImageLike:
    """Mixin for image classes
    Only define some basic operators on images here.
    """
    @classmethod
    def open(cls, filename, *args, **kwargs):
        """open an image file, and read it as an 2d array
        
        Arguments:
            filename {str} -- file name of the image
        
        Returns:
            object of current cls
        """
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

    def to_image(self, mode=None, resize=False, exposure=False, masked=False):
        if mode is None:
            mode = self.mode
        if resize:
            obj = self.resize(min_index=np.array((0,0)))
        else:
            obj = self.copy()
        if exposure:
            obj = obj.exposure()
        if masked:
            np.putmask(obj, obj < 0, 0)
            np.putmask(obj, obj > 255, 255)
        return Image.fromarray(np.asarray(np.round(obj)).astype('uint8')).convert(mode)

    def imshow(self, *args, **kwargs):
        self.to_image(*args, **kwargs).show()

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
                L = x.reduce(filter, axis=0)
                V = L.reduce(filter.check(), axis=1)
                L = L.reduce(filter, axis=1)
                H = x.reduce(filter.check(), axis=0)
                D = H.reduce(filter.check(), axis=1)
                H = H.reduce(filter, axis=1)
                # L = x.reduce(filter)
                # H = x.reduce(filter.tensor(filter.g))
                # V = x.reduce(filter.g.tensor(filter))
                # D = x.reduce(filter.g)
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

    def apply_data(self, key, low_key=None):
        cpy = self._clone(with_tree=True)
        for node in cpy.all_nodes_itr():
            if node.tag.startswith('L'):
                if low_key:
                    node.data = low_key(node.data)
            else:
                node.data = key(node.data)
        return cpy

    def mallat_rec(self, dual_filter, level=0):
        # mallat rec.
        if self.depth() == 1:
            n1 = self.get_node(level+1)
            n11, n12, n13 = (self.get_node((level+1, k)) for k in range(1,4))
            return n1.data.expand(dual_filter)\
            + n11.data.expand(dual_filter.g.tensor(dual_filter))\
            + n12.data.expand(dual_filter.tensor(dual_filter.g))\
            + n13.data.expand(dual_filter.g)
        else:
            m = self.subtree(level+1).mallat_rec(dual_filter, level=level+1)
            n11, n12, n13 = (self.get_node((level+1, k)) for k in range(1,4))
            return m.expand(dual_filter)\
            + n11.data.expand(dual_filter.g.tensor(dual_filter))\
            + n12.data.expand(dual_filter.tensor(dual_filter.g))\
            + n13.data.expand(dual_filter.g)

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
        assert DimError(3, details='Make sure the array representing the image has 3 dim.')
        if max_index is None:
            max_index = min_index + np.array(array.shape[:-1]) - 1
        return ImageRGB(array, min_index=min_index, max_index=max_index)


class ImageGray(ImageLike, Ell2d):
    _mode = 'L'

    @classmethod
    def from_image(cls, image, min_index=np.array([0,0]), max_index=None):
        array = np.asarray(image, dtype=np.float64)
        if array.ndim == 3:
            array = np.dot(array, [0.299, 0.587, 0.114])
        if max_index is None:
            max_index = min_index + np.array(array.shape) - 1
        return cls(array, min_index=min_index, max_index=max_index)


class ImageRGBA(ImageLike, MultiEll2d):
    _mode = 'RGBA'
