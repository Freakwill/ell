from .ells import *

class Filter(Ell1d):
    # only support 1d filter currently
    # not different with Ell1d essentially

    def __new__(cls, array, name="[z]", *args, **kwargs):
        obj = super().__new__(cls, array, *args, **kwargs)
        obj.name = name
        obj._g = None
        return obj

    @staticmethod
    def from_name(s):
        import pywt
        if s.startswith("db"):
            f = Filter(array=pywt.Wavelet(s).filter_bank[0][::-1], name=s)
            return f
        else:
            raise ValueError(f'`s` has an invalid value.')

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if isinstance(obj, Filter):
            self.name = obj.name
        elif not hasattr(self, "name"):
            self.name = ""

    def __format__(self, spec=None):
        if spec in {'full', 'f'}:
            return f"""Name: {self.name}
{super().__format__(spec)}"""
        else:
            return f'{self.name}: {super().__format__(spec)}'

    def workon(self, signal, axis=None):
        if isinstance(signal, Ell1d):
            return (signal @ self).down_sample()
        elif isinstance(signal, Ell2d):
            return signal.conv1d(self, axis=axis).down_sample()

    def scaling_function(self, level=10):
        assert level >=7, 'level is not enough'
        return 2**(level/2) * np.asarray(compose(self, level)), self.irange

    def wavelet(self, level=10):
        assert level >=7, 'level is not enough'
        return 2**((level+1)/2) * np.asarray(self.check().up_sample(2**level) @ compose(self, level)), self.irange

    def scaling_wavelet(self, level=10):
        assert level >=7, 'level is not enough'
        t = 2**(level/2) * compose(self, level-1)
        phi = np.asarray(t.up_sample() @ self)
        psi = np.asarray(self.check().up_sample(2**(level-1)) @ t)
        return phi, psi, self.irange, self.mirror_range

    def tensor(self, other=None):
        return Filter2d.view(super().tensor(other))


class Filter2d(Ell2d):
    pass


class TensorFilter(Filter2d):
    pass


class FilterBank(tuple):
    pass

if __name__ == '__main__':

    _sqrt3 = np.sqrt(3)
    _sqrt2 = np.sqrt(2)

    d_filter = Filter.from_name('db2')
    d_high_filter = d_filter.check()
    low_filter = Filter([_sqrt2/8, 3*_sqrt2/8, 3*_sqrt2/8, _sqrt2/8], name='bo2l');
    dual_low_filter = Filter([-_sqrt2/4, 3*_sqrt2/4, 3*_sqrt2/4,-_sqrt2/4], name='bo2h');
