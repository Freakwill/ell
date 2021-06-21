from .ells import *

class Filter(Ell1d):

    def __new__(cls, array, name="[z]", *args, **kwargs):
        obj = super().__new__(cls, array, *args, **kwargs)
        obj.name = name
        return obj

    @staticmethod
    def from_name(s):
        import pywt
        if s.startswith("db"):
            return Filter(array=pywt.Wavelet(s).filter_bank[0][::-1], name=s)
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
{super().__format__(spec)}
"""
        else:
            return f'{self.name}: {super().__format__(spec)}'

    def workon(self, signal, axis=None):
        if isinstance(signal, Ell1d):
            return (signal @ self).down_sample()
        elif isinstance(signal, Ell2d):
            obj = self.conv1d(signal, axis=0).down_sample()


class FilterBank(tuple):
    pass

if __name__ == '__main__':

    _sqrt3 = np.sqrt(3)
    _sqrt2 = np.sqrt(2)

    d_filter = Filter.from_name('db2')
    d_high_filter = d_filter.check()
    low_filter = Filter([_sqrt2/8, 3*_sqrt2/8, 3*_sqrt2/8, _sqrt2/8], name='bo2l');
    dual_low_filter = Filter([-_sqrt2/4, 3*_sqrt2/4, 3*_sqrt2/4,-_sqrt2/4], name='bo2h');
