
class EllException(Exception):
    def __str__(self):
        return self.message

    # def __repr__(self):
    #     return f'{self.__class__}: {self.message}'


class DimError(EllException):
    def __init__(self, ndim=1, details='', *args, **kwargs):
        self.ndim = ndim
        self.message = f'The dim of the underlying array should be {self.ndim}!\n{details}'
        super().__init__(self.message, *args, **kwargs)


class IndexUnavailableError(EllException):
    message = "Ell objects must have min_index/max_index attritubes with validated values! You have to provide min_index or max_index in some way!"
