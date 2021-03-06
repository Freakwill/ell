import numpy as np

def altArray(min_index, max_index):
    # alternative array
    A = np.array(1)
    for a, b in zip(min_index, max_index):
        N = b - a +1
        if N > 1:
            if N%2 == 0:
                if a % 2==0:
                    A = [A, -A] * (N//2)
                else:
                    A = [-A, A] * (N//2)
            else:
                if a % 2==0:
                    A = [A, -A] * (N//2) + [A]
                else:
                    A = [-A, A] * (N//2) + [-A]
        elif N == 1:
            if a % 2 == 0:
                A = [A]
            else:
                A = [-A]
        else:
            raise ValueError('Please make sure `min-index`<=`max-index`!')
        A = np.array(A)
    return A

def alt_sequences(k, start=1):
    a, r = divmod(k, 2)
    seq = np.tile([1, -1], a)
    if r:
        seq = np.append(seq, 1)
    return -seq if start ==-1 else seq

def altArray1d(min_index, max_index):
    # alternative array

    N = max_index - min_index +1
    return alt_sequences(N, -min_index%2)


def divround(x, k=2):
    if k == 1:
        return x, r
    if x<0:
        d, r = divmod(-x, k)
        d = -d
    else:
        d, r = divmod(x, k)
    return d, r


def monomial(c, k=1, var='z'):
    if c == 0:
        return '0'
    if k == 0:
        return str(c)
    elif k == 1:
        v = var
    else:
        v = f'{var}^{k}'
    if c==1:
        return v
    elif c==-1:
        return f'-{v}'
    else:
        return f'{c} {v}'

def compose(weight, level=1):
    # Calculate U^{n-1}h...Uh h
    if level == 1:
        return weight
    elif level > 1:
        return compose(weight, level-1).up_sample() @ weight
    else:
        raise Exception('`level` is a integer >=1.')

def max0(x, threshold=0):
    return x * (x>threshold)


def common_index(*index_pairs):
    # common index, the minimal range containing all ranges
    mi, ma = tuple(zip(*index_pairs))
    return np.min(mi, axis=0), np.max(ma, axis=0)

def shared_index(*index_pairs):
    # shared index, the maximal range contained in all ranges
    mi, ma = tuple(zip(*index_pairs))
    return np.max(mi, axis=0), np.min(ma, axis=0)

def replace_tuple(t, v, axis=None):
    if axis is None:
        return v
    else:
        a = np.array(t)
        if isinstance(axis, int):
            a[axis]=v
        else:
            a[list(axis)]=v
        return tuple(a)

def op_tuple(t, op, axis=None):
    if axis is None:
        return op(t)
    else:
        a = np.array(t)
        if isinstance(axis, int):
            a[axis] = op(a[axis])
        else:
            a[list(axis)] = op(a[list(axis)])
        return list(a)

def inc_tuple(t, v, axis=None):
    # same to op_tuple(t, lambda x: np.add(a, v), axis)
    if axis is None:
        return np.add(t, v)
    else:
        a = np.array(t)
        if isinstance(axis, int):
            a[axis]+=v
        else:
            a[list(axis)]+=v
        return tuple(a)

def is_index(i):
    return isinstance(i, (int, np.int64))


def equal_ndim(x, y):
    return isinstance(x, np.ndarray) and isinstance(x, np.ndarray) and x.ndim == y.ndim


# def index2slice(min_index, max_index):
#     return tuple(slice(0, ma-mi+1) for mi, ma in zip(min_index, max_index))

