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


def max0(x):
    return x * (x>0)


def common_index(*index_pairs):
    # common index, the minimal range containing all ranges
    mi, ma = tuple(zip(*index_pairs))
    return np.min(mi, axis=0), np.max(ma, axis=0)

