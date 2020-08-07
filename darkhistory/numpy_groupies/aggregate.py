"""numpy-groupies aggregate function, numpy implementation."""

import numpy as np

"""Common helpers without certain dependencies."""

aggregate_common_doc = """
    See readme file at https://github.com/ml31415/numpy-groupies for a full
    description.  Below we reproduce the "Full description of inputs"
    section from that readme, note that the text below makes references to
    other portions of the readme that are not shown here.
    group_idx:
        this is an array of non-negative integers, to be used as the "labels"
        with which to group the values in ``a``. Although we have so far
        assumed that ``group_idx`` is one-dimesnaional, and the same length as
        ``a``, it can in fact be two-dimensional (or some form of nested
        sequences that can be converted to 2D).  When ``group_idx`` is 2D, the
        size of the 0th dimension corresponds to the number of dimesnions in
        the output, i.e. ``group_idx[i,j]`` gives the index into the ith
        dimension in the output
        for ``a[j]``.  Note that ``a`` should still be 1D (or scalar), with
        length matching ``group_idx.shape[1]``.
    a:
        this is the array of values to be aggregated.  See above for a
        simple demonstration of what this means.  ``a`` will normally be a
        one-dimensional array, however it can also be a scalar in some cases.
    func: default='sum'
        the function to use for aggregation.  See the section above for
        details. Note that the simplest way to specify the function is using a
        string (e.g. ``func='max'``) however a number of aliases are also
        defined (e.g. you can use the ``func=np.max``, or even ``func=max``,
        where ``max`` is the
        builtin function).  To check the available aliases see ``utils.py``.
    size: default=None
        the shape of the output array. If ``None``, the maximum value in
        ``group_idx`` will set the size of the output.  Note that for
        multidimensional output you need to list the size of each dimension
        here, or give ``None``.
    fill_value: default=0
        in the example above, group 2 does not have any data, so requires some
        kind of filling value - in this case the default of ``0`` is used.  If
        you had set ``fill_value=nan`` or something else, that value would
        appear instead of ``0`` for the 2 element in the output.  Note that
        there are some subtle interactions between what is permitted for
        ``fill_value`` and the input/output ``dtype`` - exceptions should be
        raised in most cases to alert the programmer if issue arrise.
    order: default='C'
        this is relevant only for multimensional output.  It controls the
        layout of the output array in memory, can be ``'F'`` for fortran-style.
    dtype: default=None
        the ``dtype`` of the output.  By default something sensible is chosen
        based on the input, aggregation function, and ``fill_value``.
    ddof: default=0
        passed through into calculations of variance and standard deviation
        (see above).
"""

funcs_common = 'first last len mean var std allnan anynan max min argmax argmin cumsum cumprod cummax cummin'.split()
funcs_no_separate_nan = frozenset(['sort', 'rsort', 'array', 'allnan', 'anynan'])


_alias_str = {
    'or': 'any',
    'and': 'all',
    'add': 'sum',
    'count': 'len',
    'plus': 'sum',
    'multiply': 'prod',
    'product': 'prod',
    'times': 'prod',
    'amax': 'max',
    'maximum': 'max',
    'amin': 'min',
    'minimum': 'min',
    'split': 'array',
    'splice': 'array',
    'sorted': 'sort',
    'asort': 'sort',
    'asorted': 'sort',
    'rsorted': 'sort',
    'dsort': 'sort',
    'dsorted': 'rsort',
}

_alias_builtin = {
    all: 'all',
    any: 'any',
    len: 'len',
    max: 'max',
    min: 'min',
    sum: 'sum',
    sorted: 'sort',
    slice: 'array',
    list: 'array',
}


def get_aliasing(*extra):
    """The assembles the dict mapping strings and functions to the list of
    supported function names:
            e.g. alias['add'] = 'sum'  and alias[sorted] = 'sort'
    This funciton should only be called during import.
    """
    alias = dict((k, k) for k in funcs_common)
    alias.update(_alias_str)
    alias.update((fn, fn) for fn in _alias_builtin.values())
    alias.update(_alias_builtin)
    for d in extra:
        alias.update(d)
    alias.update((k, k) for k in set(alias.values()))
    # Treat nan-functions as firstclass member and add them directly
    for key in set(alias.values()):
        if key not in funcs_no_separate_nan:
            key = 'nan' + key
            alias[key] = key
    return alias

aliasing = get_aliasing()


def get_func(func, aliasing, implementations):
    """ Return the key of a found implementation or the func itself """
    try:
        func_str = aliasing[func]
    except KeyError:
        if callable(func):
            return func
    else:
        if func_str in implementations:
            return func_str
        if func_str.startswith('nan') and \
                func_str[3:] in funcs_no_separate_nan:
            raise ValueError("%s does not have a nan-version".format(func_str[3:]))
        else:
            raise NotImplementedError("No such function available")
    raise ValueError("func %s is neither a valid function string nor a "
                     "callable object".format(func))


def check_boolean(x):
    if x not in (0, 1):
        raise ValueError("Value not boolean")


try:
    basestring  # Attempt to evaluate basestring
    def isstr(s):
        return isinstance(s, basestring)
except NameError:
    # Probably Python 3.x
    def isstr(s):
        return isinstance(s, str)

#############################################################################

"""Common helper functions for typing and general numpy tools."""

_alias_numpy = {
    np.add: 'sum',
    np.sum: 'sum',
    np.any: 'any',
    np.all: 'all',
    np.multiply: 'prod',
    np.prod: 'prod',
    np.amin: 'min',
    np.min: 'min',
    np.minimum: 'min',
    np.amax: 'max',
    np.max: 'max',
    np.maximum: 'max',
    np.argmax: 'argmax',
    np.argmin: 'argmin',
    np.mean: 'mean',
    np.std: 'std',
    np.var: 'var',
    np.array: 'array',
    np.asarray: 'array',
    np.sort: 'sort',
    np.nansum: 'nansum',
    np.nanprod: 'nanprod',
    np.nanmean: 'nanmean',
    np.nanvar: 'nanvar',
    np.nanmax: 'nanmax',
    np.nanmin: 'nanmin',
    np.nanstd: 'nanstd',
    np.nanargmax: 'nanargmax',
    np.nanargmin: 'nanargmin',
    np.cumsum: 'cumsum',
    np.cumprod: 'cumprod',
}

aliasing = get_aliasing(_alias_numpy)

_next_int_dtype = dict(
    bool=np.int8,
    uint8=np.int16,
    int8=np.int16,
    uint16=np.int32,
    int16=np.int32,
    uint32=np.int64,
    int32=np.int64
)

_next_float_dtype = dict(
    float16=np.float32,
    float32=np.float64,
    float64=np.complex64,
    complex64=np.complex128
)


def minimum_dtype(x, dtype=np.bool_):
    """returns the "most basic" dtype which represents `x` properly, which
    provides at least the same value range as the specified dtype."""

    def check_type(x, dtype):
        try:
            converted = dtype.type(x)
        except (ValueError, OverflowError):
            return False
        # False if some overflow has happened
        return converted == x or np.isnan(x)

    def type_loop(x, dtype, dtype_dict, default=None):
        while True:
            try:
                dtype = np.dtype(dtype_dict[dtype.name])
                if check_type(x, dtype):
                    return np.dtype(dtype)
            except KeyError:
                if default is not None:
                    return np.dtype(default)
                raise ValueError("Can not determine dtype of %r" % x)

    dtype = np.dtype(dtype)
    if check_type(x, dtype):
        return dtype

    if np.issubdtype(dtype, np.inexact):
        return type_loop(x, dtype, _next_float_dtype)
    else:
        return type_loop(x, dtype, _next_int_dtype, default=np.float32)


def minimum_dtype_scalar(x, dtype, a):
    if dtype is None:
        dtype = np.dtype(type(a)) if isinstance(a, (int, float))\
                                  else a.dtype
    return minimum_dtype(x, dtype)


_forced_types = {
    'array': np.object,
    'all': np.bool_,
    'any': np.bool_,
    'nanall': np.bool_,
    'nanany': np.bool_,
    'len': np.int64,
    'nanlen': np.int64,
    'allnan': np.bool_,
    'anynan': np.bool_,
    'argmax': np.int64,
    'argmin': np.int64,
}
_forced_float_types = {'mean', 'var', 'std', 'nanmean', 'nanvar', 'nanstd'}
_forced_same_type = {'min', 'max', 'first', 'last', 'nanmin', 'nanmax',
                     'nanfirst', 'nanlast'}


def check_dtype(dtype, func_str, a, n):
    if np.isscalar(a) or not a.shape:
        if func_str not in ("sum", "prod", "len"):
            raise ValueError("scalar inputs are supported only for 'sum', "
                             "'prod' and 'len'")
        a_dtype = np.dtype(type(a))
    else:
        a_dtype = a.dtype

    if dtype is not None:
        # dtype set by the user
        # Careful here: np.bool != np.bool_ !
        if np.issubdtype(dtype, np.bool_) and \
                not('all' in func_str or 'any' in func_str):
            raise TypeError("function %s requires a more complex datatype "
                            "than bool" % func_str)
        if not np.issubdtype(dtype, np.integer) and func_str in ('len', 'nanlen'):
            raise TypeError("function %s requires an integer datatype" % func_str)
        # TODO: Maybe have some more checks here
        return np.dtype(dtype)
    else:
        try:
            return np.dtype(_forced_types[func_str])
        except KeyError:
            if func_str in _forced_float_types:
                if np.issubdtype(a_dtype, np.floating):
                    return a_dtype
                else:
                    return np.dtype(np.float64)
            else:
                if func_str == 'sum':
                    # Try to guess the minimally required int size
                    if np.issubdtype(a_dtype, np.int64):
                        # It's not getting bigger anymore
                        # TODO: strictly speaking it might need float
                        return np.dtype(np.int64)
                    elif np.issubdtype(a_dtype, np.integer):
                        maxval = np.iinfo(a_dtype).max * n
                        return minimum_dtype(maxval, a_dtype)
                    elif np.issubdtype(a_dtype, np.bool_):
                        return minimum_dtype(n, a_dtype)
                    else:
                        # floating, inexact, whatever
                        return a_dtype
                elif func_str in _forced_same_type:
                    return a_dtype
                else:
                    if isinstance(a_dtype, np.integer):
                        return np.dtype(np.int64)
                    else:
                        return a_dtype


def check_fill_value(fill_value, dtype):
    try:
        return dtype.type(fill_value)
    except ValueError:
        raise ValueError("fill_value must be convertible into %s"
                         % dtype.type.__name__)


def check_group_idx(group_idx, a=None, check_min=True):
    if a is not None and group_idx.size != a.size:
        raise ValueError("The size of group_idx must be the same as "
                         "a.size")
    if not issubclass(group_idx.dtype.type, np.integer):
        raise TypeError("group_idx must be of integer type")
    if check_min and np.min(group_idx) < 0:
        raise ValueError("group_idx contains negative indices")


def input_validation(group_idx, a, size=None, order='C', axis=None,
                     ravel_group_idx=True, check_bounds=True):
    """ Do some fairly extensive checking of group_idx and a, trying to
    give the user as much help as possible with what is wrong. Also,
    convert ndim-indexing to 1d indexing.
    """
    if not isinstance(a, (int, float, complex)):
        a = np.asanyarray(a)
    group_idx = np.asanyarray(group_idx)

    if not np.issubdtype(group_idx.dtype, np.integer):
        raise TypeError("group_idx must be of integer type")

    # This check works for multidimensional indexing as well
    if check_bounds and np.any(group_idx < 0):
        raise ValueError("negative indices not supported")

    ndim_idx = np.ndim(group_idx)
    ndim_a = np.ndim(a)

    # Deal with the axis arg: if present, then turn 1d indexing into
    # multi-dimensional indexing along the specified axis.
    if axis is None:
        if ndim_a > 1:
            raise ValueError("a must be scalar or 1 dimensional, use .ravel to"
                             " flatten. Alternatively specify axis.")
    elif axis >= ndim_a or axis < -ndim_a:
        raise ValueError("axis arg too large for np.ndim(a)")
    else:
        axis = axis if axis >= 0 else ndim_a + axis  # negative indexing
        if ndim_idx > 1:
            # TODO: we could support a sequence of axis values for multiple
            # dimensions of group_idx.
            raise NotImplementedError("only 1d indexing currently"
                                      "supported with axis arg.")
        elif a.shape[axis] != len(group_idx):
            raise ValueError("a.shape[axis] doesn't match length of group_idx.")
        elif size is not None and not np.isscalar(size):
            raise NotImplementedError("when using axis arg, size must be"
                                      "None or scalar.")
        else:
            # Create the broadcast-ready multidimensional indexing.
            # Note the user could do this themselves, so this is
            # very much just a convenience.
            size_in = np.max(group_idx) + 1 if size is None else size
            group_idx_in = group_idx
            group_idx = []
            size = []
            for ii, s in enumerate(a.shape):
                ii_idx = group_idx_in if ii == axis else np.arange(s)
                ii_shape = [1] * ndim_a
                ii_shape[ii] = s
                group_idx.append(ii_idx.reshape(ii_shape))
                size.append(size_in if ii == axis else s)
            # Use the indexing, and return. It's a bit simpler than
            # using trying to keep all the logic below happy
            group_idx = np.ravel_multi_index(group_idx, size, order=order,
                                             mode='raise')
            flat_size = np.prod(size)
            ndim_idx = ndim_a
            return group_idx.ravel(), a.ravel(), flat_size, ndim_idx, size

    if ndim_idx == 1:
        if size is None:
            size = np.max(group_idx) + 1
        else:
            if not np.isscalar(size):
                raise ValueError("output size must be scalar or None")
            if check_bounds and np.any(group_idx > size - 1):
                raise ValueError("one or more indices are too large for "
                                 "size %d" % size)
        flat_size = size
    else:
        if size is None:
            size = np.max(group_idx, axis=1) + 1
        elif np.isscalar(size):
            raise ValueError("output size must be of length %d"
                             % len(group_idx))
        elif len(size) != len(group_idx):
            raise ValueError("%d sizes given, but %d output dimensions "
                             "specified in index" % (len(size),
                                                     len(group_idx)))
        if ravel_group_idx:
            group_idx = np.ravel_multi_index(group_idx, size, order=order,
                                             mode='raise')
        flat_size = np.prod(size)

    if not (np.ndim(a) == 0 or len(a) == group_idx.size):
        raise ValueError("group_idx and a must be of the same length, or a"
                         " can be scalar")

    return group_idx, a, flat_size, ndim_idx, size


### General tools ###


def unpack(group_idx, ret):
    """ Take an aggregate packed array and uncompress it to the size of group_idx. 
        This is equivalent to ret[group_idx].
    """
    return ret[group_idx]


def allnan(x):
    return np.all(np.isnan(x))


def anynan(x):
    return np.any(np.isnan(x))


def nanfirst(x):
    return x[~np.isnan(x)][0]


def nanlast(x):
    return x[~np.isnan(x)][-1]


def multi_arange(n):
    """By example:
    
        #    0  1  2  3  4  5  6  7  8
        n = [0, 0, 3, 0, 0, 2, 0, 2, 1]
        res = [0, 1, 2, 0, 1, 0, 1, 0]
    That is it is equivalent to something like this :
    
        hstack((arange(n_i) for n_i in n))
        
    This version seems quite a bit faster, at least for some
    possible inputs, and at any rate it encapsulates a task 
    in a function.
    """
    if n.ndim != 1:
        raise ValueError("n is supposed to be 1d array.")

    n_mask = n.astype(bool)
    n_cumsum = np.cumsum(n)
    ret = np.ones(n_cumsum[-1] + 1, dtype=int)
    ret[n_cumsum[n_mask]] -= n[n_mask]
    ret[0] -= 1
    return np.cumsum(ret)[:-1]


def label_contiguous_1d(X):
    """ 
    WARNING: API for this function is not liable to change!!!    
    
    By example:
        X =      [F T T F F T F F F T T T]
        result = [0 1 1 0 0 2 0 0 0 3 3 3]
    
    Or:
        X =      [0 3 3 0 0 5 5 5 1 1 0 2]
        result = [0 1 1 0 0 2 2 2 3 3 0 4]
    
    The ``0`` or ``False`` elements of ``X`` are labeled as ``0`` in the output. If ``X``
    is a boolean array, each contiguous block of ``True`` is given an integer
    label, if ``X`` is not boolean, then each contiguous block of identical values
    is given an integer label. Integer labels are 1, 2, 3,..... (i.e. start a 1
    and increase by 1 for each block with no skipped numbers.)
    
    """

    if X.ndim != 1:
        raise ValueError("this is for 1d masks only.")

    is_start = np.empty(len(X), dtype=bool)
    is_start[0] = X[0]  # True if X[0] is True or non-zero

    if X.dtype.kind == 'b':
        is_start[1:] = ~X[:-1] & X[1:]
        M = X
    else:
        M = X.astype(bool)
        is_start[1:] = X[:-1] != X[1:]
        is_start[~M] = False

    L = np.cumsum(is_start)
    L[~M] = 0
    return L


def relabel_groups_unique(group_idx):
    """
    See also ``relabel_groups_masked``.
    
    keep_group:  [0 3 3 3 0 2 5 2 0 1 1 0 3 5 5]
    ret:         [0 3 3 3 0 2 4 2 0 1 1 0 3 4 4]
    
    Description of above: unique groups in input was ``1,2,3,5``, i.e.
    ``4`` was missing, so group 5 was relabled to be ``4``.
    Relabeling maintains order, just "compressing" the higher numbers
    to fill gaps.
    """

    keep_group = np.zeros(np.max(group_idx) + 1, dtype=bool)
    keep_group[0] = True
    keep_group[group_idx] = True
    return relabel_groups_masked(group_idx, keep_group)


def relabel_groups_masked(group_idx, keep_group):
    """
    group_idx: [0 3 3 3 0 2 5 2 0 1 1 0 3 5 5]
   
                 0 1 2 3 4 5
    keep_group: [0 1 0 1 1 1]
    
    ret:       [0 2 2 2 0 0 4 0 0 1 1 0 2 4 4]
    
    Description of above in words: remove group 2, and relabel group 3,4, and 5
    to be 2, 3 and 4 respecitvely, in order to fill the gap.  Note that group 4 was never used
    in the input group_idx, but the user supplied mask said to keep group 4, so group
    5 is only moved up by one place to fill the gap created by removing group 2.
    
    That is, the mask describes which groups to remove,
    the remaining groups are relabled to remove the gaps created by the falsy
    elements in ``keep_group``.  Note that ``keep_group[0]`` has no particular meaning because it refers
    to the zero group which cannot be "removed".
    ``keep_group`` should be bool and ``group_idx`` int.
    Values in ``group_idx`` can be any order, and 
    """

    keep_group = keep_group.astype(bool, copy=not keep_group[0])
    if not keep_group[0]:  # ensuring keep_group[0] is True makes life easier
        keep_group[0] = True

    relabel = np.zeros(keep_group.size, dtype=group_idx.dtype)
    relabel[keep_group] = np.arange(np.count_nonzero(keep_group))
    return relabel[group_idx]

#############################################################################

def _sum(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype_scalar(fill_value, dtype, a)

    if np.ndim(a) == 0:
        ret = np.bincount(group_idx, minlength=size).astype(dtype)
        if a != 1:
            ret *= a
    else:
        if np.iscomplexobj(a):
            ret = np.empty(size, dtype=dtype)
            ret.real = np.bincount(group_idx, weights=a.real,
                              minlength=size)
            ret.imag = np.bincount(group_idx, weights=a.imag,
                              minlength=size)
        else:
            ret = np.bincount(group_idx, weights=a,
                              minlength=size).astype(dtype)

    if fill_value != 0:
        _fill_untouched(group_idx, ret, fill_value)
    return ret


def _prod(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype_scalar(fill_value, dtype, a)
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != 1:
        ret[group_idx] = 1  # product starts from 1
    np.multiply.at(ret, group_idx, a)
    return ret


def _len(group_idx, a, size, fill_value, dtype=None):
    return _sum(group_idx, 1, size, fill_value, dtype=int)


def _last(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    ret = np.full(size, fill_value, dtype=dtype)
    # repeated indexing gives last value, see:
    # the phrase "leaving behind the last value"  on this page:
    # http://wiki.scipy.org/Tentative_NumPy_Tutorial
    ret[group_idx] = a
    return ret


def _first(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    ret = np.full(size, fill_value, dtype=dtype)
    ret[group_idx[::-1]] = a[::-1]  # same trick as _last, but in reverse
    return ret


def _all(group_idx, a, size, fill_value, dtype=None):
    check_boolean(fill_value)
    ret = np.full(size, fill_value, dtype=bool)
    if not fill_value:
        ret[group_idx] = True
    ret[group_idx.compress(np.logical_not(a))] = False
    return ret


def _any(group_idx, a, size, fill_value, dtype=None):
    check_boolean(fill_value)
    ret = np.full(size, fill_value, dtype=bool)
    if fill_value:
        ret[group_idx] = False
    ret[group_idx.compress(a)] = True
    return ret


def _min(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmax = np.iinfo(a.dtype).max if issubclass(a.dtype.type, np.integer)\
        else np.finfo(a.dtype).max
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmax:
        ret[group_idx] = dmax  # min starts from maximum
    np.minimum.at(ret, group_idx, a)
    return ret


def _max(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or a.dtype)
    dmin = np.iinfo(a.dtype).min if issubclass(a.dtype.type, np.integer)\
        else np.finfo(a.dtype).min
    ret = np.full(size, fill_value, dtype=dtype)
    if fill_value != dmin:
        ret[group_idx] = dmin  # max starts from minimum
    np.maximum.at(ret, group_idx, a)
    return ret


def _argmax(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or int)
    dmin = np.iinfo(a.dtype).min if issubclass(a.dtype.type, np.integer)\
        else np.finfo(a.dtype).min
    group_max = _max(group_idx, a, size, dmin)
    is_max = a == group_max[group_idx]
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_max = group_idx[is_max]
    argmax, = is_max.nonzero()
    ret[group_idx_max[::-1]] = argmax[::-1]  # reverse to ensure first value for each group wins
    return ret


def _argmin(group_idx, a, size, fill_value, dtype=None):
    dtype = minimum_dtype(fill_value, dtype or int)
    dmax = np.iinfo(a.dtype).max if issubclass(a.dtype.type, np.integer)\
        else np.finfo(a.dtype).max
    group_min = _min(group_idx, a, size, dmax)
    is_min = a == group_min[group_idx]
    ret = np.full(size, fill_value, dtype=dtype)
    group_idx_min = group_idx[is_min]
    argmin, = is_min.nonzero()
    ret[group_idx_min[::-1]] = argmin[::-1]  # reverse to ensure first value for each group wins
    return ret


def _mean(group_idx, a, size, fill_value, dtype=np.dtype(np.float64)):
    if np.ndim(a) == 0:
        raise ValueError("cannot take mean with scalar a")
    counts = np.bincount(group_idx, minlength=size)
    if np.iscomplexobj(a):
        dtype = a.dtype  # TODO: this is a bit clumsy
        sums = np.empty(size, dtype=dtype)
        sums.real = np.bincount(group_idx, weights=a.real,
                                minlength=size)
        sums.imag = np.bincount(group_idx, weights=a.imag,
                                minlength=size)
    else:
        sums = np.bincount(group_idx, weights=a,
                           minlength=size).astype(dtype)

    with np.errstate(divide='ignore', invalid='ignore'):
        ret = sums.astype(dtype) / counts
    if not np.isnan(fill_value):
        ret[counts == 0] = fill_value
    return ret


def _var(group_idx, a, size, fill_value, dtype=np.dtype(np.float64),
         sqrt=False, ddof=0):
    if np.ndim(a) == 0:
        raise ValueError("cannot take variance with scalar a")
    counts = np.bincount(group_idx, minlength=size)
    sums = np.bincount(group_idx, weights=a, minlength=size)
    with np.errstate(divide='ignore'):
        means = sums.astype(dtype) / counts
        ret = np.bincount(group_idx, (a - means[group_idx]) ** 2,
                          minlength=size) / (counts - ddof)
    if sqrt:
        ret = np.sqrt(ret)  # this is now std not var
    if not np.isnan(fill_value):
        ret[counts == 0] = fill_value
    return ret


def _std(group_idx, a, size, fill_value, dtype=np.dtype(np.float64), ddof=0):
    return _var(group_idx, a, size, fill_value, dtype=dtype, sqrt=True,
                ddof=ddof)


def _allnan(group_idx, a, size, fill_value, dtype=bool):
    return _all(group_idx, np.isnan(a), size, fill_value=fill_value,
                dtype=dtype)


def _anynan(group_idx, a, size, fill_value, dtype=bool):
    return _any(group_idx, np.isnan(a), size, fill_value=fill_value,
                dtype=dtype)


def _sort(group_idx, a, size=None, fill_value=None, dtype=None, reverse=False):
    sortidx = np.lexsort((-a if reverse else a, group_idx))
    # Reverse sorting back to into grouped order, but preserving groupwise sorting
    revidx = np.argsort(np.argsort(group_idx, kind='mergesort'), kind='mergesort')
    return a[sortidx][revidx]


def _array(group_idx, a, size, fill_value, dtype=None):
    """groups a into separate arrays, keeping the order intact."""
    if fill_value is not None and not (np.isscalar(fill_value) or
                                       len(fill_value) == 0):
        raise ValueError("fill_value must be None, a scalar or an empty "
                         "sequence")
    order_group_idx = np.argsort(group_idx, kind='mergesort')
    counts = np.bincount(group_idx, minlength=size)
    ret = np.split(a[order_group_idx], np.cumsum(counts)[:-1])
    ret = np.asanyarray(ret)
    if fill_value is None or np.isscalar(fill_value):
        _fill_untouched(group_idx, ret, fill_value)
    return ret


def _generic_callable(group_idx, a, size, fill_value, dtype=None,
                      func=lambda g: g, **kwargs):
    """groups a by inds, and then applies foo to each group in turn, placing
    the results in an array."""
    groups = _array(group_idx, a, size, ())
    ret = np.full(size, fill_value, dtype=dtype or np.float64)

    for i, grp in enumerate(groups):
        if np.ndim(grp) == 1 and len(grp) > 0:
            ret[i] = func(grp)
    return ret


def _cumsum(group_idx, a, size, fill_value=None, dtype=None):
    """
    N to N aggregate operation of cumsum. Perform cumulative sum for each group.
    >>> group_idx = np.array([4, 3, 3, 4, 4, 1, 1, 1, 7, 8, 7, 4, 3, 3, 1, 1])
    >>> a = np.array([3, 4, 1, 3, 9, 9, 6, 7, 7, 0, 8, 2, 1, 8, 9, 8])
    >>> _cumsum(group_idx, a, np.max(group_idx) + 1)
    array([ 3,  4,  5,  6, 15,  9, 15, 22,  7,  0, 15, 17,  6, 14, 31, 39])
    """
    sortidx = np.argsort(group_idx, kind='mergesort')
    invsortidx = np.argsort(sortidx, kind='mergesort')
    group_idx_srt = group_idx[sortidx]

    a_srt = a[sortidx]
    a_srt_cumsum = np.cumsum(a_srt, dtype=dtype)

    increasing = np.arange(len(a), dtype=int)
    group_starts = _min(group_idx_srt, increasing, size, fill_value=0)[group_idx_srt]
    a_srt_cumsum += -a_srt_cumsum[group_starts] + a_srt[group_starts]
    return a_srt_cumsum[invsortidx]


def _nancumsum(group_idx, a, size, fill_value=None, dtype=None):
    a_nonans = np.where(np.isnan(a), 0, a)
    group_idx_nonans = np.where(np.isnan(group_idx), np.nanmax(group_idx) + 1, group_idx)
    return _cumsum(group_idx_nonans, a_nonans, size, fill_value=fill_value, dtype=dtype)


_impl_dict = dict(min=_min, max=_max, sum=_sum, prod=_prod, last=_last,
                  first=_first, all=_all, any=_any, mean=_mean, std=_std,
                  var=_var, anynan=_anynan, allnan=_allnan, sort=_sort,
                  array=_array, argmax=_argmax, argmin=_argmin, len=_len,
                  cumsum=_cumsum, generic=_generic_callable)
_impl_dict.update(('nan' + k, v) for k, v in list(_impl_dict.items())
                  if k not in funcs_no_separate_nan)


def _aggregate_base(group_idx, a, func='sum', size=None, fill_value=0,
                    order='C', dtype=None, axis=None, _impl_dict=_impl_dict,
                    _nansqueeze=False, cache=None, **kwargs):
    group_idx, a, flat_size, ndim_idx, size = input_validation(group_idx, a,
                                             size=size, order=order, axis=axis)
    func = get_func(func, aliasing, _impl_dict)
    if not isstr(func):
        # do simple grouping and execute function in loop
        ret = _impl_dict.get('generic', _generic_callable)(group_idx, a, flat_size, fill_value, func=func,
                                dtype=dtype, **kwargs)
    else:
        # deal with nans and find the function
        if func.startswith('nan'):
            if np.ndim(a) == 0:
                raise ValueError("nan-version not supported for scalar input.")
            if _nansqueeze:
                good = ~np.isnan(a)
                a = a[good]
                group_idx = group_idx[good]

        dtype = check_dtype(dtype, func, a, flat_size)
        func = _impl_dict[func]
        ret = func(group_idx, a, flat_size, fill_value=fill_value, dtype=dtype,
                   **kwargs)

    # deal with ndimensional indexing
    if ndim_idx > 1:
        ret = ret.reshape(size, order=order)
    return ret


def aggregate(group_idx, a, func='sum', size=None, fill_value=0, order='C',
              dtype=None, axis=None, **kwargs):
    return _aggregate_base(group_idx, a, size=size, fill_value=fill_value,
                           order=order, dtype=dtype, func=func, axis=axis,
                           _impl_dict=_impl_dict, _nansqueeze=True, **kwargs)

aggregate.__doc__ = """
    This is the pure numpy implementation of aggregate.
    """ + aggregate_common_doc



def _fill_untouched(idx, ret, fill_value):
    """any elements of ret not indexed by idx are set to fill_value."""
    untouched = np.ones_like(ret, dtype=bool)
    untouched[idx] = False
    ret[untouched] = fill_value