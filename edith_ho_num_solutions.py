from __future__ import division, print_function, absolute_import

""" In this exercise, you will specify an array by providing a
1-dimensional list of numbers and a tuple containing the shape of the array.

For example, the following

    >>> data = range(8)
    >>> shape = (2, 2, 2)

corresponds to the following ndarray

    >>> import numpy as np
    >>> x = np.arange(8)
    >>> x.shape = (2, 2, 2)

Note that (as in part of homework 1) you will not be allowed to use NumPy
in your implementation.  Specifically, you should only use Python builtins,
`product`, and `reduce`.  I've already imported product and reduce.  You
should not add any additional imports.  The purpose of this assignment is
to have you implement everything by hand (with the exception of product
and reduce).

Your task is to implement some of the functionality of NumPy's arrays.  To do
this, you will be representing a multidimensional array as a (linear) 1-D list
of numbers.  You will store the multidimensional layout of the data array in a
shape tuple.  There are two standard ways to fill-in the multidimensional
structure specified by shape from the linear order of the list of
numbers---row-major and column-major order.

You will be using row-major ordering (NumPy's default) for this exercise.
So you will need to understand what that means.

Consider this example:

    >>> x
    array([[[0, 1],
            [2, 3]],

           [[4, 5],
            [6, 7]]])
    >>> x[0, 0, 0]
    0
    >>> x[0, 0, 1]
    1
    >>> x[0, 1, 0]
    2
    >>> x[0, 1, 1]
    3
    >>> x[1, 0, 0]
    4
    >>> x[1, 0, 1]
    5
    >>> x[1, 1, 0]
    6
    >>> x[1, 1, 1]
    7

Look carefully at the pattern.  Notice that the axis on the far right
position increments the fastest.  Then the axis next to it increments
a little slower.

Let's look at a bigger example.

    >>> x = np.arange(64)
    >>> x.shape = (2, 4, 4, 2)
    >>> x[0, 0, 0, 0]
    0
    >>> x[0, 0, 0, 1]
    1
    >>> x[0, 0, 1, 0]
    2
    >>> x[0, 0, 2, 0]
    4
    >>> x[0, 0, 3, 0]
    6
    >>> x[0, 1, 0, 0]
    8
    >>> x[0, 2, 0, 0]
    16
    >>> x[0, 3, 0, 0]
    24
    >>> x[1, 0, 0, 0]
    32

Notice that when you cycle on the ith axis each increase is in a fixed amount,
depending on the shape of the array and the position in the index that is
incrementing.

To read more about row-major order see:

    https://en.wikipedia.org/wiki/Row-major_order

Below you will find several functions involving working on arrays
represented as a list of numbers and a tuple containing the shape of the
array.  As a result, the first two arguments to many of the functions will
normally correspond to this information.
"""

# this might be useful in the `get_increment` function (more information there)
from functools import reduce

# this might be useful in the `extract` function (more information there)
from itertools import product


def size(shape, axis=None):
    """
    Return the number of elements along a given axis.

    Parameters
    ----------
    shape : tuple
        The shape of the input array
    axis : int, optional
        Axis along which the elements are counted.  By default, give
        the total number of elements.

    Returns
    -------
    element_count : int
        Number of elements along the specified axis.

    Examples
    --------
    >>> shape = (2, 3, 2)
    >>> size(shape)
    12
    >>> size(shape, axis=0)
    2
    >>> size(shape, axis=1)
    3
    """
    product = 1
    if axis in range(len(shape)):
        return shape[axis]
    else:
        for x in shape:
            product *= x
        return product


def ndim(shape):
    """
    Return the number of dimensions of an array.

    Parameters
    ----------
    shape : tuple
        The shape of the input array.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`.

    Examples
    --------
    >>> ndim((2, 2, 2))
    3
    >>> ndim((2,))
    1
    """
    return len(shape)


def reshape(data, newshape):
    """
    Gives a new shape to an array without changing its data.

    Parameters
    ----------
    data : list
        Array to be reshaped.
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.

    Returns
    -------
    newshape : tuple
        This will be the passed in shape if possible, otherwise return None

    Examples
    --------
    >>> reshape(range(8), (2, 2)) is None
    True
    >>> reshape(range(8), (2, 4))
    (2, 4)
    >>> reshape(range(8), (8,))
    (8,)
    """
    if size(newshape) == len(data):
        return newshape
    return None

def is_valid_index(shape, index):
    """
    Check whether the index is compatible with the shape.

    Parameters
    ----------
    shape : tuple
        The shape of the input array
    index : tuple
        The position of an element.

    Returns
    -------
    valid : bool
        True is the index is valid for the given shape, False otherwise

    Examples
    --------

    >>> is_valid_index((2,2), (1,3))
    False
    >>> is_valid_index((2,2), (1,1))
    True
    """
    for i in range(len(shape)):
        if zip(shape, index)[i][0] < zip(shape, index)[i][1]:
            return False
    return True


def get_increment(shape):
    """
    Return the increments corresponding to each axis or dimension in the shape.

    Parameters
    ----------
    shape : tuple
        The shape of the input array.

    Returns
    -------
    increment_per_axis : list
        The number of positions in the linear order that you need to move
        to retrieve the element specified by incrementing each axis
        in the corresponding index.

    Note
    ----
    You will need to understand row-major ordering to make sense of
    this function.  You may want to reread the module docstring if
    you are unsure of what to do.  Pay attention to the increment
    along the linear array corresponding to incrementing the index
    for each axis.

    You may also wish to use `reduce`, which has already been imported from
    functools above.

    Examples
    --------
    >>> get_increment((2, 4))
    [4, 1]
    >>> get_increment((2, 2, 2))
    [4, 2, 1]
    >>> get_increment((2, 4, 3))
    [12, 3, 1]
    """
    return [reduce(lambda x,y: x*y, shape[i:]) for i in range(1,len(shape))] + [shape[0] - 1]
    


def get_position(shape, index):
    """
    Return the position in the linear order specified by the given index in
    the multidimensional array.

    Parameters
    ----------
    shape : tuple
        The shape of the input array
    index : tuple
        The index position in the multidimensional array

    Returns
    -------
    position : int
        The position in the linear order specified by the index

    Notes
    -----
    You only have to implement this using row major order:
     https://en.wikipedia.org/wiki/Row-major_order

    Make sure the index is valid before using it.  For example, you could
    use the function you wrote above:

    assert is_valid_index(shape, index)

    Examples
    --------
    >>> shape = (2,)
    >>> get_position(shape, (0,))
    0
    >>> shape = (2, 2)
    >>> get_position(shape, (0, 0))
    0
    >>> get_position(shape, (0, 1))
    1
    >>> get_position(shape, (1, 0))
    2
    >>> get_position(shape, (1, 1))
    3
    """
    assert is_valid_index(shape, index)
    if len(shape) > 1:
        return reduce(lambda x,y: x+y,[x * get_increment(shape)[i] for i,x in enumerate(index[:-1])]) + index[-1]
    return index[0]

def get_index(shape, position):
    """
    Return the index in the multidimensional array that is corresponds to the
    element in the given position in the linear ordered.

    Parameters
    ----------
    shape : tuple
        The shape of the input array
    position : int
        The position in the linear ordering

    Returns
    -------
    index : tuple (same number of elements as shape)
        The index in the multidimensional array specified by the
        position in linear order

    Notes
    -----
    You only have to implement this using row major order:
     https://en.wikipedia.org/wiki/Row-major_order

    Make sure to check that the position is valid:

    assert position < size(shape)

    Exercises
    ---------
    >>> shape = (2,)
    >>> get_index(shape, 1)
    (1,)
    >>> shape = (2, 2, 2)
    >>> get_index(shape, 4)
    (1, 0, 0)
    >>> get_index(shape, 2)
    (0, 1, 0)
    >>> get_index((4, 5, 2, 1, 3), 17)
    (0, 2, 1, 0, 2)
    """
    assert position < size(shape)
    indexList = []
    increments = get_increment(shape)
    for increment in increments[:-1]:
        counter = 0
        while position >= increment:
            position -= increment
            counter += 1
        indexList.append(counter)
    indexList.append(position)
    return tuple(indexList)


def get_item(data, shape, index):
    """
    Return the value of the array at the given index.

    Parameters
    ----------
    data : list
        Input data.
    shape : tuple
        The shape of the input array
    index : tuple
        The index of the element to be returned.

    Returns
    -------
    element : number
        Number occurring at the position specified by the index.

    Examples
    --------
    >>> data = range(8)
    >>> shape = (2, 2, 2)
    >>> get_item(data, shape, (0, 0, 0))
    0
    >>> get_item(data, shape, (0, 0, 1))
    1
    >>> get_item(data, shape, (0, 1, 0))
    2
    >>> get_item(data, shape, (0, 1, 1))
    3
    >>> get_item(data, shape, (1, 0, 0))
    4
    >>> get_item(data, shape, (1, 0, 1))
    5
    """
    return data[get_position(shape, index)]


def nonzero(data, shape):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of `data`, containing
    the indices of the non-zero elements in that dimension.

    Parameters
    ----------
    data : list
        Input data.
    shape : tuple
        The shape of the input array

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    Examples
    --------
    >>> data = range(8)
    >>> shape = (2, 2, 2)
    >>> nonzero(data, shape)
    ((0, 0, 0, 1, 1, 1, 1), (0, 1, 1, 0, 0, 1, 1), (1, 0, 1, 0, 1, 0, 1))
    >>> [get_item(data, shape, index) for index in zip(*nonzero(data, shape))]
    [1, 2, 3, 4, 5, 6, 7]
    >>> d = [n % 2 for n in data]
    >>> shape = reshape(d, (2, 4))
    >>> nonzero(d, shape)
    ((0, 0, 1, 1), (1, 3, 1, 3))
    >>> [get_item(data, shape, index) for index in zip(*nonzero(d, shape))]
    [1, 3, 5, 7]
    """
    positionList = []
    indexList = []
    for i in range(len(data)):
        if data[i] > 0:
            positionList.append(range(len(data))[i])
    for position in positionList:
        indexList.append(get_index(shape, position))
    return tuple(zip(*indexList))


def extract(data, shape, axis, element):
    """
    Extract the subarray at the given element along the given axis.

    Parameters
    ----------
    data : list
        Input data.
    shape : tuple
        The shape of the input array
    axis : int
        Axis along which the subarray is extracted
    element : int
        Element identifying the subarray to be extracted

    Returns
    -------
    new array: (newdata, newshape)

    Notes
    -----
    Here are some examples using NumPy to help illustrate:

    In [1]: import numpy as np

    In [2]: y = np.arange(32)

    In [3]: y.shape = (2, 4, 4)

    In [4]: y
    Out[4]:
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]],

           [[16, 17, 18, 19],
            [20, 21, 22, 23],
            [24, 25, 26, 27],
            [28, 29, 30, 31]]])

    The following would be specified by `axis=0` and `element=1`

    In [5]: y[1, :, :]
    Out[5]:
    array([[16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])

    The following would be specified by `axis=1` and `element=1`

    In [6]: y[:, 1, :]
    Out[6]:
    array([[ 4,  5,  6,  7],
           [20, 21, 22, 23]])

    The following would be specified by `axis=2` and `element=3`

    In [7]: y[:, :, 3]
    Out[7]:
    array([[ 3,  7, 11, 15],
           [19, 23, 27, 31]])

    Hint
    ----
    Take a look at the docstring for itertools.product.  You can use it to
    form the Cartesian product of a list of tuples (e.g., if the input shape
    was (2, 3, 4) with `axis=1` and `element=1`, you might want to form the
    Cartesian product of [range(2), (1,), range(4)]).

    Examples
    --------
    >>> data = range(32)
    >>> shape = (2, 4, 4)
    >>> d, s = extract(data, shape, 2, 3)
    >>> d
    [3, 7, 11, 15, 19, 23, 27, 31]
    >>> s
    (2, 4)
    """
    shape_ranges = [tuple(x for x in range(shape[i])) for i in range(len(shape))]
    shape_ranges[axis] = tuple([element])
    list_of_tuples = [[]]
    for shape_range in shape_ranges:
        list_of_tuples = [x + [y] for x in list_of_tuples for y in shape_range]
    for i in range(len(list_of_tuples)):
        list_of_tuples[i] = tuple(list_of_tuples[i])
    items = []
    for i in range(len(list_of_tuples)):
        items.append(get_item(data, shape, list_of_tuples[i]))
    newShape = list(shape)
    newShape.remove(shape[axis])
    return items, tuple(newShape)

# Reduction operations
def asum(data, shape, axis=None):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    data : list
        Input data.
    shape : tuple
        The shape of the input array
    axis : int, optional
        Axis along which the sums are computed. The default is to compute
        the sum of the flattened array.

    Returns
    -------
    sum_along_axis : tuple (reduced_data, newshape)

    Examples
    --------
    >>> data = range(32)
    >>> shape = (2, 4, 4)
    >>> asum(data, shape, 1)
    ([24, 28, 32, 36, 88, 92, 96, 100], (2, 4))
    """
    if axis is None:
        return sum(data), (1,)
    itemsList = []
    for i in range(shape[axis]):
        itemsList.append(extract(data, shape, axis, i)[0])
    itemsZip = zip(*itemsList)
    sumList = []
    for i in range(len(itemsZip)):
        sumList.append(sum(itemsZip[i]))
    newshape = list(shape)
    newshape.remove(shape[axis])
    return sumList, tuple(newshape)

def mean(data, shape, axis=None):
    """
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    data : list
        Input data.
    shape : tuple
        The shape of the input array
    axis : int, optional
        Axis along which the means are computed. The default is to compute
        the mean of the flattened array.

    Returns
    -------
    mean_along_axis : tuple (reduced_data, newshape)

    Examples
    --------
    >>> data = range(32)
    >>> shape = (2, 4, 4)
    >>> mean(data, shape)
    (15.5, ())
    >>> mean(data, shape, 2)
    ([1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5], (2, 4))
    """
    if axis is None:
        Mean = sum(data)/len(data)
        return Mean, ()
    itemsList = []
    for i in range(shape[axis]):
        itemsList.append(extract(data, shape, axis, i)[0])
    itemsZip = zip(*itemsList)
    meanList = []
    for i in range(len(itemsZip)):
        meanList.append(sum(itemsZip[i])/len(itemsZip[i]))
    newshape = list(shape)
    newshape.remove(shape[axis])
    return meanList, tuple(newshape)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
