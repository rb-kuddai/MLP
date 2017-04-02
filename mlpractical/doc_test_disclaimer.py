import doctest

def dummy_backprop(igrad):
    """

    :param igrad: input gradients
    :return: output gradients usage doctest

    Usage/Doctest:
    >>> dummy_backprop([5])
    Crucial Error can do it!
    >>> dummy_backprop([2, 2])
    Crucial Error can do it!
    >>> dummy_backprop([])
    Nothing to work with
    >>> dummy_backprop(45)
    Nothing to work with
    """
    if not isinstance(igrad, list) or len(igrad) == 0:
        print "Nothing to work with"
        return
    if len(igrad) > 0:
        print "Crucial Error can do it!"
        return

if __name__ == "__main__":
    doctest.testmod(verbose=True)
