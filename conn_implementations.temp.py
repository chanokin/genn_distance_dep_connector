class MaxDistanceFixedProbabilityConnector(DistanceDependentProbabilityConnector):
    __doc__ = DistanceDependentProbabilityConnector.__doc__

    def __init__(self, max_dist, probability, allow_self_connections=True,
                 rng=None, safe=True, callback=None):
        d_expr = "%s * ( d <= %s)"%(probability, max_dist)
        DistanceDependentProbabilityConnector.__init__(
            d_expr, allow_self_connections, rng, safe, callback)
        self.probability = probability
        self.max_dist = max_dist
        self._builtin_name = 'MaxDistanceFixedProbability'
        @property
        def _conn_init_params(self):
            return {'prob': probability,
                    'max_d': max_dist,
                    'rowLength': self.n}


class FixedProbabilityConnector(AbstractGeNNConnector, FixProbPyNN):
    _row_code = """
    """

    __doc__ = FixProbPyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        FixProbPyNN.__init__(self, safe=safe, callback=callback)
        # self._row_length = 1
        # self._col_length = 1
        # self._sparse = True


class FixedTotalNumberConnector(AbstractGeNNConnector, FixTotalPyNN):
    _row_code = """
    """

    __doc__ = FixTotalPyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        FixTotalPyNN.__init__(self, safe=safe, callback=callback)


class FixedNumberPreConnector(AbstractGeNNConnector, FixNumPrePyNN):
    _row_code = """
    """

    __doc__ = FixNumPrePyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        FixNumPrePyNN.__init__(self, safe=safe, callback=callback)


class FixedNumberPostConnector(AbstractGeNNConnector, FixNumPostPyNN):
    _row_code = """
    """

    __doc__ = FixNumPostPyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        FixNumPostPyNN.__init__(self, safe=safe, callback=callback)


class DistanceDependentProbabilityConnector(AbstractGeNNConnector, DistProbPyNN):
    _row_code = """
    """

    __doc__ = DistProbPyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        DistProbPyNN.__init__(self, safe=safe, callback=callback)


class DisplacementDependentProbabilityConnector(AbstractGeNNConnector,
                                                DisplaceProbPyNN):
    _row_code = """
    """

    __doc__ = DisplaceProbPyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        DisplaceProbPyNN.__init__(self, safe=safe, callback=callback)


class IndexBasedProbabilityConnector(AbstractGeNNConnector, IndexProbPyNN):
    _row_code = """
    """

    __doc__ = IndexProbPyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        IndexProbPyNN.__init__(self, safe=safe, callback=callback)


class SmallWorldConnector(AbstractGeNNConnector, SmallWorldPyNN):
    _row_code = """
    """

    __doc__ = SmallWorldPyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        SmallWorldPyNN.__init__(self, safe=safe, callback=callback)


class FromListConnector(AbstractGeNNConnector, FromListPyNN):
    _row_code = """
    """

    __doc__ = FromListPyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        FromListPyNN.__init__(self, safe=safe, callback=callback)


class FromFileConnector(AbstractGeNNConnector, FromFilePyNN):
    _row_code = """
    """

    __doc__ = FromFilePyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        FromFilePyNN.__init__(self, safe=safe, callback=callback)


class CloneConnector(AbstractGeNNConnector, ClonePyNN):
    _row_code = """
    """

    __doc__ = ClonePyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        ClonePyNN.__init__(self, safe=safe, callback=callback)


class ArrayConnector(AbstractGeNNConnector, ArrayPyNN):
    _row_code = """
    """

    __doc__ = ArrayPyNN.__doc__

    def __init__(self, safe=True, callback=None,
                 on_device_init=False, procedural=False):
        AbstractGeNNConnector.__init__(self, on_device_init, procedural)
        ArrayPyNN.__init__(self, safe=safe, callback=callback)


