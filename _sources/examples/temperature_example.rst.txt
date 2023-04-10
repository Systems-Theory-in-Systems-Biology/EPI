Temperature Model
-----------------
The temperature model is contained in :file:`eulerpi/examples/temperature/temperature.py`.
The model :math:`y(q)=60 \cos(q)-30=s(q)` describes the temperature for a place on the earth :math:`y` by using the latitude coordinates :math:`q`.
The jacobian :math:`{\frac{dy}{dq}}(q_i)=-60 \sin(q_i)` can be calculated analytically.

.. literalinclude:: ../../../eulerpi/examples/temperature/temperature.py
  :language: python
  :pyobject: Temperature

Specialities
____________

* Additional fixed parameters: The model includes fixed parameters :code:`self.low_T=30.0` and :code:`self.high_T=30.0`.
  These fixed parameters are passed to the forward function separately. You can create models with different parameters by
  creating several model objects.

  The best way to seperate the outputs for the parametrized models is to pass a string based on the fixed_params to the attribute :py:attr:`run_name` of the :py:func:`~eulerpi.core.inference.inference` function.

.. literalinclude:: ../../../eulerpi/examples/temperature/temperature.py
  :language: python
  :pyobject: TemperatureWithFixedParams

.. literalinclude:: ../../../tests/test_fixed_params.py
  :language: python
  :pyobject: test_fixed_params
  :lines: 9-

.. note::

  The functions :py:meth:`~eulerpi.examples.temperature.temperature.TemperatureWithFixedParams.calc_forward` is not strictly necessary.
  However it can help to make it work with jax.
