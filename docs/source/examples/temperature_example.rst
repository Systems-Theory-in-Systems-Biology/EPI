Temperature Model
-----------------
The temperature model is contained in :file:`epi/examples/temperature/temperature.py`.
The model :math:`y(q)=60 \cos(q)-30=s(q)` describes the temperature for a place on the earth :math:`y` by using the latitude coordinates :math:`q`.
The jacobian :math:`{\frac{dy}{dq}}(q_i)=-30 \sin(q_i)` can be calculated analytically.

Specialities
____________

* Additional fixed parameters: The model includes fixed parameters :code:`self.low_T=30.0` and :code:`self.high_T=30.0`.
  These fixed parameters are passed to the forward function separately. You can create models with different parameters by
  creating several model objects.
  
  To seperate the output for models with the same class but different parameters, you can overwrite the attribute :py:attr:`epi.core.model.Model.name`
  and include the fixed parameters of the model object. Alternatively you can use a :py:class:`~epi.core.result_manager.ResultManager` object with custom :py:attr:`~epi.core.result_manager.ResultManager.run_name` attribute.

.. literalinclude:: ../../../epi/examples/temperature/temperature.py
  :language: python
  :pyobject: Temperature
