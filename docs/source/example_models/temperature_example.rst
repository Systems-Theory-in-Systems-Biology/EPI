Temperature Model
-----------------
The temperature model is contained in :code:`epic/example_models/applications/temperature`.
The model :math:`y_i(q_i)=60 \cos(q_i)-30=s(q_i)` describes the temperature for a place on the earth :math:`y_i` by using the latitude coordinates :math:`q_i`.
The jacobian :math:`{\frac{dy}{dq}]_i(q_i)=-30 \sin(q_i)` can be calculated analytically.

Specialities
____________

* Additional fixed parameters: The model includes fixed parameters :code:`self.lowT=30.0` and :code:`self.highT=30.0`.
  These fixed parameters are passed to the forward function separately. You can create models with different parameters by
  creating several model objects. However you should think about overwriting the method :py:meth:`epic.core.model.Model.getModelName()`
  to include the fixed parameters of the model object. Else your results for different fixed parameter sets will be mixed.

.. TODO::

    Fix _forward, call, ... and maybe adapt this documentation part then

.. literalinclude:: ../../epic/example_models/applications/temperature.py
  :language: python
  :pyobject: Temperature
