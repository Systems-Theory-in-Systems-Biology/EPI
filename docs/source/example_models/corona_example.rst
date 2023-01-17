Corona ODE Model
----------------
The corona ode model is contained in :code:`epic/example_models/applications/corona`. 

.. TODO::

    The model implements the function :math:`y_i(q_i)=???`.

Specialities
____________

* ODE solver: To solve the ODE problem the jax based ode solver library :code:`diffrax` is used: https://github.com/patrick-kidger/diffrax.
* Automatic Differentation: The derivatives are calculated automatically with jax by deriving from the class :py:class:`~epic.core.model.JaxModel`,
  which automatically calculates the :py:meth:`epic.core.model.jacobian`.
* JIT compilation: Inhereting from :py:class:`~epic.core.model.JaxModel` also enables jit compilation / optimization for the forward and jacobian method.
  This usually results in a significant execution speedup. It also allows to run your model on the gpu.
* vectorization of model calls using :code:`jax.vmap`:

  .. code-block:: python

    artificialData = vmap(self.forward, in_axes=0)(trueParamSample)

.. literalinclude:: ../../epic/example_models/applications/corona.py
  :language: python
  :pyobject: Corona
