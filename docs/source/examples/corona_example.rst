Corona ODE Model
----------------
The corona ode model is contained in :file:`epi/examples/corona/corona.py`. 

.. TODO::

    The model implements the function :math:`y_i(q_i)=???`.

Specialities
____________

* ODE solver: To solve the ODE problem the jax based ode solver library :code:`diffrax` is used: https://github.com/patrick-kidger/diffrax.
* Automatic Differentiation: The derivatives are calculated automatically with jax by deriving from the class :py:class:`~epi.core.model.JaxModel`,
  which automatically calculates the :py:meth:`epi.core.model.jacobian`.
* JIT compilation: Inheriting from :py:class:`~epi.core.model.JaxModel` also enables jit compilation / optimization for the forward and jacobian method.
  This usually results in a significant execution speedup. It also allows to run your model on the gpu.
* vectorization of model calls using :code:`jax.vmap`:

  .. code-block:: python

    artificialData = vmap(self.forward, in_axes=0)(true_param_sample)

.. literalinclude:: ../../../epi/examples/corona/corona.py
  :language: python
  :pyobject: Corona
