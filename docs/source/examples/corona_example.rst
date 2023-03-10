Corona ODE Model
----------------
A corona ode model is contained in :file:`epi/examples/corona/corona.py`. 

Specialities
____________

* ODE solver: To solve the ODE problem the jax based ode solver library :code:`diffrax` is used: https://github.com/patrick-kidger/diffrax.
* Automatic Differentiation: The derivatives are calculated automatically with jax by deriving from the class :py:class:`~epi.core.model.JaxModel`,
  which automatically calculates the :py:meth:`epi.core.model.jacobian`.
* JIT compilation: Inheriting from :py:class:`~epi.core.model.JaxModel` also enables jit compilation / optimization for the forward and jacobian method.
  This usually results in a significant execution speedup. It also allows to run your model on the gpu.

.. literalinclude:: ../../../epi/examples/corona/corona.py
  :language: python
  :pyobject: Corona
