SBML Models
----------
The :py:class:`~eulerpi.core.model.SBMLModel` loads a biological model from an external file written in the sbml standard format.
It generates the forward and jacobian method automatically and derives the parameter and data dimension from the sbml model.

The two example models are included in :file:`eulerpi/examples/sbml/`.


Specialities
____________

* Super simple setup
* No need to write any model code

Here's a code snippet to load your own sbml model and to do the parameter inference:

.. code-block:: python

    import numpy as np
    from eulerpi.core.model import SBMLModel
    from eulerpi.core.inference import inference

    central_param = np.array([1.0, 1.0])  # initial guess, evaluation must have nonzero density
    param_limits = np.array([[0.0, 2.0], [0.0, 2.0]])
    param_names = ['k1', 'k2']

    model = SBMLModel('model.xml',
                      central_param=central_param,
                      param_limits=param_limits,
                      param_names=param_names)
    inference(model, 'data.csv')

The attribute :py:attr:`~eulerpi.core.model.SBMLModel.param_names` contains the names of the parameters in the sbml model, for which the inference should be performed.
Per default it contains all parameters from the sbml model file.

.. note::
    For the SBML Standard see https://sbml.org/.
    You can visualize sbml files with https://sbml4humans.de/.

.. .. literalinclude:: ../../../eulerpi/examples/sbml/sbml_model.py
..   :language: python
..   :pyobject: MySBMLModel
