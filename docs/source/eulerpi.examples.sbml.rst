SBML Models
===========

The :py:class:`~eulerpi.model.SBMLModel` loads a biological model from an external file written in the sbml standard format.
It generates the forward and jacobian method automatically and derives the parameter and data dimension from the sbml model.
This allows for extremely fast prototyping of sbml models and no model specific code has to be written.

Here's a code snippet to load your own sbml model and to do the parameter inference:

.. code-block:: python

    import numpy as np
    from eulerpi.model import SBMLModel
    from eulerpi import inference

    # This line is needed for multiprocessing in python
    if __name__ == "__main__":
        central_param = np.array([1.0, 1.0])  # initial guess, evaluation must have nonzero density
        param_limits = np.array([[0.0, 2.0], [0.0, 2.0]])
        param_ids = ['k1', 'k2']
        timepoints = np.array([1.0])

        model = SBMLModel(sbml_file='model.xml',
                        central_param=central_param,
                        param_limits=param_limits,
                        timepoints=timepoints,
                        param_ids=param_ids)
        inference(model, 'data.csv')

The attribute :py:attr:`~eulerpi.model.SBMLModel.param_names` contains the names of the parameters in the sbml model, for which the inference should be performed.
Per default it contains all parameters from the sbml model file.

.. note::
    For the SBML Standard see https://sbml.org/.
    You can visualize sbml files with https://sbml4humans.de/.


We provide two examples for sbml models:

.. toctree::

   Caffeine Model<eulerpi.examples.sbml.sbml_caffeine_model>
   Menten Model<eulerpi.examples.sbml.sbml_menten_model>
