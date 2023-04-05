SBML Models
----------
The :py:class:`~epipy.core.model.SBMLModel` loads a biological model from an external file written in the sbml standard format.
It generates the forward and jacobian method automatically and derives the parameter and data dimension from the sbml model.

The two example models are included in :file:`epipy/examples/sbml/`.


Specialities
____________

* Super simple setup
* No need to write any model code

Here's a code snippet to load your own sbml model and to do the parameter inference:

.. code-block:: python

    from epipy.core.model import SBMLModel
    from epipy.core.inference import inference

    model = SBMLModel('model.xml', central_param=[1.0, 1.0], param_limits=[[0.0, 2.0], [0.0, 2.0]], param_names=['k1', 'k2'])
    model.inference(model, 'data.csv')

The attribute :py:attr:`~epipy.core.model.SBMLModel.param_names` contains the names of the parameters in the sbml model, for which the inference should be performed.
Per default it contains all parameters from the sbml model file.

.. note::
    For the SBML Standard see https://sbml.org/.
    You can visualize sbml files with https://sbml4humans.de/.

.. .. literalinclude:: ../../../epipy/examples/sbml/sbml_model.py
..   :language: python
..   :pyobject: MySBMLModel
