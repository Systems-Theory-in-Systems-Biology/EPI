.. image:: images/epi.png
   :width: 200pt

-------------------------------
EPI - Euler Parameter Inference
-------------------------------


The Euler Parameter Inference Codebase (EPI) is a python package for inverse parameter inference.
The EPI algorithm takes observed data and a model as input and returns a parameter distribution, which is consistent with the observed data by solving the inverse problem directly. In the case of a one-to-one mapping, this is the true underlying distribution.
We support SBML ode models and user provided models.

.. Put the badges here?

------------
Installation
------------

The package is available on pypi and can be installed with:

.. code-block:: bash
   
   pip install epi

You can also build the library from the newest source code by following the :doc:`Development Quickstart Guide </MarkdownLinks/development>`.

------------
How to start
------------

| Derive your model from :py:class:`epi.core.model.Model` and implement the abstract functions :py:meth:`~epi.core.model.Model.forward` and :py:meth:`~epi.core.model.Model.jacobian`.

.. code-block:: python
   
   # my_model.py

   import jax.numpy as jnp
   from epi.core.model import Model

   class MyModel(Model):
      def forward(self, param):
         return jnp.array(...)

      def getParamSamplingLimits(self):
         return np.array([[-1.,1.], [-101.1, 13.4],...]) # [[UpperBound_dim1,LowerBound_dim1],...]

      def getCentralParam(self):
         return np.array([0.5, -30.0,...])

      def jacobian(self, param):
         return jnp.array(...)

To evaluate the model and infer the parameter distribution, call:

.. code-block:: python

   from epi.core.sampling import inference
   from my_model import MyModel

   model = MyModel()
   inference(model=model, dataPath="my_data.csv")

The file :file:`my_data.csv` has to contain the data in csv format with :code:`seperator=,` in the format

.. code-block:: text
   
   # my_data.csv

   datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
   datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
   datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
   ...
   datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN

which corresponds to a matrix with the shape `nSamples x dataDim`.

.. note::
   
   Please read the documentation for our :doc:`Examples </examples>`.

.. TODO: move this ?

.. You can also derive your model from

.. * :py:class:`~epi.core.model.JaxModel`: The jacobian of your forward method is automatically calculated. Use jax.numpy instead of numpy for the forward method implementation!
.. * :py:class:`~epi.core.model.SBMLModel`: The complete model is derived from the given sbml file. You don't need to define the Model manually.

.. Optionally you can also inherit, and implement the abstract functions from

.. * :py:class:`~epi.core.model.ArtificialModelInterface`: This allows you to check if the inversion algorithm is working for your model using the function :py:meth:`~epi.core.model.Model.test`.
   
.. * :py:class:`~epi.core.model.VisualizationModelInterface`: This allows you to plot the results of the data inference using the function :py:meth:`~epi.core.model.Model.plot`.

.. .. warning:: TODO: The functions plot and test may not exist yet!!!
