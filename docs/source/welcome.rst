.. image:: images/epic.png
   :width: 200pt

--------------------------------------------
EPIC - Euler Parameter Inference Codebase
--------------------------------------------


The Euler Parameter Inference Codebase (EPIC) is a python package for inverse parameter inference.
The EPI algorithm takes observed data and a model as input and returns a parameter distribution, which is consistent with the observed data by solving the inverse problem directly. In the case of a one-to-one mapping, this is the true underlying distribution.
We support SBML ode models and user provided models.

.. Put the badges here?

------------
Installation
------------

The package is available on pypi and can be installed with:

.. code-block::
   
   pip install epic

You can also build the library from the newest source code by following the :doc:`Development Quickstart Guide </MarkdownLinks/development>`.

------------
How to start
------------

| Derive your model from :py:class:`epic.core.model.Model` and implement the abstract functions :py:meth:`~epic.core.model.Model.forward` and :py:meth:`~epic.core.model.Model.jacobian`. Then call :py:meth:`Model.inference` with you data file.

.. warning:: TODO: The function inference may not exist yet!!!


You can also derive your model from

* :py:class:`~epic.core.model.JaxModel`: The jacobian of your forward method is automatically calculated. Use jax.numpy instead of numpy for the forward method implementation!
* :py:class:`~epic.core.model.SBMLModel`: The complete model is derived from the given sbml file. You dont need to define the Model manually.


Optionally you can also inherit, and implement the abstract functions from

* :py:class:`~epic.core.model.ArtificialModelInterface`: This allows you to check if the inversion algorithm is working for your model using the function :py:meth:`~epic.core.model.Model.test`.
   .. warning:: TODO: The function test may not exist yet!!!
* :py:class:`~epic.core.model.VisualizationModelInterface`: This allows you to plot the results of the data inference using the function :py:meth:`~epic.core.model.Model.plot`.
   .. warning:: TODO: The function plot may not exist yet!!!


Please read the documentation for our :doc:`Examples </examples>`.
