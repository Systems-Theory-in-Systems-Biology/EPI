Tutorial
========

This tutorial provides a full walk-through on how to apply Eulerian Parameter Inference (EPI) to an example problem. We only assume that you already installed :code:`eulerpi`.
The tutorial is divided into four sections:

0. :ref:`Introduction`
1. :ref:`Define your data`
2. :ref:`Define your model`
3. :ref:`Inference`

.. .. note::
..     The tutorial is based on the :py:class:`eulerpi.examples.temperature.Temperature` example model and uses the data in :file:`Data/TemperatureData.csv`.
..     Everything needed will be provided in the tutorial.

Let's start!

Introduction
------------
Eulerian Parameter Inference is an algorithm to infer a parameter distribution :math:`Q` satisfying :math:`Y = s(Q)` given a (discrete) data probability distribution :math:`y_i \sim Y`
and a model implementing the mapping :math:`s: Q \to Y`. The (forward) model describes the mapping from the parameter points :math:`q_i` to the data points :math:`y_i`.

In the following we will look at temperature data over the globe and a model for the dependence of the temperature :math:`y_i` on the latitude :math:`q_i`.

|Fig1|

The goal is to derive the parameter distribution :math:`\Phi_Q` from the data distribution :math:`\Phi_Y`. This is the inverse of what our (forward) model is providing.
To solve the inverse problem, EPI uses the multi-dimension transformation formula:

|Fig2|

.. |Fig1| image:: EarthTemperature.png
    :width: 450px
    :alt: The model mapping the latitude :math:`q_i` to the temperature :math:`y_i`


.. |Fig2| image:: EarthTemperature_solved.png
  :width: 450
  :alt: The (forward) transformation rule applied to our model


In the real world, problems with a known continuous data distribution are very sparse. Instead, we often rely on discrete measurements.
EPI starts with discrete data points as input and derives a continuous distribution using Kernel Density Estimation (KDE) techniques.
From this data distribution, the EPI algorithm derives the parameter distribution. To close the cycle between the data and parameters, we can again sample from this distribution and use the forward model to get a discrete distribution of the parameters.

.. It does this using the inverse of the multi-dimensional transformation formula, sampling techniques and the KDE again. 

.. image:: real_data.png
    :width: 800
    :alt: The cycle described before as image.

With this picture in mind, we can start to implement the temperature problem in eulerpi.



Define your data
----------------

Your data needs to be stored in a :file:`.csv` file in the following format:

.. code-block:: text

    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN
    ...
    datapoint_dim1, datapoint_dim2, datapoint_dim3, ..., datapoint_dimN

Each of the lines defines an N-dimensional data point. The :file:`.csv` file will be loaded into an :math:`\mathrm{R}^{M \times N}` numpy matrix in EPI.

In the following, we will use the example data :file:`TemperatureData.csv`. You can download it here: :download:`Download Temperature Data<TemperatureData.csv>`.
It has 455 data points with two dimensions each. Nonuniform data is not supported in EPI.

Define your model
-----------------

Next, you need to define your model. The most basic way is to derive from the :py:class:`eulerpi.core.model.Model` base class.

.. literalinclude:: ../../../eulerpi/examples/temperature/temperature.py
  :language: python
  :pyobject: Temperature

Of course, you also need the imports:

.. code-block:: python

    import importlib
    import jax.numpy as jnp
    import numpy as np
    from eulerpi.core.model import Model

A model inheriting from :py:class:`~eulerpi.core.model.Model` must implement the methods :py:meth:`~eulerpi.core.model.Model.forward` and :py:meth:`~eulerpi.core.model.Model.jacobian`.
In addition, it must provide the methods :py:meth:`~eulerpi.core.model.Model.getcentral_param` and :py:meth:`~eulerpi.core.model.Model.getParamSamplingLimits` to provide the sampling algorithm with sensible starting values and boundary values.
The jacobian is derived analytically here and implemented explicitly.

.. important::
    
    If you use jax_ to calculate the forward evaluation of your model, you can get the jacobian using :code:`jax.jacrev(self.forward)`.
    In all other cases, you have to provide the jacobian using your own implementation.

.. note::

    If you define your forward method using jax and as a class method, you can also inherit from `JaxModel` and the jacobian will be provided for you.
    Additionally, JaxModel will use the jit compiler to speed up the calculation of your jacobian but also of your forward call.

.. warning::

    Using jit on functions with instance variables is dangerous. You can get false results if the variables change internally and jax does not *see* these changes.

.. note::

    For more possibilities on how to define your model, e.g. using external c++ code, see the :ref:`example section <Example Models>`

.. _jax: https://jax.readthedocs.io/en/latest/index.html




Inference
---------

Now we can now use EPI to infer the parameter distribution from the data.

.. code-block:: python

    from eulerpi.core.inference import inference

    # This line is needed for multiprocessing in python
    if __name__ == "__main__":
        model = Temperature()
        inference(model, data = "TemperatureData.csv")

Depending on the complexity of your model the sampling can take a long time.
Due to this reason, not only the final results but also intermediate sampling results are saved.
You can find them in the folder :file:`Applications/Temperature/`. The final results are stored in the file :file:`Applications/Temperature/<run_nam>/<slice_name>/OverallSimResults.csv`.
The ``slice_name`` results from the optional parameter :py:attr:`slice` of the :py:func:`~eulerpi.core.inference.inference` function.

.. .. code-block:: python

..     model.plot(plot_type="spyder")
..     model.plot(plot_type="standard")

.. As the last step, you can plot the results using the built-in plotting functionality.
.. You can choose between a spider-web representation and a standard plot.
.. The standard plot is only working for one or two-dimensional parameters and data.

.. only:: builder_html or readthedocs

    .. note::

        This tutorial is also available as a jupyter notebook: :download:`Download Temperature Tutorial<tutorial.ipynb>`.

.. Preparation
.. -----------

.. Create a folder for your project and package, if you haven't done it yet.

.. .. code-block:: bash

..     mkdir my_project && cd my_project

.. .. code-block:: bash

..     pip install eulerpi
