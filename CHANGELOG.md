# Changelog

All notable changes to this project will be documented in this file.

## [Template]

### Added

- Feature X
- Feature Y

### Changed

- Refactored module A to improve performance

### Fixed

- Bug in module B that caused a crash

## [Unreleased] - unreleased

### Added

### Changed

### Fixed
  
## [0.11.0] - 2025-07-20

### Changed

- Switched from poetry to uv as project management tool
- Switched to python 3.13 as default
- Updating minimal python version to 3.11 to fix issues with cpp dependencies
- Increased upper policy version of cmake in cpp example

## [0.10.0] - 2024-11-22

### Added

- Added the `AffineTransformation` class to perform affine transformations on the data using jax

- Added the method function `forward_vectorized` to the (Base)Model class and a specialized jax version in the `JaxModel` class.

### Changed

- Refactored the module `eulerpi.core.data_transformation` into the subpackage `eulerpi.core.data_transformations`. The data transformation classes can be import from this subpackage directly
- Renamed `eulerpi.core.data_transformation.DataNormalizer` into `eulerpi.core.data_transformations.DataNormalization`
- Removed the class/factory functions `from_data` and `from_transformation` from the `DataTransformation` subclasses. Use the constructors instead.
- The inference function no longer has the parameter `custom_data_transformation`. You can pass an instance of the provided `DataTransformation` subclasses or pass a custom subclass.
- The inference function no longer has the parameter `n_components_pca`. Instantiate a `PCATransformation` with the data and the number of components to keep. Pass it as as argument for the parameter `data_transformation` in the inference function.

- Renamed the abstract base `Model` class to `BaseModel`. The usage of `Model` is still supported, but deprecated, and possibly removed in a future release.
- Splitted the file model.py into multiple modules, one for each class, located in the subpackage `eulerpi.core.models`. The imports change from `eulerpi.core.model` to `eulerpi.core.models`.
- Removed the function `is_artificial` from the (Base)Model class. Use `isinstance(custom_model, ArtificialModelInterface)` or `issubclass(CustomModel, ArtificialModelInterface)` instead.

## [0.9.1] - 2024-09-08

### Changed

- Raised required amici version to 0.26.1
- Ignoring sampler warnings caused by parameters outside the parameter limits
- Limited the parameter range and increased the solver accuracy of the Corona model to avoid warnings

### Fixed

- Seed numpy random number generator for reproducibility of pytest runs

## [0.9.0] - 2024-05-31

### Changed

- Raised the required python version to 3.10
- Removed upper limit on the version number of dependencies
- Updated amici to newest version to fix compilation with Intel MKL

## [0.8.1] - 2024-05-31

### Added

- Added `PyQt6` as dev dependency to allow developers to view plots in the development environment

### Changed

- Removed the redundant function `evaluate_sample` from `sampling.py`

### Fixed

- Switched from `importlib.resources.path` to `importlib.resources.files` due to deprecation
- Removed argument name in call to `files(...)` that is changing from `package` to `anchor`
- Removed unused kwargs passed to the plotting function `contour` that generated a warning in `test_heat_model()`
- Bug that changed the model attribute `central_param` during `inference`

## [0.8.0] - 2024-03-21

### Added

- Basic plotting function for sample results
- Users can now check the models they want to use with a basic functionality check and a quick inference check on artificial data.
- Added code usage examples to documentation.

### Fixed

- Fixed deprecated jax.config import

## [0.7.0] - 2023-11-03

Breaking changes: The stock model is removed from the available example models! You can copy the model code and the data files from a previous version and import it as custom model, if you need it.

### Changed

- Removed the stock model, because it does not fulfill the requirements for the EPI method
- Minor dependency updates

## [0.6.1] - 2023-11-02

### Added

- Added link to changelog on pypi

## [0.6.0] - 2023-11-02

Breaking changes: SBML support is now only available when install eulerpi with the "sbml" extra!

### Added

### Changed

- Updated the dependencies to the latest possible versions
- Made the sbml model support optional using the extra `sbml`

### Fixed

- Fixed poetry.lock containing faulty version 1.6.8 of debugpy
- Fixed github actions failing due to missing `sudo apt update`

## [0.5.0] - 2023-07-20

### Added

- Added support for for sbml models to select parameters and species used in the inference
- Added support for evaluating sbml models at multiple time points

### Changed

- Switched from using parameter names to using parameter ids in the sbml model
- Change spatial discretization of heat model example to proper second order central differences
- Change the initial walker position generation of the emcee sampler

## [0.4.0] - 2023-06-24

### Added

- Added a new example model, a 2d heat conduction equation
- Added a function to model to specify more complex parameter domains

### Changed

- Removed old functions from corona model
- Temporarily removed stock and sbml models from test examples

### Fixed

- Updated tutorial to be consistent with current state of the project
- Fixed bug in dense grid generation causing parameter limits to be applied inconsistently when using slices

## [0.3.1] - 2023-05-02

### Fixed

- Bug in result manager causing burn-in and thinning to be performed on the wrong samples.
- Bug in result manager that caused density evals to be saved as data samples for non-full slices.

## [0.3.0] - 2023-04-27

### Added

- Added data normalization to improve numerics and performance in the inference method.
- Added possiblity to use no PCA transformation or completely custom transformation for inference.
- More tests to validate EPI

### Fixed

- Fixed grid results beeing unusable due to numerical issues
- Fixed sbml models ignoring parameter_names argument

## [0.2.1] - 2023-04-21

### Fixed

- Fix sparse grid returning meaningless data
- Fix menten data shape
- Fix SBMLModel data_dim: Return states count instead of observables count

### Added

- Validating the shapes of the inference results

## [0.2.0] - 2023-04-20

### Changed

- Now using default emcee move policy during MCMC sampling to improve convergence speed
- Changed argument calc_walker_acceptance_bool to get_walker_acceptance
- Changed meaning of argument num_burn_in_samples to refer to the number of samples burned per chain, not in total

### Fixed

- Fixed SwigPyObject pickling
- Fixed dead lock for larger data sets and models

### Added

- ResultManager is saving meta data for inference
- ResultManager uses saved meta data to load inference results

## [0.1.5] - 2023-04-06

### Changed

- Renamed epipy to eulerpi

## [0.1.4] - 2023-04-06

### Changed

- Updated installation instructions

## [0.1.3] - 2023-04-05

### Changed

- Updated dependencies

## [0.1.2] - 2023-04-05

### Fixed

- removed `__pycache__`, `build` and `.so` from pypi

## [0.1.1] - 2023-04-05

### Fixed

- fixed epi logo not showing on pypi

## [0.1.0] - 2023-04-05

### Added

- Initial release
