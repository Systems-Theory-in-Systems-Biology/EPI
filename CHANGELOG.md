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
- Bug that caused thinning to be performed incorrectly.

## [Unreleased]

### Fixed

- Bug in result manager causing burn-in and thinning to be performed on the wrong samples.

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
