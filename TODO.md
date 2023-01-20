# TODOs

This implementation should in the end be usable with the Systems Biology Markup Language (SBML) format.
The reason for this goal is to enable reuse code, facilitate cooperation and boost visability.

## Implementation

- [ ] Use importlib in test_applications to avoid failing due to import of other module.
- [ ] Document classmethod solution for pickling function, how to use fixed params
- [ ] But also investigate why jitting the temperature model generates super slow code while super fast for e. g. corona. jitted should be not slower than standard.
- [ ] Use them in tutorial / provide tutorial.py
- [ ] make the toolbox usable for SBML models
- [ ] Build example for: How to deal with emcee requiring pickable but jax is not?! It seems like the class method is working and i can even add the fixed params from the model. I should just rework the underscore thingy! https://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-multiprocessing-pool-map/41959862#41959862

## Documentation

- [ ] [Dependabot badge](https://github.com/dependabot/dependabot-core/issues/1912)
- [ ] Deactivate todos in conf.py

## Deployment

- [ ] Ship example data and test somehow seperately, but preferabely examples still in the same or at least some package namespace?
- [ ] Test real deployment to pypi
- [ ] Direct deployment to pypi?

## SBML

- Complete sbml class using one of
  - [ ] [SymbolicSBML](https://gitlab.com/wurssb/Modelling/symbolicsbml)
  - [ ] [RoadRunner](https://sys-bio.github.io/roadrunner/docs-build/index.html) Could be a good option
  - [ ] [Sbmltodepy](https://github.com/AnabelSMRuggiero/sbmltoodepy) they dont want users to make issues/ pull requests / .. :(

## Before pypi

- [ ] Check if all tests are passing, reduce sample number to spend less runtime on github and get faster feedback
- [ ] Check pyproject toml project urls

## Large performance optimization

- Remove the loop(s) in the following snippet

  ```python
  for i in range(numSteps):
    for j in range(numWalkers):
        allRes[i * numWalkers + j, :] = sampler.blobs[i][j]
  ```

## Postponed

- [ ] Fix [Development Quickstart Guide](./DEVELOPMENT.md#quickstart) link in sphinx
- [ ] create single function ```Model.plot()``` that allows the user to visualize his results
- [ ] create single function ```Model.test()``` that allows the user to test the inversion for his model on artificial data

## Outlook

- [ ] Testing on github: Replace apt install and then also test on windows and mac machine using github test matrix.
- [ ] Allow to use the library with own data flow without writing and loading all the files. Return everything we save to files at the moment. So give more control to the user and rely less on the file system and default paths.
- [ ] Profiling with scalene
- [ ] Check if switching between jax.numpy (jnp and numpy (np) in the functions affects performance
- [ ] Remove duplicate code from the tests (test_applications, test_cpp_model,...)
- [ ] Is providing inference and the model types in epi.core a good idea?
