# TODOs

This implementation should in the end be usable with the Systems Biology Markup Language (SBML) format.
The reason for this goal is to enable reuse code, facilitate cooperation and boost visability.

## Implementation

- [ ] create module for straight forward EPI usage, current classmethod solution is not optimal
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

## Profiling

- scalene

## Before pypi:

- [ ] Check pyproject toml project urls

## Postponed

- [ ] Fix [Development Quickstart Guide](./DEVELOPMENT.md#quickstart) link in sphinx
- [ ] create single function ```Model.plot()``` that allows the user to visualize his results
- [ ] create single function ```Model.test()``` that allows the user to test the inversion for his model on artificial data
