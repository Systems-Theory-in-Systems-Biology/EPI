# TODOs

This implementation should in the end be usable with the Systems Biology Markup Language (SBML) format.
The reason for this goal is to enable reuse code, facilitate cooperation and boost visability.

## Implementation

- [ ] create module for straight forward EPI usage, current classmethod solution is not optimal
- [ ] create single function ```Model.interference()``` that allows the user to evaluate his model on custom data
- [ ] create single function ```Model.plot()``` that allows the user to visualize his results
- [ ] create single function ```Model.test()``` that allows the user to test the inversion for his model on artificial data
- [ ] create and implement example usecases
- [ ] make the toolbox usable for SBML models
- [ ] Test Jax for gpu: ```pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```

Type Hints:

- [ ] np.double vs np.ndarray if more dimensions are passed?

## Documentation

- [ ] Fix [Development Quickstart Guide](./DEVELOPMENT.md#quickstart) link in sphinx
- [ ] Check with supervisors:
  - [x] License file
  - [x] Contributing file
  - [ ] Citation file
  - [x] README file
  - [ ] pyproject toml project urls
- [x] Beautify documentation
  - [x] read <https://www.reddit.com/r/Python/comments/5gqxyk/learning_resources_for_sphinx/>
  - [x] choose a theme? (Take a look at the template from seaborn or numpy) current theme looks good
  - [x] switch to mkdocs, pdocs, ...? no
  - [ ] Add doc dependencies. E. g. myst_parser, sphinx-copybutton
- [ ] [Dependabot badge](https://github.com/dependabot/dependabot-core/issues/1912)
- [ ] Complete all models! and test them
- [ ] Remove all toods
- [ ] Run through grammarly

## Deployment

- [ ] Test test deployment
- [ ] Test real deployment to pypy

## SBML

- Complete sbml class using one of
  - [ ] [SymbolicSBML](https://gitlab.com/wurssb/Modelling/symbolicsbml)
  - [ ] [RoadRunner](https://sys-bio.github.io/roadrunner/docs-build/index.html) Could be a good option
  - [x] [Sbmltodepy](https://github.com/AnabelSMRuggiero/sbmltoodepy) they dont want users to make issues/ pull requests / .. :(

## Dependencies

- Add c++ / example dependencies in general and explanation on how to run c++ model?
