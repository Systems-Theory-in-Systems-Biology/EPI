# TODOs

This implementation should in the end be usable with the Systems Biology Markup Language (SBML) format.
The goal is to enable reuse, facilitated cooperation and boost visability.

For the implementation it is important to follow the PEP8 guidelines and write a proper documentation for everything.

## TODO

Implementation:

- [ ] create module for straight forward EPI usage
- [ ] create and implement example usecases
- [ ] make the toolbox usable for SBML models
- [ ] Test Jax for gpu: ```pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```

Type Hints:
- [ ] np.double vs np.ndarray if more dimensions are passed?

Documentation:

- [ ] Fix [Development Quickstart Guide](./DEVELOPMENT.md#quickstart) link in sphinx
- [ ] Check with supervisors:
  - [ ] License file
  - [ ] Contributing file
  - [ ] Citation file
  - [ ] README file
  - [ ] pyproject toml project urls
- [ ] Beautify documentation
  - [ ] read <https://www.reddit.com/r/Python/comments/5gqxyk/learning_resources_for_sphinx/>
  - [ ] choose a theme? (Take a look at the template from seaborn or numpy)
  - [ ] switch to mkdocs, pdocs, ...?

- [Dependabot badge]<https://github.com/dependabot/dependabot-core/issues/1912>
