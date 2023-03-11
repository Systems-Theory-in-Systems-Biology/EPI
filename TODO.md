# TODOs

## Redesign

- Change from data "file" flow to data flow?

## Pickling and jax

- [ ] Document classmethod solution for pickling function, how to use fixed params
- [ ] But also investigate why jitting the temperature model generates super slow code while super fast for e. g. corona. jitted should be not slower than standard.
- [ ] Use them in tutorial / provide tutorial.py
- [ ] Build example for: How to deal with emcee requiring pickable but jax is not?! It seems like the class method is working and i can even add the fixed params from the model. I should just rework the underscore thingy! https://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-multiprocessing-pool-map/41959862#41959862

## Documentation

- [ ] [Dependabot badge](https://github.com/dependabot/dependabot-core/issues/1912)
- [ ] How to run jupyternotebook with poetry in vs code and in terminal

## Deployment

- [ ] Deployment to pypi

## Postponed

- [ ] Fix [Development Quickstart Guide](./DEVELOPMENT.md#quickstart) link in sphinx
- [ ] create single function ```Model.plot()``` that allows the user to visualize his results
- [ ] create single function ```Model.test()``` that allows the user to test the inversion for his model on artificial data
- [ ] Use save, load from numpy and not savetxt, loadtxt
- [ ] Or maybe better: use hdf5 backend for intermediate results, and inference takes data array as input and returns the three arrays which are currently saved as files
- [ ] Allow to use the library with own data flow without writing and loading all the files. Return everything we save to files at the moment. So give more control to the user and rely less on the file system and default paths.

## Outlook

- [ ] Testing on other systems than linux on github: Replace apt install and then also test on windows and mac machine using github test matrix.
- [ ] More / Systematic profiling with scalene
