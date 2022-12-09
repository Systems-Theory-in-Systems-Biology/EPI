# EPIC

[![pages-build-deployment](https://github.com/Systems-Theory-in-Systems-Biology/EPIC/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/Systems-Theory-in-Systems-Biology/EPIC/actions/workflows/pages/pages-build-deployment)

## About

EPIC = Euler Parameter Inference Codebase

The EPI returns a parameter distribution, which is consistent with the observed data by solving the inverse problem directly. In the case of a one-to-one mapping, this is the true underlying distribution.

## Documentation

The documentation can be found under [Go to documentation](https://Systems-Theory-in-Systems-Biology.github.io/EPIC/)

### For developers

Here are some infos on how to work with sphinx, github, ...

#### Poetry

We use poetry as build system and for the dependency management. Most commans can be simply run in the virtual environment by prepending ```poetry run``` before the command.

You can also use ```poetry shell``` to activate the virtual environment and exit it with ```exit```.

Run ```poetry add xxx``` to add ```xxx``` as dependencie to your project or ```poetry add --group dev myPackage``` to add ```myPackage``` to your ```dev```dependencies. You can have arbitrary group names.

For more information read [Poetry]<https://python-poetry.org/docs/basic-usage/#initialising-a-pre-existing-project>.

#### Code quality checks

We use black and flake8 to maintain a common style and check the code. Please check your code:

``` bash
pre-commit install
```

for changed files without trying to commit

``` bash
bash .git/hooks/pre-commit
```

or for checking all files:

``` bash
pre-commit run --all-files
```

#### GitHub Pages

To publish the documentation you have to set

``` text
Settings -> Code and automation -> Pages -> Build and Deployment:
- Source: Deploy from a branch
- Branch: gh-pages && /(root)
```

#### Build the docs manually

``` bash
mkdir docs
cd docs
sphinx-apidoc -f -o source/ ../
make html
```

#### Packaging the modules

``` bash
python3 -m pip install --upgrade build
python3 -m build
```

#### Uploading the package

``` bash
python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*
```

Remove `--repository` including the argument `testpypi` if using the real Python Packade Index PyPi.

#### Testing the upload

``` bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps epic
python3
```

Remove `--index-url` including the argument and `--no-deps` if using the real Python Packade Index PyPi.

<!-- 
``` python
from epic import ???
print(???)
``` -->
