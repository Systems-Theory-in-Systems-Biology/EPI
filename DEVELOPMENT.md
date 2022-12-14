# Development

## Quickstart

- Clone the repository: \
```git clone https://github.com/Systems-Theory-in-Systems-Biology/EPIC.git``` or \ ```git clone git@github.com:Systems-Theory-in-Systems-Biology/EPIC.git```.

- Install the library: \
```cd EPIC && pip install poetry``` \
```poetry install```

- Run the tests: \
```poetry run pytest```

## Maintaining the repository

Here are the most important infos on how to maintain this repository.

- **Dependency Management with Poetry**: \
  We use poetry as build system and for the dependency management. Most commans can be simply run in the virtual environment by prepending ```poetry run``` before the command. You can also use ```poetry shell``` to activate the virtual environment and exit it with ```exit```. Run ```poetry add xxx``` to add ```xxx``` as dependencie to your project or ```poetry add --group dev myPackage``` to add ```myPackage``` to your ```dev``` dependencies. You can have arbitrary group names.
  
  For more information read the [Poetry Documentation](https://python-poetry.org/docs/basic-usage/#initialising-a-pre-existing-project).

- **Code quality checks**: \
  We use black, flake8, isort to maintain a common style and check the code. Please check your code install the pre-commit hook:

    ``` bash
    pre-commit install
    ```

    You can also check your changes manually:

    ``` bash
    pre-commit run --all-files
    ```

- **Testing with pytest**:

  ```bash
  poetry run pytest
  ```

- **Documentation with Sphinx**:

  ``` bash
  mkdir docs
  cd docs
  sphinx-apidoc -f -o source/ ../
  make html
  ```

- **Hosting with GitHub Pages**: \
  To publish the documentation on github pages you probably have to change some settings in the [GitHub Repository](https://github.com/Systems-Theory-in-Systems-Biology/EPIC)

  ``` text
  Settings -> Code and automation -> Pages -> Build and Deployment:
  - Source: Deploy from a branch
  - Branch: gh-pages && /(root)
  ```

- **Test Deployment with TestPyPi**: \
    You have to setup testpypi once:

    ```bash
    poetry config repositories.testpypi https://test.pypi.org/legacy/
    poetry config http-basic.testpypi __token__ pypi-your-api-token-here
    ```

    Build and deploy:

    ```bash
    poetry build
    poetry publish -r testpypi
    ```

    Test this with

    ```bash
    python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps epic
    ```

- **Deployment with PyPi**: \
    You have to setup pypi once:

    ```bash
    poetry config pypi-token.pypi pypi-your-token-here
    ```

    Build and deploy:

    ```bash
    poetry publish --build
    ```

    Test this with

    ```bash
    pip install epic
    ```
