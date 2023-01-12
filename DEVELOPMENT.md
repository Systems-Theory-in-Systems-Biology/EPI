# Development

## Quickstart

- Clone the repository:

  ```bash
  git clone https://github.com/Systems-Theory-in-Systems-Biology/EPIC.git
  ```

  ```bash
  git clone git@github.com:Systems-Theory-in-Systems-Biology/EPIC.git
  ```

  <details>
  <summary>Should I choose https or ssh?</summary>
  You can clone the repository over https or ssh. Use https if you only want to obtain the code. Use ssh if you are a registered as developer on the repository and want to push changes to the code base. If you want to contribute to the project but are not a registered developer, create a fork of the project first. In this case you have to clone your fork, not this repository. </details>

- Install epic:

  ```bash
  cd EPIC && pip install poetry && poetry install --with=dev
  ```

- Run the tests:

  ```bash
  poetry run pytest
  ```

## Maintaining the repository

Here are the most important infos on how to maintain this repository.

- **Dependency Management with Poetry**: \
  We use poetry as build system and for the dependency management. Most commans can be simply run in the virtual environment by prepending ```poetry run``` before the command. You can also use ```poetry shell``` to activate the virtual environment and exit it with ```exit```. Run ```poetry add package_name``` to add the library/package with the name ```package_name``` as dependencie to your project. Use ```poetry add --group dev package_name``` to add ```package_name``` to your ```dev``` dependencies. You can have arbitrary group names.
  
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
