# Development

## Quickstart

- Clone the repository:

  ```bash
  git clone https://github.com/Systems-Theory-in-Systems-Biology/EPI.git
  ```

  ```bash
  git clone git@github.com:Systems-Theory-in-Systems-Biology/EPI.git
  ```

  <details>
  <summary>Should I choose https or ssh?</summary>
  You can clone the repository over https or ssh. Use https if you only want to obtain the code. Use ssh if you are a registered as developer on the repository and want to push changes to the code base. If you want to contribute to the project but are not a registered developer, create a fork of the project first. In this case you have to clone your fork, not this repository. </details>

- Install [poetry](https://python-poetry.org/docs/):
  
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
  
- Install dependencies:
  
  - For amici (sbml):

    ```bash
    sudo apt install swig
    sudo apt install libblas-dev
    sudo apt install libatlas-base-dev
    ```
  
  - For cpp:

    ```bash
    sudo apt install cmake
    sudo apt install libeigen3-dev
    sudo apt install pybind11-dev
    ```

- Install epi:

  ```bash
  poetry install --with=dev
  ```

- Run the tests:

  ```bash
  poetry run pytest
  ```

  You can add the ```--verbose``` parameter to get a more detailed report.

## Maintaining the repository

Here are the most important infos on how to maintain this repository.

- **Dependency Management with Poetry**: \
  We use poetry as build system, for the dependency management and the virtual environment. During the [Quickstart](#quickstart) we installed all dependencies into the virtual environment, therefore:

  ---
  **IMPORTANT**

  Run all commands in the next section in the poetry shell. It can be started with `poetry shell`. Alternatively you can run commands with `poetry run <yourcommand>`.

  ---

  Run ```poetry add package_name``` to add the library/package with the name ```package_name``` as dependencie to your project. Use ```poetry add --group dev package_name``` to add ```package_name``` to your ```dev``` dependencies. You can have arbitrary group names.
  
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
  pytest
  ```

  You can generate a coverage report by running the following code block in your terminal. Please be aware that it might take a long time, think about lowering the number of steps in the sampling.

  ```bash
  coverage run -m pytest -v
  coverage report
  coverage html
  ```

- **Profiling with scalene**:

  You can profile epi with scalene (or gprofile) using the commands:

  ```bash
  python3 -m pip install -U scalene
  scalene tests/profiling.py
  ```

  This will create a `profile.html` file, which you can open using your browser. Do not rely on the OPENAI optimization proposals. They are often plain wrong in scalene.

<!-- TODO: Add a docker development environment -->
<!-- - **Working with docker**:

  ```bash
  curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
  sudo service docker start
  sudo docker run hello-world
  sudo service docker stop
  ``` -->

- **Documentation with Sphinx**:

  ``` bash
  cd docs
  sphinx-apidoc -f -o source/ ../
  make html
  ```

  All extensions of sphinx which are used to create this documentation and further settings are stored in the file `docs/source/conf.py`.
  If you add extensions to `conf.py` which are not part of sphinx, add them to the `docs/source/requirement.txt` file to allow github action `mmaraskar/sphinx-action@master` to still build the documentation.

  A [cheatsheet](https://docs.typo3.org/m/typo3/docs-how-to-document/main/en-us/WritingReST/CheatSheet.html) for reStructuredText with Sphinx.

- **Hosting with GitHub Pages**: \
  To publish the documentation on github pages you probably have to change some settings in the [GitHub Repository](https://github.com/Systems-Theory-in-Systems-Biology/EPI)

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
    python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps epi
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
    pip install epi
    ```

## Jax with CUDA

[Jax can be run with cuda on the gpu](https://github.com/google/jax#pip-installation-gpu-cuda). However you need a recent nvidia-graphics-driver, the cuda-toolkit (cuda) and cudnn installed. Getting the versions right can cause headaches ;)

I used the following tricks to get it running:

```bash
# for cuda toolkit
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda-12.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11-fake/lib64:$LD_LIBRARY_PATH"
```

Follow [this issue](https://github.com/google/jax/issues/13637) to see whether you possibly need to create the cuda-11-fake folder as a copy from the 12.0 folder and "create the .so.11" libraries as symlinks using the script below.

```bash
#!/bin/bash

# Find all files in the current directory that match the pattern "lib*.so"
for file in lib*.so; do
    # Extract the base name of the file (without the ".so" extension)
    base_name="${file%.*}"

    # Construct the name of the symbolic link we want to create
    link_name="${base_name}.so.11"

    # Check if the link already exists
    if [ ! -e "$link_name" ]; then
        # Create the symbolic link if it doesn't exist
        ln -s "$file" "$link_name"
    fi
done
```

It can happen that old code is executed due to the generated pycache. For example an old version of cuda or cudnn could be used. If you believe that this is happening:

```bash
pip install pyclean
py3clean .
```
