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
  You can clone the repository over https or ssh. Use https if you only want to obtain the code. Use ssh if you are a registered as developer on the repository and want to push changes to the code base. If you want to contribute to the project but are not a registered developer, create a fork of the project first. In this case, you have to clone your fork, not this repository. </details>

- Install [uv](https://docs.astral.sh/uv/):
  
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
  
- Install dependencies:
  
  - For amici (sbml):

    ```bash
    sudo apt install swig
    sudo apt install libblas-dev
    sudo apt install libatlas-base-dev
    sudo apt install libhdf5-dev
    ```
  
  - For cpp:

    ```bash
    sudo apt install libeigen3-dev
    ```

- Install eulerpi:

  ```bash
  uv sync --all-extras
  ```

- Run the tests:

  ```bash
  uv run pytest
  ```

  You can add the ```--verbose``` parameter to get a more detailed report.

## Maintaining the repository

Here are the most important information on how to maintain this repository.

### Dependency Management with uv

We use uv to manage the project dependencies and the virtual python environment. During the [Quickstart](#quickstart) we installed all dependencies into the virtual environment, therefore:

---
**IMPORTANT**

Run all commands in the next section in the uv shell. It can be started with `source .venv/bin/activate`. Alternatively, you can run commands with `uv run <yourcommand>`.

---

Run ```uv add package_name``` to add the library/package with the name ```package_name``` as dependency to your project. Use ```uv add --group dev package_name``` to add ```package_name``` to your ```dev``` dependencies. You can have arbitrary group names.
  
For more information read the [uv Documentation](https://docs.astral.sh/uv/concepts/projects/).

### Code quality checks

We use ruff to maintain a common code style to lint the code. Please check your code install the pre-commit hook:

  ``` bash
  pre-commit install
  ```

  You can also check your changes manually:

  ``` bash
  pre-commit run --all-files
  ```

Testing with pytest

```bash
pytest
```

You can generate a coverage report by running the following code block in your terminal. Please be aware that it might take a long time, think about lowering the number of steps in the sampling.

```bash
coverage run -m pytest -v
coverage report
coverage html
```

### Running the tutorial (jupyter notebook)

The jupyter notebook can be run using

- vs code: https://code.visualstudio.com/docs/datascience/jupyter-notebooks
- shell + browser: `jupyter notebook`

In the first case you need to select the uv virtual env when selecting the interpreter, in the second case you need to run the command in the uv shell.

### Profiling with scalene

You can profile eulerpi with scalene (or gprofile) using the commands:

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

### Documentation with Sphinx

``` bash
cd docs
sphinx-apidoc -e -f -o source/ ../
make html
```

All extensions of sphinx which are used to create this documentation and further settings are stored in the file `docs/source/conf.py`.
If you add extensions to `conf.py` which are not part of sphinx, add them to the `docs/source/requirement.txt` file to allow github action `mmaraskar/sphinx-action@master` to still build the documentation.

A [cheatsheet](https://docs.typo3.org/m/typo3/docs-how-to-document/main/en-us/WritingReST/CheatSheet.html) for reStructuredText with Sphinx.

### Hosting with GitHub Pages

To publish the documentation on github pages you probably have to change some settings in the [GitHub Repository](https://github.com/Systems-Theory-in-Systems-Biology/EPI)

``` text
Settings -> Code and automation -> Pages -> Build and Deployment:
- Source: Deploy from a branch
- Branch: gh-pages && /(root)
```

### Changelog

We use the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format for the changelog. It should be updated with every pull request.

### Versioning

We use [Semantic Versioning](https://semver.org/). A version number is composed of three parts: major.minor.patch

1. The major version should be incremented when you make incompatible changes.
2. The minor version should be incremented when you add new functionality in a backward-compatible manner.
3. The patch version should be incremented when you make backward-compatible bug fixes.

Every time a new version is tagged, a GitHub Action workflow is triggered which builds and uploads the version to pypi.

Please update the version number in the `pyproject.toml` file before tagging the version.

### Test Deployment to TestPyPi

Build and deploy:

```bash
uv build
uv publish --index testpypi --token $TESTPYPI_TOKEN
```

Test this with

```bash
uvx --directory ./.. --no-project --with eulerpi --refresh-package eulerpi --index testpypi -- python -c "from importlib.metadata import version; import eulerpi; print(version('eulerpi')); print(eulerpi)"
```

### Deployment with GitHub CI (recommended)

0. Checkout the main branch `git checkout main`
1. Update the version number `X.X.X` in `CHANGELOG.md` and `pyproject.toml`
2. Set a new version tag `git tag -a vX.X.X -m "Release version X.X.X"` and push it `git push origin vX.X.X`
3. Check if the CI deployment was successful on [GitHub](https://github.com/Systems-Theory-in-Systems-Biology/EPI/actions/workflows/publish.yml) and finally on [PyPi](https://pypi.org/project/eulerpi/#history). The CI and PyPi may needs some time to run and update.

### Local Deployment (not recommended)

```bash
export UV_PUBLISH_TOKEN
```

Build and deploy:

```bash
uv build
uv publish --token $UV_PUBLISH_TOKEN
```

Test this with

```bash
uvx --directory ./.. --no-project --with eulerpi --refresh-package eulerpi -- python -c "from importlib.metadata import version; import eulerpi; print(version('eulerpi')); print(eulerpi)"
```

## Jax with CUDA

[Jax can be run with cuda on the gpu](https://github.com/google/jax#pip-installation-gpu-cuda). However, you need a recent nvidia-graphics-driver, the cuda-toolkit (cuda) and cudnn installed. Getting the versions right can cause headaches ;)

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

It can happen that old code is executed due to the generated pycache. For example, an old version of cuda or cudnn could be used. If you believe that this is happening:

```bash
pip install pyclean
py3clean .
```
