<h1 align="center"> Epseon Backend </h1>

## Installation

Epseon Backend can be installed with `pip` from PyPI:

```
pip install epseon_backend
```

Alternatively, it is also possible to install it directly from repository:

```
pip install git+https://github.com/UniversityOfGdanskTeamPython/epseon_backend.git
```

## Development

To quickly set up development environment, first you have to install `poetry` globally:

```
pip install poetry
```

Afterwards you will be able to create development virtual environment:

```
poetry shell
```

Then You have to install dependencies into this environment:

```
poetry install --with=docs
```

And pre-commit hooks:

```
poe install-hooks
```

Now you are good to go. Whenever you commit changes, pre-commit hooks will be invoked.
If they fail or change files, you will have to re-add changes and commit again.

## Build from source

To build Epseon Backend you will need some dependencies which have to be installed
manually:

-   `Vulkan SDK`, version `1.3.268` is recommended, you can find it
    [here](https://vulkan.lunarg.com/sdk/home).
-   `Python 3` interpreter, version 3.8 - 3.12, can be downloaded from
    [here](https://www.python.org/downloads/).
-   `poetry`, version `1.6.1` is known to work correctly, which can be obtained with
    `pip install poetry`.

After obtaining all the dependencies you should be able to run following command:

```
poetry build
```

It should create `dist/` directory with `.whl` file which can be installed directly with
`pip`.

## Build documentation

To locally build documentation site, first you will need to install all documentation
related dependencies. This can be achieved with following command:

```
poetry install --with docs
```

Afterwards you can invoke `mkdocs` to generate documentation in form of HTML website:

```
mkdocs build
```

**Important** this is not how CI builds documentation, do not use this approach to
upload documentation to GitHub pages.
