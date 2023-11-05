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

## Build Python from source on Linux

When building Python interpreter on Linux to be used for extension development,
`libpython3.x` must be compiled with `-fPIC` to generate code which can be embedded into
shared library. To do that configure Python build with following command:

```bash
CFLAGS=-fPIC ./configure --enable-shared=no --enable-optimizations
```

then you can follow up same as in any other build:

```bash
make
sudo make altinstall
```

Altinstall above causes Python to be installed as a secondary Python version. If it is
only Python version you have on your machine, use `sudo make install` instead.
