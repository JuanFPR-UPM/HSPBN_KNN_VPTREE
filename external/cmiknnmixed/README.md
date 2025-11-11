# Conditional Independence Testing for Mixed-type Variables

This repository is the official implementation to reproduce the experiments described in "Non-parametric Conditional Independence Testing for Mixed Continuous-Categorical Variables: A Novel Method and Numerical Evaluation".

## Requirements

To install the requirements create an environment and install the requirements file:

```setup
pip install -r requirements.txt
```

## License

You can redistribute and/or modify the code under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

The packages listed in the ```requirements.txt``` have the following licenses.

| Package                   | Version    | License        | Description |
|---------------------------|------------|----------------|-------------|
| anyio                     | 3.5.0      | MIT            | Async I/O framework. |
| argon2-cffi               | 21.3.0     | MIT            | Secure password hashing via Argon2. |
| argon2-cffi-bindings      | 21.2.0     | MIT            | Low-level C bindings for argon2-cffi. |
| asttokens                 | 2.0.5      | Apache 2.0     | Python code parsing and token analysis. |
| async-lru                 | 2.0.4      | MIT            | LRU cache for asyncio. |
| attrs                     | 23.1.0     | MIT            | Python classes with minimal boilerplate. |
| Babel                     | 2.11.0     | BSD            | Internationalization library for Python. |
| backcall                  | 0.2.0      | BSD            | Provides callback handling for Python. |
| beautifulsoup4            | 4.12.2     | MIT            | Screen-scraping library for parsing HTML and XML. |
| bleach                    | 4.1.0      | Apache 2.0     | HTML sanitizing library. |
| Bottleneck                | 1.3.5      | BSD            | Fast array operations for NumPy. |
| Brotli                    | 1.0.9      | MIT            | Compression algorithm library. |
| cairocffi                 | 1.6.1      | BSD            | CFFI-based Cairo bindings. |
| CairoSVG                  | 2.7.1      | LGPL 3.0       | SVG to other formats converter using Cairo. |
| certifi                   | 2023.11.17 | MPL 2.0        | Provides Mozilla's root certificates for Python. |
| cffi                      | 1.16.0     | MIT            | Foreign Function Interface for Python. |
| charset-normalizer        | 2.0.4      | MIT            | Character encoding detection library. |
| comm                      | 0.1.2      | BSD            | Messaging for Jupyter kernels. |
| contourpy                 | 1.2.0      | BSD            | Contouring library for matplotlib. |
| cryptography              | 41.0.7     | Apache 2.0     | Library for cryptographic recipes and primitives. |
| cssselect2                | 0.7.0      | BSD            | CSS selectors for XML and HTML documents. |
| cycler                    | 0.12.1     | BSD            | Composable cycles of style properties for matplotlib. |
| debugpy                   | 1.6.7      | MIT            | Debugger for Python. |
| decorator                 | 5.1.1      | BSD            | Simplifies the usage of decorators in Python. |
| defusedxml                | 0.7.1      | PSFL           | XML bomb protection for Python. |
| exceptiongroup            | 1.0.4      | MIT            | Manage multiple exceptions in a single group. |
| executing                 | 0.8.3      | MIT            | Get information about executing code in Python. |
| fastjsonschema            | 2.16.2     | BSD            | Fast JSON schema validation. |
| fonttools                 | 4.47.0     | MIT            | Tools to manipulate font files. |
| idna                      | 3.4        | BSD            | Support for Internationalized Domain Names in Python. |
| ipykernel                 | 6.25.0     | BSD            | IPython kernel for Jupyter. |
| ipython                   | 8.15.0     | BSD            | Interactive computing in Python. |
| jedi                      | 0.18.1     | MIT            | Autocompletion and static analysis library for Python. |
| Jinja2                    | 3.1.2      | BSD            | Template engine for Python. |
| joblib                    | 1.3.2      | BSD            | Lightweight library for parallel computing. |
| json5                     | 0.9.6      | Apache 2.0     | JSON5 parser for Python. |
| jsonschema                | 4.19.2     | MIT            | JSON Schema validation library. |
| jsonschema-specifications | 2023.7.1   | MIT            | Specification support for JSON Schema. |
| jupyter_client            | 8.6.0      | BSD            | Jupyter protocol client library. |
| jupyter_core              | 5.5.0      | BSD            | Core package for Jupyter projects. |
| jupyter-events            | 0.8.0      | BSD            | Event logging for Jupyter. |
| jupyter-lsp               | 2.2.0      | MIT            | Jupyter support for Language Server Protocol. |
| jupyter_server            | 2.10.0     | BSD            | Core server application for Jupyter. |
| jupyter_server_terminals  | 0.4.4      | BSD            | Terminal support for Jupyter server. |
| jupyterlab                | 4.0.8      | BSD            | Web-based interactive development environment. |
| jupyterlab-pygments       | 0.1.2      | BSD            | Syntax highlighting for JupyterLab. |
| jupyterlab_server         | 2.25.1     | BSD            | Backend server for JupyterLab. |
| kiwisolver                | 1.4.5      | BSD            | Efficient algorithms for constraint solving. |
| llvmlite                  | 0.39.1     | BSD            | Lightweight LLVM binding for Python. |
| MarkupSafe                | 2.1.3      | BSD            | Safely handle untrusted HTML and XML. |
| matplotlib                | 3.8.2      | PSFL            | Plotting library for Python. |
| matplotlib-inline         | 0.1.6      | BSD            | Inline plotting for Jupyter notebooks. |
| mistune                   | 2.0.4      | BSD            | Fast Markdown parser in pure Python. |
| mkl-fft                   | 1.3.8      | BSD            | FFT functions powered by Intel MKL. |
| mkl-random                | 1.2.4      | BSD            | Random number generators powered by Intel MKL. |
| mkl-service               | 2.4.0      | BSD            | Simplifies usage of Intel MKL in Python. |
| nbclient                  | 0.8.0      | BSD            | Client for executing Jupyter Notebooks. |
| nbconvert                 | 7.10.0     | BSD            | Convert Jupyter Notebooks to other formats. |
| nbformat                  | 5.9.2      | BSD            | Notebook format support for Jupyter. |
| nest-asyncio              | 1.5.6      | BSD            | Allows nested asyncio event loops in Python. |
| networkx                  | 3.2.1      | BSD            | Network analysis library for Python. |
| notebook_shim             | 0.2.3      | BSD            | Shim to support compatibility with legacy notebook code. |
| numba                     | 0.56.4     | BSD            | Just-in-time compiler for Python functions. |
| numexpr                   | 2.8.7      | MIT            | Fast numerical array expression evaluation. |
| numpy                     | 1.23.5     | BSD            | Core library for numerical computing in Python. |
| overrides                 | 7.4.0      | Apache 2.0     | Decorator for method overrides. |
| packaging                 | 23.1       | Apache 2.0     | Utilities for packaging Python projects. |
| pandas                    | 2.1.4      | BSD            | Data analysis and manipulation library. |
| pandocfilters             | 1.5.0      | BSD            | Utilities for manipulating pandoc ASTs. |
| parso                     | 0.8.3      | MIT            | Python parser library. |
| pexpect                   | 4.8.0      | ISC            | Control interactive applications in Python. |
| pickleshare               | 0.7.5      | MIT            | File system-based database for Python. |
| pillow                    | 10.2.0     | MIT-CMU        | Imaging library for Python. |
| pip                       | 23.3.1     | MIT            | Python package installer. |
| platformdirs              | 3.10.0     | MIT            | Cross-platform directories module. |
| prometheus-client         | 0.14.1     | Apache 2.0     | Prometheus instrumentation for Python applications. |
| prompt-toolkit            | 3.0.43     | BSD            | Library for building command line interfaces. |
| psutil                    | 5.9.0      | BSD            | Process and system utilities for Python. |
| ptyprocess                | 0.7.0      | ISC            | Run processes in pseudo terminals in Python. |
| pure-eval                 | 0.2.2      | MIT            | Pure evaluation of Python expressions. |
| pycparser                 | 2.21       | BSD            | C parser in Python. |
| Pygments                  | 2.15.1     | BSD            | Syntax highlighting for code. |
| pyOpenSSL                 | 23.2.0     | Apache 2.0     | Python bindings for OpenSSL. |
| pyparsing                 | 3.1.1      | MIT            | Text processing library. |
| PySocks                   | 1.7.1      | BSD            | SOCKS protocol support for Python. |
| python-dateutil           | 2.8.2      | BSD            | Date and time manipulation library. |
| python-json-logger        | 2.0.7      | BSD            | JSON-based logger for Python. |
| pytz                      | 2023.3.post1 | MIT          | Time zone library for Python. |
| PyYAML                    | 6.0.1      | MIT            | YAML parser and emitter for Python. |
| pyzmq                     | 25.1.0     | BSD            | ZeroMQ bindings for Python. |
| referencing               | 0.30.2     | MIT            | JSON references library. |
| requests                  | 2.31.0     | Apache 2.0     | HTTP library for Python. |
| rfc3339-validator         | 0.1.4      | MIT            | Validator for RFC3339 timestamps. |
| rfc3986-validator         | 0.1.1      | MIT            | Validator for RFC3986 URIs. |
| rpds-py                   | 0.10.6     | MIT            | Immutable data structures for Python. |
| scikit-learn              | 1.4.0rc1   | BSD            | Machine learning library for Python. |
| SciPy                     | 1.12.0rc1  | BSD            | Core library for scientific computing. |
| Send2Trash                | 1.8.2      | BSD            | Send files to trash/recycle bin in Python. |
| setuptools                | 68.2.2     | MIT            | Package building and distribution library for Python. |
| six                       | 1.16.0     | MIT            | Python 2 and 3 compatibility library. |
| sniffio                   | 1.2.0      | MIT            | Detect which async library is in use. |
| soupsieve                 | 2.5        | MIT            | CSS selector support for BeautifulSoup. |
| stack-data                | 0.2.0      | MIT            | Utilities for working with Python stack frames. |
| terminado                 | 0.17.1     | BSD            | Terminal emulator for Jupyter. |
| threadpoolctl             | 3.2.0      | BSD            | Control thread-pools in Python. |
| tigramite                 | 5.2.2.3    | GNU            | Causal discovery library for time series data. |
| tinycss2                  | 1.2.1      | BSD            | Parser for CSS in Python. |
| tomli                     | 2.0.1      | MIT            | TOML parser for Python. |
| tornado                   | 6.3.3      | Apache 2.0     | Asynchronous networking library for Python. |
| traitlets                 | 5.7.1      | BSD            | Configuration and introspection library for Python. |
| typing_extensions         | 4.7.1      | PSFL           | Backports for typing features in Python. |
| tzdata                    | 2023.3     | MIT            | Time zone data for Python. |
| urllib3                   | 1.26.18    | MIT            | HTTP library with thread safety and connection pooling. |
| wcwidth                   | 0.2.5      | MIT            | Determines printable width of a string in a terminal. |
| webencodings              | 0.5.1      | BSD            | Web browser encoding conversion library. |
| websocket-client          | 0.58.0     | BSD            | WebSocket client for Python. |
| wheel                     | 0.41.2     | MIT            | Build platform for Python packages. |
