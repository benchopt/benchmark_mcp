Benchmark repository for MCP
==============================

|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
Regression with the Minimax Concave Penalty (MCP) consists in solving the following program:

$$\\min_w \\frac{1}{2 n} \\Vert y - Xw \\Vert^2_2 \\ + \\ \\sum_j \\rho_{\\lambda, \\gamma}(w_j)$$

with the penalty

$$ \\rho_{\\lambda, \\gamma} (t) = \\begin{cases} \\lambda \\vert t \\vert - \\frac{t^2}{2\\gamma} & , & \\text{ if }  \\vert t \\vert \\ \\leq \\ \\gamma \\lambda \\\\ \\frac{\\lambda^2 \\gamma}{2} & , & \\text{ if } \\vert t \\vert \\ > \\ \\gamma \\lambda \\end{cases}$$

where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features and


$$y \\in \\mathbb{R}^n, \\ X \\in \\mathbb{R}^{n \\times p}$$

Install
--------

To demonstrate the use of benchopt on a simple benchmark configuration, one can run, from the `benchmark_mcp` folder:

.. code-block::

   $ benchopt install . -s cd -s pgd --env
   $ benchopt run . --config example_config.yml --env

Alternatively, one can use the command line interface to select which objectives, datasets and solvers are used:

.. code-block::

   $ benchopt run -s cd -s pgd -d simulated --max-runs 10 --n-repetitions 5


Use `benchopt run -h` for more details about these options.

.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
