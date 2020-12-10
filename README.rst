Benchmark repository for MCP
==============================

|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
The Lasso consists in solving the following program:

.. math::

    \min_w \frac{1}{2} \|y - Xw\|^2_2 + \sum_{j=1}^p \rho_{\gamma, \lambda}(w_j)

with 

.. math::

	\rho_{\lambda,\gamma}(t) =
	\begin{cases}
	\lambda |t| - \frac{t^2}{2\gamma} , &\text{if } |t| \leq \gamma\lambda ,\\
	\frac{\lambda^2 \gamma}{2} , &\text{if } |t| > \gamma \lambda.
	\end{cases}

where n (or n_samples) stands for the number of samples, p (or n_features) stands for the number of features and

.. math::

 y \in \mathbb{R}^n, X = [x_1^\top, \dots, x_n^\top]^\top \in \mathbb{R}^{n \times p}

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_lasso
   $ benchopt run ./benchmark_lasso

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run ./benchmark_lasso -s sklearn -d boston --max-runs 10 --n-repetitions 10


Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_lasso/workflows/build/badge.svg
   :target: https://github.com/benchopt/benchmark_lasso/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
