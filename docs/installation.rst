Installation
============

ReactionOpt requires Python 3.8 or later.

From PyPI (Recommended)
-----------------------

The easiest way to install ReactionOpt is using pip:

.. code-block:: bash

    pip install reactionopt

Development Installation
------------------------

If you want to contribute to ReactionOpt or use the latest development version:

.. code-block:: bash

    git clone https://github.com/yourusername/reactionopt.git
    cd reactionopt
    pip install -e .

With Development Dependencies
-----------------------------

To install with all development dependencies (testing, linting, documentation):

.. code-block:: bash

    pip install -e ".[dev]"

Dependencies
------------

ReactionOpt depends on the following packages:

* PyTorch >= 1.9.0
* Botorch >= 0.6.0
* Ax-platform >= 0.2.0
* NumPy >= 1.20.0
* Pandas >= 1.3.0
* Scikit-learn >= 1.0.0
* RDKit >= 2021.9.1
* Matplotlib >= 3.3.0
* Seaborn >= 0.11.0

GPU Support
-----------

ReactionOpt supports GPU acceleration through PyTorch. To use GPU features:

1. Install PyTorch with CUDA support
2. Ensure your system has a compatible NVIDIA GPU
3. Set the `device='cuda'` parameter when initializing the optimizer

Verification
------------

To verify your installation, run:

.. code-block:: python

    import synbo
    print(synbo.__version__)

    # Create a simple optimizer
    from synbo import ReactionOptimizer
    optimizer = ReactionOptimizer(objectives=['yield'])
    print("Installation successful!")