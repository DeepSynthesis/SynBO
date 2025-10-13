Quick Start Guide
=================

This guide will get you up and running with ReactionOpt in just a few minutes.

Basic Example
-------------

Here's a simple example of using ReactionOpt to optimize a reaction:

.. code-block:: python

    from rxnopt import ReactionOptimizer
    import pandas as pd
    import numpy as np

    # Create sample reaction data
    np.random.seed(42)
    data = pd.DataFrame({
        'temperature': np.random.uniform(20, 80, 50),
        'catalyst_loading': np.random.uniform(0.1, 10.0, 50),
        'solvent': np.random.choice(['DCM', 'THF', 'Toluene'], 50),
        'yield': np.random.uniform(0, 100, 50),
        'ee': np.random.uniform(0, 99, 50)
    })

    # Initialize the optimizer
    optimizer = ReactionOptimizer(
        objectives=['yield', 'ee'],
        n_initial_points=10,
        n_iterations=20
    )

    # Define parameter space
    parameter_space = {
        'temperature': [20, 80],
        'catalyst_loading': [0.1, 10.0],
        'solvent': ['DCM', 'THF', 'Toluene']
    }

    # Run optimization
    results = optimizer.optimize(data, parameter_space)

    # Visualize results
    optimizer.plot_optimization_history()
    optimizer.plot_pareto_front()

Multi-objective Optimization
-----------------------------

ReactionOpt excels at multi-objective optimization. You can optimize for multiple objectives simultaneously:

.. code-block:: python

    # Optimize for both yield and enantioselectivity
    optimizer = ReactionOptimizer(
        objectives=['yield', 'ee'],
        acquisition_function='EHVI',  # Expected Hypervolume Improvement
        n_initial_points=15,
        n_iterations=50
    )

Advanced Configuration
----------------------

For more control over the optimization process:

.. code-block:: python

    optimizer = ReactionOptimizer(
        objectives=['yield', 'ee'],
        acquisition_function='EHVI',
        surrogate_model='GP',  # Gaussian Process
        n_initial_points=20,
        n_iterations=100,
        random_seed=42,
        device='cuda',  # Use GPU acceleration
        verbose=True
    )

Loading Real Data
-----------------

Load your experimental data from various formats:

.. code-block:: python

    # From CSV
    data = pd.read_csv('reaction_data.csv')
    
    # From Excel
    data = pd.read_excel('reaction_data.xlsx', sheet_name='Sheet1')
    
    # Ensure required columns are present
    required_columns = ['temperature', 'catalyst_loading', 'yield', 'ee']
    assert all(col in data.columns for col in required_columns)

Next Steps
----------

* Check out the :doc:`tutorials/index` for more detailed examples
* Explore the :doc:`api/index` for complete function reference
* Look at :doc:`examples/index` for real-world use cases