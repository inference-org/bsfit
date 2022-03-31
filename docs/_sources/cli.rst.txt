Commands
--------

You can run most of the analyses in your terminal. This is the usage pattern:

.. code-block:: console

   python main.py --<my_command> <option> --<my_command> <option> ..

For instance, to quickly fit a standard Bayesian model, use the command below:

.. code-block:: console

   python main.py --model standard_bayes --analysis fit

All available commands are listed in the table below:

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - command arguments
     - options
     - description
   * - "--model"
     - "standard_bayes""
     - instantiate a standard bayesian model
   * - 
     - "cardinal_bayes""
     - instantiate a cardinal bayesian model
   * - "--analysis"
     - "fit"
     - fit the model
   * - 
     - "simulate_data"
     - simulate stochastic estimate data
