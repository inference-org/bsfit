Commands
--------

Usage pattern (in the terminal):

.. code-block:: console

   python main.py --<my_command> <option> --<my_command> <option> ..

For example:

.. code-block:: console

   python main.py simulate_data_by_standard_bayes


Below we show a list of the available commands to run the built-in analyses:

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - command arguments
     - description
   * - "simulate_data_by_standard_bayes"
     - generates a dataset by simulating standard Bayes model's prediction for single trials
   * - "fit_standard_bayes"
     - fits the standard Bayes model to a built-in dataset:

.. list-table::
   :widths: 25 25 25
   :header-rows: 1

   * - command arguments
     - options
     - description
   * - "--model"
     - "standard_bayes""
     - instantiate a standard bayesian model
   * - ""
     - "cardinal_bayes""
     - instantiate a cardinal bayesian model
   * - "--analysis"
     - "fit"
     - fit the model
   * - ""
     - "simulate_data"
     - "simulate stochastic estimate data""
