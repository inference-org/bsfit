Design pattern
==============

Codebase description
^^^^^^^^^^^^^^^^^^^^

The core codebase structure should remain as follows:

.. code-block:: console

    bsfit/
        nodes/
        pipes/
        requirements.txt
    conf/
    data/
    Docs/
        source/
    logs/
    tutorials/
    .gitignore
    deploy.sh
    main.py
    README.md
    setup.py


Source code design
^^^^^^^^^^^^^^^^^^

The source code organizes arount two central components:

#. `nodes`: they are python modules that organize semantically related functions.
#. `pipes` (for pipelines): they are analyses that chain nodes to transform data and produce plots.
  

Docstrings
==========

#. Please use the `Google Style Guide format`.
  
Update docs
============

#. Please edit `docs/source/`
#. Go to the "Build & deploy" section

Build & deploy
==============

Core steps: 

#. Develop your features on branch `develop`.
#. Pull request to obtain validation of your changes on branch `pre-prod` (homologation). 
#. A code master (I for now) will commit your final changes on branch `master` (production).

In detail:

#. On branch `develop` (to add features):
    After you've added a feature, run in the terminal: 

    .. code-block:: console

        bash deploy.sh

#. Pull request `pre-prod` branch (homologation, check docs rendering)
    After deploying sphinx/ to docs/, run in the terminal:

#. Pull request to `master` (production branch for end-users)
    when: the entire codebase is clean

Unit-testing
=============

Unit-test the package's functions:

Run in the terminal: 

.. code-block:: console

    pytest bsfit/test.py


Packaging
=========

.. code-block:: console

    # install setuptools
    pip install --upgrade setuptools wheel

    # package in development mode then import
    # run conda list bsfit or pip freeze to check
    # that it is listed among dependencies
    pip install -e .

