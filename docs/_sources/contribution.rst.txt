Best practices
==============

Docstrings
==========

- Keep Docstrings in Google Style Guide format.
  
Update docs
============

#. Edit `docs/source/`
#. Go to the "Build & deploy" section

Build & deploy
==============

Basic steps: develop (features) -> pre-prod (homologation) -> master (production)

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

    pytest src/test.py
