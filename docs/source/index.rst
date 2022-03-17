.. bsfit documentation master file, created by
   sphinx-quickstart on Mon Mar 14 14:37:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to bsfit's software!
============================
..
   # This is a comment: Below, I create the
   main table of contents with sections
   intro, get_started ...

.. toctree::
   :maxdepth: 2
   :caption: Contents

   intro
   get_started
   tutorials   
   
..
   # This is a comment: Below, I created custom
   module and class templates to generate
   pages for functions and classes which
   autosummary does not do by default

.. autosummary::
   :toctree: _autosummary   
   :template: custom-module-template.rst
   :recursive:
   
   src

