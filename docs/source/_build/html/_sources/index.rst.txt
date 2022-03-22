.. 
   # This is a comment: bsfit documentation master file, created by
   sphinx-quickstart on Mon Mar 14 14:37:58 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive. 
   End of comment.

Welcome to bsfit's software!
============================

Python software to model psychophysics data
with Bayesian and heuristics models.

#. Simple API inspired by scikit-learn 
#. Free software: BSD license 

..
   # This is a comment: Below, I create the
   main table of contents with sections
   intro, get_started ...End of comment.

.. toctree::
   :maxdepth: 3
   :caption: Getting started

   installation
   tutorials
      
..
   # This is a comment: Below, I created custom
   module and class templates to generate
   pages for functions and classes which
   autosummary does not do by default.
   End of comment.

.. autosummary::
   :toctree: _autosummary   
   :template: custom-module-template.rst
   :recursive:
   
   src

.. toctree::
   :maxdepth: 3
   :caption: contribution

   contribution


Citing
======

If you want to cite bsfit, please refer to the publication 
"A Switching Observer for human perceptual estimation" 