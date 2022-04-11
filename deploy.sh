#!/bin/bash
# [TODO]: It currently does not clean up 
# deprecated files in source/_autosummary
# by itself.

# build and deploy docs
cd docs/source;
make clean; 
make html;
cd ../..;

# deploy docs/source/ to docs/
# - copy to deploy path
# - ignore jekyll on github
cp -a docs/source/_build/html/. docs/ 
touch docs/.nojekyll 