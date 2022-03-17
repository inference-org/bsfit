#!/bin/bash

# build and deploy docs
cd docs/source;
make clean; 
make html;
cd ../..;

# deploy docs/source/ to docs/
cp -a docs/source/_build/html/. docs/ # copy to deploy path
touch docs/.nojekyll # ignore jekyll on github