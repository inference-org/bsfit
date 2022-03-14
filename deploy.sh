#!/bin/bash

# build and deploy docs
cd docs/sphinx;
make clean; 
make html;
cd ../..;

# deploy docs/sphinx/ to docs/
cp -a docs/sphinx/_build/html/. docs/ # copy to deploy path
touch docs/.nojekyll # ignore jekyll on github