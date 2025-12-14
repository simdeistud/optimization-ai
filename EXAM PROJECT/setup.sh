#!/bin/bash
# This script sets up the environment for the EXAM PROJECT
git clone https://github.com/PonyGE/PonyGE2.git deps/ponyge2
python -m venv ./.venv
source ./.venv/bin/activate
pip install -r ./deps/ponyge2/requirements.txt
pip install -r requirements.txt
deactivate

# Files need to be copied to the correct locations inside the PonyGE2 package
cp parameters deps/ponyge2/parameters
cp lilypond_grammar.bnf deps/ponyge2/grammars/lilypond_grammar.bnf
cp score_ff.py deps/ponyge2/src/fitness/score_ff.py