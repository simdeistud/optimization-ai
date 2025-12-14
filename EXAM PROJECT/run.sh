 # Files need to be copied to the correct locations inside the PonyGE2 package
cp parameters deps/ponyge2/parameters
cp lilypond_grammar.bnf deps/ponyge2/grammars/lilypond_grammar.bnf
cp score_ff.py deps/ponyge2/src/fitness/score_ff.py
source ./.venv/bin/activate
lilypond --png -o "/tmp/targetscore" target.ly
cd deps/ponyge2/src/
python ponyge.py --parameters parameters
deactivate