#!/bin/bash

mkdir -p out

python -m ael.train \
    Train \
    ../tests/testdata/systemsvs.dat \
    ../tests/testdata/systemsvs.dat \
    -t ../tests/testdata/systemsvs.dat \
    -d ../tests/testdata \
    -vs ../tests/testdata \
    -r 3.5 \
    -p 0.5 \
    -lr 0.0005 \
    -b 2 \
    -l 256 128 64 1 \
    -e 10 \
    -c 3 \
    -cm '{"X": ["P", "S"]}' \
    --removeHs \
    --scale \
    -o out \
    --plot \
    --seed 42
