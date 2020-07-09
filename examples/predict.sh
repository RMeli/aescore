#!/bin/bash

python -m ael.predict \
    Predict \
    ../tests/testdata/systems.dat \
    -d ../tests/testdata \
    -m out/best_0.pth \
    -e out/aevc.pth \
    -am out/amap.json \
    -cm out/cmap.json \
    -r 3.5 \
    -b 2 \
    -o out \