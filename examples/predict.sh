#!/bin/bash

python -m ael.predict \
    Predict \
    ../tests/testdata/systems.dat \
    out/best_0.pth out/best_1.pth out/best_2.pth \
    -d ../tests/testdata \
    -e out/aevc.pth \
    -am out/amap.json \
    -cm out/cmap.json \
    -r 3.5 \
    -b 2 \
    -o out \