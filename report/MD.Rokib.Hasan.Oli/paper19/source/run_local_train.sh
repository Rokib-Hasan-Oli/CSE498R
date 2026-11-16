#!/bin/bash
# The extra runs of local_train.py, one at a time (Git Bash), ~9 min each on the RTX 5060 Ti. Arg 1 = unpacked results.zip.
PY=D:/498R/_dataset_build/venv/Scripts/python.exe
RUN="$1"
for s in 42 1 2; do $PY local_train.py "$RUN" failure quality ce $s; done
for s in 1 2; do for l in ema ce; do $PY local_train.py "$RUN" kit baseline4 $l $s; done; done
for s in 42 1; do $PY local_train.py "$RUN" balanced baseline4 ema $s; done
for s in 42 1; do $PY local_train.py "$RUN" text baseline4 ema $s; done
for s in 1 2; do for l in ema ce; do $PY local_train.py "$RUN" kit quality $l $s; done; done
echo ALL_DONE
