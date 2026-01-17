#!/bin/bash

uv run dmlrun -n 1 python -m softfsq.train configs/base.yaml configs/taming-f8-pre.yaml
