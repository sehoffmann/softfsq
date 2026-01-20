#!/bin/bash

uv run python -m softfsq.train --mode entropic --softness 0.5 configs/base.yaml configs/taming-f8-pre.yaml configs/fsq_8192.yaml
