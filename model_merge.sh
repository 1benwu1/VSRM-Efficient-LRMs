#!/bin/bash

set -x

python3 /.../scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path "" \
    --local_dir "" \
    --target_dir ""

