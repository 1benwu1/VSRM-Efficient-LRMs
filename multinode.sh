set -e

set -o pipefail

hope_torch_distributed_launch=$(python3 hope_torch_distributed_launch.py)
echo $hope_torch_distributed_launch train.py $@

$hope_torch_distributed_launch train.py $@