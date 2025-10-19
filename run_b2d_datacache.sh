#!/usr/bin/env bash
set -euo pipefail

# # ---- conda 초기화 & 활성화 ----
# if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
#   . /opt/conda/etc/profile.d/conda.sh
# elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#   . "$HOME/anaconda3/etc/profile.d/conda.sh"
# elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#   . "$HOME/miniconda3/etc/profile.d/conda.sh"
# else
#   echo "conda.sh not found"; exit 1
# fi
# conda activate ipad

# ---- 환경 변수 ----
export PYTHONPATH="/data1/shyuA6000/iPad"
export WORKDIR="/data1/shyuA6000/iPad"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data1/shyuA6000/iPad/navsim_2509/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/mnt/T7_2/iPad/exp"
export NAVSIM_DEVKIT_ROOT="/data1/shyuA6000/iPad/navsim"
export OPENSCENE_DATA_ROOT="/data1/shyuA6000/iPad/dataset"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

# 헤드리스/즉시 출력
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export PYTHONUNBUFFERED=1

# conda lib 우선
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "[Conda Env] $CONDA_DEFAULT_ENV @ $CONDA_PREFIX"
echo "[LD_LIBRARY_PATH] $LD_LIBRARY_PATH"

cd "$WORKDIR"

SCRIPT_PATH="$WORKDIR/Bench2Drive/leaderboard/pad_team_code/b2d_datacache.py"

# 인자 전달: 스크립트 호출 시 넘긴 모든 인자를 그대로 전달
# 예) bash run_b2d_cache.sh worker=sequential dataloader.params.num_workers=0
echo "[Running] python -u $SCRIPT_PATH $*"
python -u "$SCRIPT_PATH" "$@"
