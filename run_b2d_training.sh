#!/usr/bin/env bash
set -euo pipefail

# ---- 환경 변수 ----
export RAY_DISABLE_LOGGING=1
export RAY_VERBOSITY=-1
export RAY_LOG_TO_STDERR=0

unset CUDA_VISIBLE_DEVICES
unset NVIDIA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export PYTHONPATH="/data1/shyuA6000/iPad"
export WORKDIR="/data1/shyuA6000/iPad"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data1/shyuA6000/iPad/navsim_2509/navsim/dataset/maps"
export NAVSIM_EXP_ROOT="/mnt/T7_2/iPad/exp"
export NAVSIM_DEVKIT_ROOT="/data1/shyuA6000/iPad/navsim"
export OPENSCENE_DATA_ROOT="/data1/shyuA6000/iPad/dataset"
export HYDRA_FULL_ERROR=1

# 헤드리스/즉시 출력
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
export PYTHONUNBUFFERED=1

export CARLA_ROOT=/mnt/mydisk/carla
export LD_LIBRARY_PATH="$CARLA_ROOT/LibLinux:${LD_LIBRARY_PATH:-}"

# conda lib 우선
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "[Conda Env] $CONDA_DEFAULT_ENV @ $CONDA_PREFIX"
echo "[LD_LIBRARY_PATH] $LD_LIBRARY_PATH"
echo "[CUDA_VISIBLE_DEVICES] $CUDA_VISIBLE_DEVICES"

cd "$WORKDIR"

SCRIPT_PATH="$WORKDIR/navsim/planning/script/run_b2d_training.py"

# ---- 기본 학습 인자 설정 ----
DEFAULT_ARGS=(
  trainer.params.accelerator=gpu
  trainer.params.devices=4
  trainer.params.strategy=ddp
  trainer.params.precision=16-mixed
  dataloader.params.batch_size=32
  dataloader.params.num_workers=20
  hydra.run.dir="${NAVSIM_EXP_ROOT}/b2d_ddp_run"
)

# ---- 스크립트 실행 ----
echo "[Running] python -u $SCRIPT_PATH ${DEFAULT_ARGS[*]} $*"
python -u "$SCRIPT_PATH" "${DEFAULT_ARGS[@]}" "$@"
