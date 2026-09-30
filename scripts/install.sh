#!/usr/bin/env bash
set -euo pipefail
export PIP_USER=0
export PYTHONNOUSERSITE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS_DIR="${CURRENT_DIR}/.tools"
VENV_DIR="${CURRENT_DIR}/.venv"
UV_BIN="${TOOLS_DIR}/bin/uv"
export PATH="${TOOLS_DIR}/bin:${PATH}"
export UV_CACHE_DIR="${CURRENT_DIR}/.cache/uv"
export PIP_CACHE_DIR="${CURRENT_DIR}/.cache/pip"
export UV_PYTHON_INSTALL_DIR="${TOOLS_DIR}/python"
export UV_PYTHON_BIN_DIR="${TOOLS_DIR}/bin"
ISAACLAB_RL_FRAMEWORK="${ISAACLAB_RL_FRAMEWORK:-none}"
# ── Helpers ────────────────────────────────────────────────────────────────────
info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m"; exit 1; }
pip_install() {
  python -m pip install "$@"
}
pip_install_with_isaac_constraints() {
  local constraints
  constraints="$(mktemp)"
  cat >"$constraints" <<'EOF'
numpy==1.26.0
packaging==23.0
typing_extensions==4.12.2
filelock==3.13.1
websockets==12.0
scipy==1.15.3
warp-lang==1.11.0
click==8.1.7
psutil==5.9.8
wheel==0.45.1
starlette==0.45.3
stable-baselines3<2.8
onnx>=1.18,<1.22
ipython<9
virtualenv==20.30.0
EOF
  if pip_install "$@" --constraint "$constraints"; then
    rm -f "$constraints"
  else
    local status=$?
    rm -f "$constraints"
    return "$status"
  fi
}
ensure_torch_cuda_stack() {
  if ! python -m pip show torchaudio &>/dev/null; then
    warn "    torchaudio missing, installing PyTorch cu128 stack..."
    pip_install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu128
  fi
}
pin_runtime_deps() {
  pip_install \
    "numpy==1.26.0" \
    "packaging==23.0" \
    "typing_extensions==4.12.2" \
    "filelock==3.13.1" \
    "websockets==12.0" \
    "click==8.1.7" \
    "psutil==5.9.8" \
    "wheel==0.45.1" \
    "starlette==0.45.3" \
    "scipy==1.15.3" \
    "warp-lang==1.11.0" \
    "onnx>=1.18,<1.22" \
    "ipython<9" \
    "virtualenv==20.30.0"
  # virtualenv 21.x pulled this in; virtualenv 20.x doesn't need it and it
  # requires filelock>=3.15.4 (conflicts with isaacsim-core's filelock==3.13.1)
  python -m pip uninstall -y python-discovery 2>/dev/null || true
}
# ── Step functions ─────────────────────────────────────────────────────────────
setup_tools() {
  for tool in curl git gcc g++ make; do
    command -v "$tool" >/dev/null 2>&1 || error "Missing host tool: $tool. This installer cannot add system packages."
  done
  git lfs version >/dev/null 2>&1 || error "git-lfs is required to fetch RoboDojo assets."
  if [[ ! -x "$UV_BIN" ]]; then
    info "[0/8] Installing uv under ${TOOLS_DIR}..."
    mkdir -p "$TOOLS_DIR/bin"
    # The official installer only writes the uv executable here; it does not edit shell profiles.
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${TOOLS_DIR}/bin" UV_NO_MODIFY_PATH=1 sh
  fi
  [[ -x "$UV_BIN" ]] || error "uv bootstrap did not create ${UV_BIN}"
}
activate_venv() {
  [[ -f "${VENV_DIR}/bin/activate" ]] || error "Missing ${VENV_DIR}; start with --from venv."
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]] \
    || error "${VENV_DIR} must use Python 3.11"
}
setup_venv() {
  info "[1/8] Creating repo-local Python 3.11 environment..."
  [[ -x "$UV_BIN" ]] || setup_tools
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    "$UV_BIN" venv --python 3.11 --seed "$VENV_DIR"
  fi
  activate_venv
  pip_install cmake ninja
}
setup_base_deps() {
  info "[2/8] Installing base pip dependencies..."
  pip_install -r "$CURRENT_DIR/scripts/requirements.txt"
  pip_install opencv-python-headless==4.11.0.86 pillow matplotlib "scipy==1.15.3" scikit-learn
  pip_install numpy==1.26.0
}
setup_submodules() {
  cd "$CURRENT_DIR" || exit 1
  local subs=(third_party/IsaacLab third_party/curobo XPolicyLab)
  info "[3/8] Initializing submodules at the commits pinned by this checkout..."
  git submodule sync -- "${subs[@]}"
  for sub in "${subs[@]}"; do
    info "    Initializing ${sub}..."
    git submodule update --init --progress -- "$sub" || {
      [ "$sub" = "XPolicyLab" ] && error "Failed to clone XPolicyLab. Ensure HTTPS auth (e.g. gh auth login)."
      error "Failed to update $sub."
    }
  done
  [ -f "XPolicyLab/client_server/ws/model_client.py" ] \
    || error "XPolicyLab init failed. Check repo access."
}
activate_cuda() {
  local candidate
  for candidate in "${CUDA_HOME:-}" "${TOOLS_DIR}/cuda-12.8" /usr/local/cuda-12.8; do
    [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]] || continue
    if "$candidate/bin/nvcc" --version | grep -q 'release 12\.8'; then
      export CUDA_HOME="$candidate"
      export PATH="${CUDA_HOME}/bin:${PATH}"
      export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
      return 0
    fi
  done
  return 1
}
setup_cuda() {
  if activate_cuda; then
    info "[4/8] Using CUDA toolkit 12.8 at ${CUDA_HOME}"
    return
  fi
  info "[4/8] Installing CUDA toolkit 12.8 under ${TOOLS_DIR}; host driver is untouched..."
  local installer="${CURRENT_DIR}/.cache/cuda/cuda_12.8.1_570.124.06_linux.run"
  mkdir -p "$(dirname "$installer")" "${TOOLS_DIR}/cuda-12.8" "${TOOLS_DIR}/cuda-defaultroot"
  curl -fL --retry 3 -C - -o "$installer" \
    https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
  sh "$installer" --silent --toolkit \
    --toolkitpath="${TOOLS_DIR}/cuda-12.8" \
    --defaultroot="${TOOLS_DIR}/cuda-defaultroot"
  activate_cuda || error "CUDA toolkit 12.8 installation failed"
}
setup_isaacsim() {
  if ! python -m pip show isaacsim 2>/dev/null | grep -q "5.1.0"; then
    info "[5/8] Installing PyTorch + IsaacSim 5.1..."
    pip_install --upgrade pip
    pip_install "numpy==1.26.0" "typing_extensions==4.12.2" "filelock==3.13.1"
    pip_install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu128
    pip_install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
    pin_runtime_deps
  else
    warn "[5/8] IsaacSim 5.1.0 already installed, skipping..."
    ensure_torch_cuda_stack
  fi
}
setup_isaaclab() {
  cd "$CURRENT_DIR" || exit 1
  if ! python -m pip show isaaclab &>/dev/null; then
    info "[6/8] Installing IsaacLab (rl-framework: ${ISAACLAB_RL_FRAMEWORK})..."
    cd third_party/IsaacLab || error "third_party/IsaacLab not found"
    export OMNI_KIT_ACCEPT_EULA=YES
    # isaaclab.sh calls `tabs`; fails when TERM=dumb (CI / piped shells)
    export TERM=xterm-256color
    ./isaaclab.sh --install "$ISAACLAB_RL_FRAMEWORK"
    cd "$CURRENT_DIR"
    ensure_torch_cuda_stack
    pin_runtime_deps
  else
    warn "[6/8] IsaacLab already installed, skipping..."
  fi
}
setup_curobo() {
  cd "$CURRENT_DIR" || exit 1
  local need_install=1
  if python -m pip show nvidia-curobo &>/dev/null; then
    if python - <<'PY' &>/dev/null
from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ToolPoseCriteria
PY
    then
      need_install=0
    fi
  fi
  if [ "$need_install" -eq 1 ]; then
    info "[7/8] Installing CuRobo..."
    cd third_party/curobo || error "third_party/curobo not found"
    python -m pip uninstall -y nvidia-curobo curobo 2>/dev/null || true
    pip_install_with_isaac_constraints -e ".[cu12]" --no-build-isolation
    cd "$CURRENT_DIR"
    pin_runtime_deps
  else
    warn "[7/8] CuRobo v2 already installed and importable, skipping..."
    pin_runtime_deps
  fi
}
# ── Entry point ────────────────────────────────────────────────────────────────
usage() {
  echo "Usage: $0 [-i | --from <step>]"
  echo "  -i, --install          Full install (all steps)"
  echo "  --from <step>          Resume from a specific step:"
  echo "                           tools | venv | base_deps | submodules | cuda | isaacsim | isaaclab | curobo"
  echo "  Tools, Python packages, CUDA toolkit, and caches stay under this checkout."
  echo "  ISAACLAB_RL_FRAMEWORK  IsaacLab RL extras to install (default: none; use all/sb3/skrl/etc. if needed)"
  echo "  -h, --help             Show this help"
}
run_from() {
  local from="$1"
  local steps=(tools venv base_deps submodules cuda isaacsim isaaclab curobo)
  local start=0
  local found=0
  for i in "${!steps[@]}"; do
    if [ "${steps[$i]}" == "$from" ]; then
      start=$i
      found=1
      break
    fi
  done
  [ "$found" -eq 1 ] || error "Unknown step '$from'. Valid: ${steps[*]}"
  if [[ "$from" != "tools" && "$from" != "venv" ]]; then
    activate_venv
  fi
  if [[ "$from" == "isaacsim" || "$from" == "isaaclab" || "$from" == "curobo" ]]; then
    activate_cuda || error "CUDA toolkit 12.8 missing; resume with --from cuda"
  fi
  for i in "${!steps[@]}"; do
    if [ "$i" -ge "$start" ]; then
      "setup_${steps[$i]}"
    fi
  done
}
case "${1:-}" in
  -h|--help)
    usage
    ;;
  -i|--install)
    run_from tools
    info "Develop environment setup completed."
    info "Activate the environment: source ${VENV_DIR}/bin/activate"
    ;;
  --from)
    [ -n "${2:-}" ] || { echo "Error: --from requires a step name"; usage; exit 1; }
    run_from "$2"
    info "Resumed from '$2' — done."
    info "Activate the environment: source ${VENV_DIR}/bin/activate"
    ;;
  *)
    usage
    exit 1
    ;;
esac

cd "$CURRENT_DIR" || exit
