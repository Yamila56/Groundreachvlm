#!/usr/bin/env bash
# Prepare an isolated CPU pilot environment on Ubuntu 24.04 x86-64.
set -euo pipefail

pilot_env="${PHYSVLM_PILOT_ENV:-$HOME/.local/share/physvlm-reachability}"
mkdir -p "$pilot_env/tools" "$pilot_env/native"
uv_bin="$pilot_env/tools/uv-x86_64-unknown-linux-gnu/uv"
if [[ ! -x "$uv_bin" ]]; then
    curl -fLsS --connect-timeout 10 --max-time 90 \
        https://github.com/astral-sh/uv/releases/download/0.11.0/uv-x86_64-unknown-linux-gnu.tar.gz \
        -o "$pilot_env/tools/uv.tgz"
    tar -xf "$pilot_env/tools/uv.tgz" -C "$pilot_env/tools"
fi
if [[ ! -x "$pilot_env/venv/bin/python" ]]; then
    "$uv_bin" venv --python 3.11 "$pilot_env/venv"
fi
"$uv_bin" pip install --python "$pilot_env/venv/bin/python" --only-binary :all: \
    pybullet==3.2.7 numpy scipy opencv-python-headless pandas openpyxl tqdm open3d==0.20.0

usb_lib="$pilot_env/native/usr/lib/x86_64-linux-gnu/libusb-1.0.so.0"
if [[ ! -e "$usb_lib" ]] && ! ldconfig -p | grep 'libusb-1.0.so.0' >/dev/null; then
    (
        cd "$pilot_env/native"
        apt-get download libusb-1.0-0
        dpkg-deb -x libusb-1.0-0_*.deb .
    )
fi