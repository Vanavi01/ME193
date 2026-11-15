#!/bin/sh
# Run ON the UnoQ (as the arduino user) after unoq/deploy.py copied the project to ~/safe.
# Creates a Python environment with PyTorch + OpenCV and installs the autostart service.
#   sh ~/safe/unoq/setup_board.sh
set -e
cd "$HOME/safe"

echo "== python: $(python3 --version)"
if [ ! -x .venv/bin/pip ]; then
  rm -rf .venv
  if ! python3 -m venv .venv 2>/dev/null; then
    # no python3-venv package (needs sudo): make the venv without pip, then bootstrap pip
    rm -rf .venv
    python3 -m venv --without-pip .venv
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    .venv/bin/python /tmp/get-pip.py
  fi
fi
. .venv/bin/activate
pip install --upgrade pip wheel
pip install numpy opencv-python-headless
# CPU-only PyTorch for the board's ARM processor (same model file as on the PC)
pip install torch --index-url https://download.pytorch.org/whl/cpu

python - <<'EOF'
import cv2, numpy, torch
print("== versions: torch", torch.__version__, "| opencv", cv2.__version__, "| numpy", numpy.__version__)
from src.infer import Predictor
p = Predictor()
print("== model loaded OK:", sum(x.numel() for x in p.net.parameters()), "parameters")
EOF

# camera devices the board can see
echo "== cameras:"; ls /dev/video* 2>/dev/null || echo "   none - plug the webcam into the UnoQ's USB hub"

# autostart service (needs sudo once)
if [ "${1:-}" = "--service" ]; then
  sudo cp unoq/safe-vision.service /etc/systemd/system/safe-vision.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now safe-vision
  echo "== service installed: sudo systemctl status safe-vision   |   journalctl -u safe-vision -f"
fi
echo "== done"
