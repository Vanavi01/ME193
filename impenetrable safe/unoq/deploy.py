"""Copy what the board needs to ~/safe on the UnoQ (over SSH). Run on the PC:
    python unoq/deploy.py 10.243.84.49
Sends: src/ (code), the trained model, your hashed passcode, the servo timing, the guard photos
and the board scripts. NOT the training pictures and NOT the PC's camera calibration (the board's
camera is calibrated on the board: python3 -m src.calibrate --quick)."""
import io
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["models/finger_net.pt", "models/code.json", "data/_calibration/servo.json",
         "unoq/setup_board.sh", "unoq/safe-vision.service"]
DIRS = ["src", "assets"]


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    tar.add(p, arcname=p.relative_to(ROOT).as_posix())
        for f in FILES:
            p = ROOT / f
            if p.exists():
                tar.add(p, arcname=f)
            else:
                print(f"  (skipping missing {f})")
    return buf.getvalue()


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python unoq/deploy.py <UnoQ IP> [user]")
    host, user = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "arduino")
    data = bundle()
    print(f"sending {len(data) / 1e6:.1f} MB to {user}@{host}:~/safe ...")
    r = subprocess.run(["ssh", f"{user}@{host}",
                        "mkdir -p ~/safe && tar -xzf - -C ~/safe && echo copied OK"],
                       input=data)
    if r.returncode:
        raise SystemExit("copy failed (is SSH working? try: ssh arduino@<IP>)")
    print("next, on the board:  sh ~/safe/unoq/setup_board.sh")


if __name__ == "__main__":
    main()
