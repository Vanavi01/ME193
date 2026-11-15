"""Live lock on the PC with the webcam.
  python -m src.run_lock                         # simulated hardware (keys b / d)
  python -m src.run_lock --host <UnoQ IP>        # real servo, LEDs and button on the UnoQ
Keys: b = button (works in both modes), d = toggle door (simulation only), q = quit."""
import argparse
import time

import cv2

from . import code_store, config
from .camera import open_from_calibration, read_frame
from .infer import Predictor
from .io_backend import SimIO, UnoQNetIO
from .lock_policy import LockPolicy, State


def main():
    calib = config.load_calibration()
    if calib is None:
        raise SystemExit("No calibration. Run: python -m src.calibrate")
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=calib["camera_index"])
    ap.add_argument("--host", help="UnoQ IP address (omit for simulated hardware)")
    args = ap.parse_args()
    if not config.CODE_PATH.exists():
        raise SystemExit("No code set. Run: python -m src.code_store 2 5 3")

    io = UnoQNetIO(args.host) if args.host else SimIO()
    print("hardware:", f"UnoQ at {args.host}" if args.host else "simulated (keys b / d)")
    cap, _ = open_from_calibration(calib, args.camera)
    predictor = Predictor()
    policy = LockPolicy(code_store.verify)
    x, y, w, h = calib.get("border_roi", calib["roi"])

    try:
        while True:
            frame = read_frame(cap)
            pred = predictor.predict(frame) if policy.state is State.LISTEN else None
            out = policy.tick(time.monotonic(), pred, io.read_button(), io.read_door_open())
            io.apply(out)
            if out.event:
                print(out.event, out.state.name)

            view = frame.copy()
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if pred:
                cv2.putText(view, f"{pred[0]} ({pred[1]:.2f})", (x, max(25, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 128, 0), 2)
            for i, line in enumerate(io.overlay_lines()):
                cv2.putText(view, line, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 0), 2)
            cv2.imshow("impenetrable safe", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            io.handle_key(key)
    finally:
        io.close()                    # leaves the real safe locked
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
