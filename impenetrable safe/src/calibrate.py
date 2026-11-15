"""Camera calibration.
  python -m src.calibrate --view         # live view: is the border detected? (n = next cam)
  python -m src.calibrate --check        # is the saved calibration still valid? (square empty)
  python -m src.calibrate --relight      # lighting changed: keep the crop, redo background + hand
  python -m src.calibrate                # full calibration (camera moved/rotated, or first time)
  python -m src.calibrate --probe        # which camera index sees the border?
Options: --camera N, --auto-exposure (default is a locked exposure chosen during calibration)

Full calibration: (1) find the tape border (c = accept, m = draw the box by hand),
(2) capture the EMPTY area as the background reference, (3) hold your OPEN hand still to
measure hand size and sharpness. Old calibrations are backed up to data/_calibration/history/,
and the new one is compared with the old one so you know if the model needs new data."""
import argparse
import json
import shutil
import time

import cv2
import numpy as np

from . import config
from .camera import choose_exposure, open_camera, open_from_calibration, read_frame
from .preprocess import crop_roi, model_view
from .quality import (SIDES, border_red_fraction, calibration_drift, compare_calibrations,
                      cut_off_sides, detect_red_square, diagnose_red_square, edge_contact,
                      hand_mask, iou, red_mask, sharpness)

WIN = "calibrate"
GREEN, RED, YELLOW, ORANGE, WHITE, GRAY = ((0, 220, 0), (0, 0, 255), (0, 255, 255), (0, 200, 255),
                                           (230, 230, 230), (160, 160, 160))


# ------------------------------------------------------------------ helpers
def show(frame, lines, boxes=()):
    view = frame.copy()
    for (x, y, w, h), color in boxes:
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
    panel = np.zeros((28 * len(lines) + 10, view.shape[1], 3), np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(panel, text, (10, 25 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imshow(WIN, np.vstack([view, panel]))
    return cv2.waitKey(1) & 0xFF


def status_line(det, reason):
    if det is None:
        return (f"NOT DETECTED: {reason}", RED)
    size = f"crop {det['roi'][2]}x{det['roi'][3]} px, tilt {det['tilt_deg']:.1f} deg"
    return (f"DETECTED  {size}", GREEN) if reason == "OK" else (f"{reason}  ({size})", ORANGE)


def saved_overlap_line(det, saved):
    """How well the current view matches the saved calibration (realign instead of retraining)."""
    if saved is None:
        return ("no saved calibration yet", GRAY)
    if det is None:
        return ("saved border shown in yellow", YELLOW)
    ov = iou(det["outer"], saved["outer"])
    ok = ov >= config.MOVE_IOU_MIN
    return (f"matches saved calibration: {ov:.0%}" + ("" if ok else "  - line the blue box up "
                                                      "with the yellow one"), GREEN if ok else ORANGE)


def wait_for_space(cap, roi, prompt):
    while True:
        frame = read_frame(cap)
        k = show(frame, [(prompt, YELLOW), ("SPACE = capture   q = quit", GRAY)], [(roi, GREEN)])
        if k == ord(" "):
            return True
        if k == ord("q"):
            return False


def capture_crops(cap, roi, n):
    return [crop_roi(read_frame(cap), roi).copy() for _ in range(n)]


def backup_existing():
    if not config.CALIB_PATH.exists():
        return
    hist = config.CALIB_DIR / "history"
    hist.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy(config.CALIB_PATH, hist / f"calibration_{ts}.json")
    if config.BACKGROUND_PATH.exists():
        shutil.copy(config.BACKGROUND_PATH, hist / f"background_{ts}.png")


# ------------------------------------------------------------------ steps
def crop_boxes(det, frame_shape):
    """(model crop, inside-of-border box). In "frame" mode the model sees the whole frame."""
    border = tuple(int(v) for v in det["roi"])
    if config.CROP_MODE == "frame":
        h, w = frame_shape[:2]
        return (0, 0, w, h), border
    return border, border


def step_background(cap, roi, border=None, wait=True):
    """Reference image of the whole crop; brightness stats from the paper inside the border."""
    if wait and not wait_for_space(cap, roi, "Keep the area EMPTY, then press SPACE"):
        return None
    crops = capture_crops(cap, roi, 30)
    bx, by, bw, bh = border or roi
    sub = lambda a: a[by - roi[1]:by - roi[1] + bh, bx - roi[0]:bx - roi[0] + bw]   # noqa: E731
    stack = np.stack([cv2.cvtColor(sub(c), cv2.COLOR_BGR2GRAY) for c in crops]).astype(np.float32)
    bg = np.mean(np.stack(crops).astype(np.float32), axis=0).astype(np.uint8)
    g = cv2.cvtColor(sub(bg), cv2.COLOR_BGR2GRAY)
    means = stack.mean(axis=(1, 2))
    return bg, {"bg_mean": float(g.mean()), "bg_std": float(g.std()),
                "bg_drift": float(means.max() - means.min()),
                "temporal_noise": float(stack.std(axis=0).mean())}


def step_hand(cap, roi, bg):
    if not wait_for_space(cap, roi, "OPEN hand (5 fingers) in the area, hold STILL, press SPACE"):
        return None
    bg_mean = float(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).mean())
    fracs, sharps, shifts, contacts = [], [], [], []
    for c in capture_crops(cap, roi, 15):
        m = hand_mask(c, bg)
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        fracs.append(m.mean())
        contacts.append(edge_contact(m))
        sharps.append(sharpness(g, m))
        if (~m).sum() > 0.05 * m.size:
            shifts.append(float(g[~m].mean()) - bg_mean)
    open_frac, open_sharp = float(np.median(fracs)), float(np.median(sharps))
    contact = {s: float(np.median([c[s] for c in contacts])) for s in SIDES}
    arm_side = max(SIDES, key=lambda s: contact[s])   # the arm enters where contact is largest
    return {"open_hand_frac": open_frac, "open_hand_sharpness": open_sharp,
            "arm_side": arm_side, "open_hand_edges": contact,
            "open_hand_cut_off": cut_off_sides(contact, arm_side),
            "exposure_shift_with_hand": float(np.median(shifts)) if shifts else 0.0,
            "hand_frac_min": max(config.DEFAULT_HAND_FRAC_MIN, 0.25 * open_frac),
            "sharpness_min": 0.35 * open_sharp}


def build_checks(calib, bg):
    lo, hi = config.BG_MEAN_RANGE
    fl, fh = config.OPEN_HAND_FRAC_RANGE
    w, h = calib["roi"][2], calib["roi"][3]
    shift = calib["exposure_shift_with_hand"]
    red = border_red_fraction(bg)
    return [
        ("resolution", (calib["width"], calib["height"]) == (config.FRAME_W, config.FRAME_H),
         f"{calib['width']}x{calib['height']}"),
        ("border tilt", calib["tilt_deg"] <= config.MAX_TILT_DEG, f"{calib['tilt_deg']:.1f} deg"),
        ("crop size", min(w, h) >= config.MIN_ROI_SIDE and 0.5 <= w / h <= 2.0, f"{w}x{h}"),
    ] + ([] if calib.get("crop_mode") == "frame" else [
        ("no tape in crop", red <= config.BAND_RED_MAX, f"{red:.3f}"),
    ]) + [
        ("background brightness", lo <= calib["bg_mean"] <= hi, f"{calib['bg_mean']:.0f} (want {lo}-{hi})"),
        ("background even (shadows/glare)", calib["bg_std"] <= config.BG_STD_MAX, f"std {calib['bg_std']:.1f}"),
        ("brightness stable", calib["bg_drift"] <= config.BG_DRIFT_MAX, f"drift {calib['bg_drift']:.1f}"),
        ("exposure locked", calib.get("exposure") is not None,
         f"value {calib.get('exposure')}" if calib.get("exposure") is not None
         else "auto (brightness will change when a hand enters)"),
    ] + ([] if calib.get("open_hand_frac") is None else [
        ("whole open hand inside the crop", not calib["open_hand_cut_off"],
         f"arm enters from {calib['arm_side']}" + (
             f"; fingers run past {'/'.join(calib['open_hand_cut_off'])} -> camera higher "
             "or bigger border" if calib["open_hand_cut_off"] else "")),
        ("open hand size", fl <= calib["open_hand_frac"] <= fh,
         f"{calib['open_hand_frac']:.2f} of crop (want {fl}-{fh})"),
        ("exposure steady when hand enters", abs(shift) <= config.EXPOSURE_SHIFT_MAX,
         f"shift {shift:+.1f}"),
    ])


def save_and_report(cap, calib, bg, old):
    checks = build_checks(calib, bg)
    calib["checks"] = {name: {"ok": ok, "value": val} for name, ok, val in checks}
    warns = compare_calibrations(old, calib) if old else []
    backup_existing()
    config.CALIB_DIR.mkdir(parents=True, exist_ok=True)
    config.CALIB_PATH.write_text(json.dumps(calib, indent=2))
    cv2.imwrite(str(config.BACKGROUND_PATH), bg)
    snap = read_frame(cap)
    x, y, w, h = calib.get("border_roi", calib["roi"])
    cv2.rectangle(snap, (x, y), (x + w, y + h), GREEN, 2)
    cv2.imwrite(str(config.CALIB_DIR / "snapshot.png"), snap)

    print("\nCALIBRATION REPORT")
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'WARN'}] {name}: {val}")
    for w_ in warns:
        print(f"  [CHANGED] {w_}")
    print(f"\n  thresholds: hand_frac_min {calib['hand_frac_min']:.3f}, "
          f"sharpness_min {calib['sharpness_min']:.1f}")
    print(f"  saved {config.CALIB_PATH} (previous one in {config.CALIB_DIR / 'history'})")
    ok = all(ok for _, ok, _ in checks)
    print("RESULT:", "PASS" if ok and not warns else "WARN (read the lines above)")


# ------------------------------------------------------------------ modes
def probe(max_index=5, verbose=True):
    """Only cameras that see the border get a snapshot (no photos of the room/you)."""
    found = []
    for i in range(max_index):
        try:
            cap, info = open_camera(i, warmup=15)
            frame = read_frame(cap)
            cap.release()
        except RuntimeError:
            if verbose:
                print(f"[{i}] not available")
            continue
        det, reason = diagnose_red_square(frame)
        if det:
            found.append(i)
            out = config.CALIB_DIR / "probe"
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"camera_{i}.png"), frame)
        if verbose:
            print(f"[{i}] {info['width']}x{info['height']}  border: "
                  f"{'YES' if det else 'no'} ({reason})")
    return found


def live_view(cam, max_index=5):
    """Keys: n = next camera, r = tape overlay, s = snapshot, q = quit. If a calibration is
    saved, its border is drawn in yellow so the camera can be put back where it was."""
    saved = config.load_calibration()
    same = saved is not None and saved["camera_index"] == cam
    cap, info = open_from_calibration(saved, cam) if same else open_camera(cam, warmup=10)
    show_red, n, fps, t0 = False, 0, 0.0, time.time()
    det, reason = None, ""
    while True:
        frame = read_frame(cap)
        det, reason = diagnose_red_square(frame)
        view = frame.copy()
        if show_red:
            m = red_mask(frame) > 0
            view[m] = (0.3 * view[m] + 0.7 * np.array([255, 0, 255])).astype(np.uint8)
        if det:
            mv = model_view(crop_roi(frame, crop_boxes(det, frame.shape)[0]))
            view[10:10 + mv.shape[0], view.shape[1] - mv.shape[1] - 10:view.shape[1] - 10] = mv
        n += 1
        if time.time() - t0 >= 1.0:
            fps, n, t0 = n / (time.time() - t0), 0, time.time()
        lines = [(f"camera {cam}  {info['width']}x{info['height']}  {fps:.0f} fps   "
                  "blue = border, green = inside, yellow = saved", WHITE),
                 status_line(det, reason), saved_overlap_line(det, saved),
                 ("n next camera | r show tape pixels | s snapshot | q quit", GRAY)]
        boxes = []
        if saved:
            boxes.append((tuple(saved["outer"]), YELLOW))
        if det:
            boxes += [(det["outer"], (255, 0, 0)), (det["roi"], GREEN)]
        k = show(view, lines, boxes)
        if k == ord("r"):
            show_red = not show_red
        elif k == ord("s"):
            config.CALIB_DIR.mkdir(parents=True, exist_ok=True)
            p = config.CALIB_DIR / f"view_camera_{cam}.png"
            cv2.imwrite(str(p), frame)
            print("saved", p)
        elif k == ord("n"):
            cap.release()
            for _ in range(max_index):
                cam = (cam + 1) % max_index
                try:
                    cap, info = open_camera(cam, warmup=10)
                    break
                except RuntimeError:
                    continue
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"camera {cam}: {reason}" + (f". Next: python -m src.calibrate --camera {cam}"
                                       if det and reason == "OK" else ""))


def check():
    calib = config.load_calibration()
    if calib is None:
        raise SystemExit("No calibration yet. Run: python -m src.calibrate")
    cap, _ = open_from_calibration(calib)
    frame = read_frame(cap)
    cap.release()
    print(f"calibration from {calib.get('created')} (make sure the area is EMPTY)")
    issues = calibration_drift(frame, calib)
    for i in issues:
        print("  -", i)
    print("RESULT:", "still valid" if not issues else "recalibrate (see above)")
    return not issues


def lock_exposure_for(cap, roi):
    """Pick and lock the exposure (area must be EMPTY)."""
    print("choosing exposure (keep the area empty)...")
    value, level = choose_exposure(cap, roi)
    if value is None:
        print("  this camera/driver won't lock exposure; staying on auto")
    else:
        print(f"  locked exposure {value} -> empty paper brightness {level:.0f}")
    return value


def relight(auto_exposure):
    old = config.load_calibration()
    if old is None:
        raise SystemExit("No calibration yet. Run: python -m src.calibrate")
    cap, info = open_camera(old["camera_index"], old["width"], old["height"])
    det = detect_red_square(read_frame(cap))
    if det is None or iou(det["outer"], old["outer"]) < config.MOVE_IOU_MIN:
        cap.release()
        raise SystemExit("The border moved or isn't visible: relight only fixes lighting. "
                         "Realign with --view or run a full calibration.")
    roi = tuple(old["roi"])
    border = tuple(old.get("border_roi", old["roi"]))
    exposure = None if auto_exposure else lock_exposure_for(cap, border)
    info.update(exposure=exposure, exposure_locked=exposure is not None)
    bgr = step_background(cap, roi, border)
    if bgr is None:
        return
    bg, bg_stats = bgr
    hand = step_hand(cap, roi, bg)
    if hand is None:
        return
    calib = {**old, **bg_stats, **hand, "exposure": info["exposure"],
             "exposure_locked": info["exposure_locked"],
             "created": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "relight"}
    save_and_report(cap, calib, bg, old)
    cap.release()
    cv2.destroyAllWindows()


QUICK_HAND = {"open_hand_frac": None, "open_hand_sharpness": None, "arm_side": None,
              "open_hand_edges": None, "open_hand_cut_off": [], "exposure_shift_with_hand": 0.0,
              "hand_frac_min": config.DEFAULT_HAND_FRAC_MIN,
              "sharpness_min": config.DEFAULT_SHARPNESS_MIN}


def quick_calibration(cam, auto_exposure):
    """No windows, no hand step: border + locked exposure + empty background. Default hand
    thresholds (run a full calibration later to measure them). The area must be EMPTY."""
    old = config.load_calibration()
    if cam is None:
        cam = old["camera_index"] if old else 0
    cap, info = open_camera(cam)
    frame = read_frame(cap)
    det, reason = diagnose_red_square(frame)
    if det is None:
        cap.release()
        raise SystemExit(f"border not detected: {reason}")
    roi, border = crop_boxes(det, frame.shape)
    exposure = None if auto_exposure else lock_exposure_for(cap, border)
    bg, bg_stats = step_background(cap, roi, border, wait=False)
    if bg_stats["bg_std"] > config.BG_STD_MAX:
        cap.release()
        raise SystemExit(f"area doesn't look empty/even (std {bg_stats['bg_std']:.0f}); clear it and rerun")
    calib = {"camera_index": cam, "width": info["width"], "height": info["height"],
             "exposure": exposure, "exposure_locked": exposure is not None,
             "roi": list(roi), "border_roi": list(border), "crop_mode": config.CROP_MODE,
             "outer": list(det["outer"]), "tilt_deg": det["tilt_deg"],
             **bg_stats, **QUICK_HAND,
             "created": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "quick"}
    save_and_report(cap, calib, bg, old)
    cap.release()


def full_calibration(cam, auto_exposure):
    old = config.load_calibration()
    if cam is None:
        found = probe(verbose=False)
        if len(found) != 1:
            raise SystemExit(f"cameras seeing a border: {found}. "
                             "Use --view (n = next camera) to find it, then --camera N.")
        cam = found[0]
    print(f"using camera {cam}")
    cap, info = open_camera(cam)

    roi = outer = None
    tilt = 0.0
    while True:
        frame = read_frame(cap)
        det, reason = diagnose_red_square(frame)
        lines = [("STEP 1: area must be EMPTY.  c = accept   m = draw box   q = quit", YELLOW),
                 status_line(det, reason), saved_overlap_line(det, old)]
        boxes = ([(tuple(old["outer"]), YELLOW)] if old else []) + \
                ([(det["outer"], (255, 0, 0)), (det["roi"], GREEN)] if det else [])
        k = show(frame, lines, boxes)
        if k == ord("c") and det:
            (roi, border), outer, tilt = crop_boxes(det, frame.shape), det["outer"], det["tilt_deg"]
            break
        if k == ord("m"):
            r = cv2.selectROI(WIN, frame, showCrosshair=False)
            if r[2] > 0 and r[3] > 0:
                border = outer = tuple(int(v) for v in r)
                roi, _ = crop_boxes({"roi": border}, frame.shape)
                break
        if k == ord("q"):
            return

    exposure = None if auto_exposure else lock_exposure_for(cap, border)
    bgr = step_background(cap, roi, border)
    if bgr is None:
        return
    bg, bg_stats = bgr
    hand = step_hand(cap, roi, bg)
    if hand is None:
        return
    calib = {"camera_index": cam, "width": info["width"], "height": info["height"],
             "exposure": exposure, "exposure_locked": exposure is not None,
             "roi": list(roi), "border_roi": list(border), "crop_mode": config.CROP_MODE,
             "outer": list(outer),
             "tilt_deg": tilt, **bg_stats, **hand,
             "created": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": "full"}
    save_and_report(cap, calib, bg, old)
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--probe", action="store_true")
    g.add_argument("--view", action="store_true", help="live camera view for setup")
    g.add_argument("--check", action="store_true", help="is the saved calibration still valid?")
    g.add_argument("--relight", action="store_true", help="lighting changed, camera did not")
    g.add_argument("--quick", action="store_true", help="no windows/hand step (area empty)")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--auto-exposure", action="store_true",
                    help="don't lock exposure (not recommended: brightness shifts with the hand)")
    args = ap.parse_args()

    if args.probe:
        probe()
    elif args.view:
        saved = config.load_calibration()
        live_view(args.camera if args.camera is not None else
                  (saved["camera_index"] if saved else 0))
    elif args.check:
        check()
    elif args.relight:
        relight(args.auto_exposure)
    elif args.quick:
        quick_calibration(args.camera, args.auto_exposure)
    else:
        full_calibration(args.camera, args.auto_exposure)


if __name__ == "__main__":
    main()
