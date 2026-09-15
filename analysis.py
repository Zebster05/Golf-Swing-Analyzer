"""Swing biomechanics from MediaPipe pose frames. Numpy only."""

import numpy as np

LEAD_IDX = {
    "right": {
        "shoulder": 11,
        "elbow": 13,
        "wrist": 15,
        "pinky": 17,
        "index": 19,
        "hip": 23,
    },
    "left": {
        "shoulder": 12,
        "elbow": 14,
        "wrist": 16,
        "pinky": 18,
        "index": 20,
        "hip": 24,
    },
}

NA = "n/a"


def calculate_angle(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _xy(lm):
    if hasattr(lm, "x"):
        return (float(lm.x), float(lm.y))
    return (float(lm[0]), float(lm[1]))


def _xyz(lm):
    if hasattr(lm, "x"):
        return [float(lm.x), float(lm.y), float(lm.z)]
    return [float(lm[0]), float(lm[1]), float(lm[2]) if len(lm) > 2 else 0.0]


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def hands_mid(frame):
    return _mid(frame["lw"], frame["rw"])


def extract_frame_pose(landmarks, frame_idx, handedness="right"):
    handedness = "left" if handedness == "left" else "right"
    lead = LEAD_IDX[handedness]
    ls, rs = _xy(landmarks[11]), _xy(landmarks[12])
    le, re = _xy(landmarks[13]), _xy(landmarks[14])
    lw, rw = _xy(landmarks[15]), _xy(landmarks[16])
    nose = _xy(landmarks[0])
    left_ear, right_ear = _xy(landmarks[7]), _xy(landmarks[8])
    return {
        "frame": int(frame_idx),
        "left_arm_angle": calculate_angle(
            _xyz(landmarks[11]), _xyz(landmarks[13]), _xyz(landmarks[15])
        ),
        "right_arm_angle": calculate_angle(
            _xyz(landmarks[12]), _xyz(landmarks[14]), _xyz(landmarks[16])
        ),
        "head_x": (nose[0] + left_ear[0] + right_ear[0]) / 3.0,
        "head_y": (nose[1] + left_ear[1] + right_ear[1]) / 3.0,
        "ls": ls,
        "rs": rs,
        "le": le,
        "re": re,
        "lw": lw,
        "rw": rw,
        "lh": _xy(landmarks[23]),
        "rh": _xy(landmarks[24]),
        "lf": _xy(landmarks[31]),
        "rf": _xy(landmarks[32]),
        "lead_index": _xy(landmarks[lead["index"]]),
        "lead_pinky": _xy(landmarks[lead["pinky"]]),
    }


def _smooth(values, window=3):
    if len(values) < window:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _empty_phases(reason, address=None):
    return {
        "ok": False,
        "reason": reason,
        "address": address,
        "top": None,
        "mid_down": None,
        "impact": None,
    }


def _guess_address(ys, top=None):
    if top is None:
        end = max(3, int(0.15 * len(ys)))
    else:
        end = max(1, min(top, max(2, int(0.25 * top) if top > 0 else 2)))
    return int(np.argmax(ys[: end + 1]))


def _first_backswing_peak(ys):
    """First hands-high peak with a rise then a drop. Ignores a higher finish."""
    n = len(ys)
    start = max(3, int(0.05 * n))
    end = n - max(4, int(0.10 * n))
    if end <= start + 4:
        start, end = 2, n - 2
    order = 2
    look = max(8, n // 5)
    for i in range(max(start, order), min(end, n - order)):
        if not (
            ys[i] <= ys[i - 1]
            and ys[i] <= ys[i + 1]
            and ys[i] <= ys[i - order]
            and ys[i] <= ys[i + order]
        ):
            continue
        if ys[start:i].size == 0 or ys[start:i].max() - ys[i] < 0.04:
            continue
        after = ys[i : min(n, i + look)]
        if after.size == 0 or after.max() - ys[i] < 0.04:
            continue
        return i
    return None


def detect_phases(frames):
    n = len(frames)
    if n < 12:
        return _empty_phases("too_few_frames")

    ys = np.array([hands_mid(f)[1] for f in frames], dtype=float)
    ys_s = _smooth(ys)
    address_guess = _guess_address(ys_s)
    top = _first_backswing_peak(ys_s)
    if top is None:
        return _empty_phases("top_not_locked", address_guess)

    impact = top + int(np.argmax(ys_s[top:]))
    if impact - top < 3:
        return _empty_phases("no_downswing", address_guess)

    address = _guess_address(ys_s, top)
    if ys_s[address] - ys_s[top] < 0.04:
        return _empty_phases("no_hand_rise", address)

    mid_down = top + max(1, (impact - top) // 2)
    return {
        "ok": True,
        "reason": "ok",
        "address": address,
        "top": top,
        "mid_down": mid_down,
        "impact": impact,
    }


def _shoulder_angle(frame):
    ls, rs = frame["ls"], frame["rs"]
    return float(np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0])))


def _triangle_width(frame):
    return _dist(_mid(frame["ls"], frame["rs"]), hands_mid(frame))


def _trail_points(frame, handedness):
    if handedness == "left":
        return frame["le"], frame["lh"]
    return frame["re"], frame["rh"]


def _lead_points(frame, handedness):
    if handedness == "left":
        return frame["re"], frame["rw"]
    return frame["le"], frame["lw"]


def analyze_connection(frames, phases, handedness="right"):
    handedness = "left" if handedness == "left" else "right"
    if phases.get("address") is None or phases.get("top") is None:
        return {
            "connection": NA,
            "triangle_at_top": NA,
            "trail_elbow": NA,
            "skip_reason": "top_not_locked",
        }

    addr = frames[phases["address"]]
    top = frames[phases["top"]]
    y_addr = hands_mid(addr)[1]
    rise_total = y_addr - hands_mid(top)[1]
    target_rise = 0.35 * max(rise_total, 1e-6)
    take_i = phases["top"]
    for i in range(phases["address"], phases["top"] + 1):
        if y_addr - hands_mid(frames[i])[1] >= target_rise:
            take_i = i
            break
    take = frames[take_i]

    width_addr = _triangle_width(addr)
    width_top = _triangle_width(top)
    triangle = "collapsed" if width_addr > 1e-6 and (width_top / width_addr) < 0.78 else "intact"

    rot = abs(_shoulder_angle(take) - _shoulder_angle(addr))
    sh_addr = _dist(addr["ls"], addr["rs"])
    sh_take = _dist(take["ls"], take["rs"])
    turned = rot >= 8.0 or abs(sh_take - sh_addr) >= 0.02
    connection = "early_arm_lift" if not turned else "connected"

    te_addr, th_addr = _trail_points(addr, handedness)
    te_top, th_top = _trail_points(top, handedness)
    d_addr = _dist(te_addr, th_addr)
    d_top = _dist(te_top, th_top)
    trail_elbow = "flying" if d_top > d_addr * 1.28 or d_top > 0.22 else "tucked"

    return {
        "connection": connection,
        "triangle_at_top": triangle,
        "trail_elbow": trail_elbow,
        "skip_reason": None,
    }


def _closest_x_at_y(frames, y_target):
    best = frames[0]
    best_d = abs(hands_mid(best)[1] - y_target)
    for f in frames[1:]:
        d = abs(hands_mid(f)[1] - y_target)
        if d < best_d:
            best, best_d = f, d
    return best


def analyze_plane(frames, phases, view, handedness="right"):
    """Plane / OTT from green (backswing) vs orange (downswing) at the same height."""
    view = (view or "dtl").lower().replace("-", "_").replace(" ", "_")
    if view == "45":
        view = "45"
    blank = {
        "plane": NA,
        "ott": None,
        "hand_path": NA,
        "plane_a": None,
        "plane_b": None,
    }
    if view != "dtl":
        return {**blank, "skip_reason": "view_not_dtl"}
    if (
        phases.get("address") is None
        or phases.get("top") is None
        or phases.get("mid_down") is None
        or phases.get("impact") is None
    ):
        return {**blank, "skip_reason": "top_not_locked"}

    mid_f = frames[phases["mid_down"]]
    y_target = hands_mid(mid_f)[1]
    up_f = _closest_x_at_y(frames[phases["address"] : phases["top"] + 1], y_target)
    down_f = _closest_x_at_y(frames[phases["top"] : phases["impact"] + 1], y_target)
    x_up = hands_mid(up_f)[0]
    x_down = hands_mid(down_f)[0]
    spine = (mid_f["ls"][0] + mid_f["rs"][0]) / 2.0
    if abs(x_down - spine) > abs(x_up - spine) + 0.015:
        hand_path = "outside_in"
        plane = "steep"
    elif abs(x_down - spine) + 0.015 < abs(x_up - spine):
        hand_path = "inside_out"
        plane = "shallow"
    else:
        hand_path = "on"
        plane = "on_plane"

    return {
        "plane": plane,
        "ott": hand_path == "outside_in",
        "hand_path": hand_path,
        "plane_a": None,
        "plane_b": None,
        "skip_reason": None,
    }


def analyze_wrist(frames, phases, handedness="right"):
    if phases.get("top") is None:
        return {
            "lead_wrist_at_top": NA,
            "wrist_confidence": "none",
            "skip_reason": "top_not_locked",
        }

    handedness = "left" if handedness == "left" else "right"
    lo = max(0, phases["top"] - 2)
    hi = min(len(frames), phases["top"] + 3)
    votes = []
    for f in frames[lo:hi]:
        elbow, wrist = _lead_points(f, handedness)
        kn = _mid(f["lead_index"], f["lead_pinky"])
        head = (f["head_x"], f["head_y"])
        fore = np.array(wrist) - np.array(elbow)
        ext = np.array(wrist) + fore
        d_kn = _dist(kn, head)
        d_ext = _dist(ext, head)
        bend = calculate_angle(elbow, wrist, kn)
        if d_kn + 0.012 < d_ext and bend < 168:
            votes.append("cupped")
        elif d_kn > d_ext + 0.012 and bend < 168:
            votes.append("bowed")
        else:
            votes.append("flat")

    if not votes:
        return {
            "lead_wrist_at_top": NA,
            "wrist_confidence": "none",
            "skip_reason": "wrist_unreadable",
        }
    label = max(set(votes), key=votes.count)
    return {
        "lead_wrist_at_top": label,
        "wrist_confidence": "low",
        "skip_reason": None,
    }


def _head_sway_label(frames):
    xs = [f["head_x"] for f in frames]
    if len(xs) < 5:
        return NA, 0.0
    std = float(np.std(xs))
    if std > 0.08:
        return "excessive", std
    if std > 0.04:
        return "moderate", std
    return "stable", std


def _arm_at(frame, handedness, which):
    if which == "lead":
        return frame["right_arm_angle"] if handedness == "left" else frame["left_arm_angle"]
    return frame["left_arm_angle"] if handedness == "left" else frame["right_arm_angle"]


def _observe(observed, not_scored, field, value, skip_reason=None, missing=None):
    if skip_reason:
        not_scored.append({"field": field, "reason": skip_reason})
        return
    if missing is not None and (value is None or value == missing):
        not_scored.append({"field": field, "reason": "unreadable"})
        return
    observed[field] = value


def _notes(view_key, phases, not_scored):
    parts = []
    if phases.get("address") is not None:
        parts.append("Address frame available.")
    if phases.get("ok"):
        parts.append("Address, top, and impact locked.")
    else:
        parts.append(f"Phases incomplete ({phases.get('reason', 'unknown')}).")
    if view_key != "dtl":
        parts.append("Camera is not DTL; plane and OTT are not scored.")
    if any(item["field"] == "ott" for item in not_scored):
        parts.append("Do not infer over-the-top from missing plane data.")
    return " ".join(parts)


def build_report(frames, phases, view="dtl", handedness="right"):
    view_key = (view or "dtl").lower().replace("-", "_").replace(" ", "_")
    if view_key in ("faceon", "face"):
        view_key = "face_on"
    handedness = "left" if handedness == "left" else "right"

    observed = {}
    not_scored = []

    sway, sway_std = _head_sway_label(frames)
    _observe(observed, not_scored, "head_sway", sway, missing=NA)

    if phases.get("address") is not None:
        observed["address_seen"] = True
    else:
        not_scored.append({"field": "address_seen", "reason": "address_not_locked"})

    conn = analyze_connection(frames, phases, handedness)
    for field in ("connection", "triangle_at_top", "trail_elbow"):
        _observe(observed, not_scored, field, conn[field], conn.get("skip_reason"), missing=NA)

    plane = analyze_plane(frames, phases, view_key, handedness)
    _observe(observed, not_scored, "plane", plane["plane"], plane.get("skip_reason"), missing=NA)
    _observe(observed, not_scored, "hand_path", plane["hand_path"], plane.get("skip_reason"), missing=NA)
    if plane.get("skip_reason"):
        not_scored.append({"field": "ott", "reason": plane["skip_reason"]})
    else:
        observed["ott"] = bool(plane["ott"])

    wrist = analyze_wrist(frames, phases, handedness)
    _observe(
        observed,
        not_scored,
        "lead_wrist_at_top",
        wrist["lead_wrist_at_top"],
        wrist.get("skip_reason"),
        missing=NA,
    )
    if wrist.get("skip_reason"):
        not_scored.append({"field": "wrist_confidence", "reason": wrist["skip_reason"]})
    else:
        observed["wrist_confidence"] = wrist["wrist_confidence"]

    lead_top = trail_top = None
    if phases.get("top") is not None:
        top_f = frames[phases["top"]]
        lead_top = round(_arm_at(top_f, handedness, "lead"), 1)
        trail_top = round(_arm_at(top_f, handedness, "trail"), 1)
        observed["lead_arm_at_top_deg"] = lead_top
        observed["trail_arm_at_top_deg"] = trail_top
    else:
        not_scored.append({"field": "lead_arm_at_top_deg", "reason": "top_not_locked"})
        not_scored.append({"field": "trail_arm_at_top_deg", "reason": "top_not_locked"})

    # De-dupe not_scored (plane skip adds plane + hand_path + ott with same reason)
    seen = set()
    unique = []
    for item in not_scored:
        key = (item["field"], item["reason"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    not_scored = unique

    severity = []
    if observed.get("ott") is True:
        severity.append("CRITICAL: Over-the-top (hand path outside-in)")
    if observed.get("connection") == "early_arm_lift":
        severity.append("CRITICAL: Arms disconnect from body (early arm lift)")
    if observed.get("triangle_at_top") == "collapsed":
        severity.append("MODERATE: Arm triangle collapsed at top")
    if observed.get("lead_wrist_at_top") == "cupped":
        severity.append("MODERATE: Lead wrist cupped at top")
    if observed.get("head_sway") == "excessive":
        severity.append("CRITICAL: Excessive lateral sway")
    elif observed.get("head_sway") == "moderate":
        severity.append("MODERATE: Minor head sway")
    if lead_top is not None and lead_top < 135:
        severity.append("CRITICAL: Lead arm collapse at top")
    elif lead_top is not None and lead_top < 155:
        severity.append("MODERATE: Lead arm soft at top")

    notes = _notes(view_key, phases, not_scored)

    return {
        "view": view_key,
        "handedness": handedness,
        "phases_ok": phases["ok"],
        "phases": phases,
        "notes": notes,
        "observed": observed,
        "not_scored": not_scored,
        "lead_arm_at_top_deg": lead_top,
        "trail_arm_at_top_deg": trail_top,
        "head_sway": observed.get("head_sway", NA),
        "head_sway_std": round(sway_std, 4),
        "connection": observed.get("connection", NA),
        "triangle_at_top": observed.get("triangle_at_top", NA),
        "trail_elbow": observed.get("trail_elbow", NA),
        "plane": observed.get("plane", NA),
        "ott": observed.get("ott"),
        "hand_path": observed.get("hand_path", NA),
        "plane_a": plane["plane_a"],
        "plane_b": plane["plane_b"],
        "lead_wrist_at_top": observed.get("lead_wrist_at_top", NA),
        "wrist_confidence": observed.get("wrist_confidence", "none"),
        "severity": severity,
    }


_OVERLAY_NONE = {
    "flaw_id": None,
    "focus": "pose",
    "rank": None,
    "skeleton": True,
    "head": False,
    "triangle": False,
    "plane": False,
    "handpath": False,
}

_OVERLAY_BY_FLAW = {
    "ott": {
        "focus": "over-the-top",
        "skeleton": False,
        "head": False,
        "triangle": False,
        "plane": False,
        "handpath": True,
    },
    "steep": {
        "focus": "steep plane",
        "skeleton": False,
        "head": False,
        "triangle": False,
        "plane": False,
        "handpath": True,
    },
    "shallow": {
        "focus": "shallow plane",
        "skeleton": False,
        "head": False,
        "triangle": False,
        "plane": False,
        "handpath": True,
    },
    "head": {
        "focus": "head sway",
        "skeleton": True,
        "head": True,
        "triangle": False,
        "plane": False,
        "handpath": False,
    },
    "disconnect": {
        "focus": "early arm lift",
        "skeleton": True,
        "head": False,
        "triangle": True,
        "plane": False,
        "handpath": False,
    },
    "triangle": {
        "focus": "collapsed triangle",
        "skeleton": True,
        "head": False,
        "triangle": True,
        "plane": False,
        "handpath": False,
    },
    "lead_arm": {
        "focus": "lead arm",
        "skeleton": True,
        "head": False,
        "triangle": False,
        "plane": False,
        "handpath": False,
    },
    "wrist": {
        "focus": "lead wrist",
        "skeleton": True,
        "head": False,
        "triangle": False,
        "plane": False,
        "handpath": False,
    },
}


def _overlay_candidates(observed):
    """CRITICAL first (definition order), then MODERATE. Not Gemini severity append order."""
    candidates = []
    lead_top = observed.get("lead_arm_at_top_deg")

    if observed.get("ott") is True:
        candidates.append(("ott", "CRITICAL"))
    if observed.get("connection") == "early_arm_lift":
        candidates.append(("disconnect", "CRITICAL"))
    if observed.get("head_sway") == "excessive":
        candidates.append(("head", "CRITICAL"))
    if lead_top is not None and lead_top < 135:
        candidates.append(("lead_arm", "CRITICAL"))

    if observed.get("triangle_at_top") == "collapsed":
        candidates.append(("triangle", "MODERATE"))
    if observed.get("ott") is not True:
        if observed.get("plane") == "steep":
            candidates.append(("steep", "MODERATE"))
        elif observed.get("plane") == "shallow":
            candidates.append(("shallow", "MODERATE"))
    if observed.get("lead_wrist_at_top") == "cupped":
        candidates.append(("wrist", "MODERATE"))
    if observed.get("head_sway") == "moderate":
        candidates.append(("head", "MODERATE"))
    if lead_top is not None and 135 <= lead_top < 155:
        candidates.append(("lead_arm", "MODERATE"))
    return candidates


def overlay_layers(report):
    observed = report.get("observed") or {}
    candidates = _overlay_candidates(observed)
    chosen = next((item for item in candidates if item[1] == "CRITICAL"), None)
    if chosen is None:
        chosen = next((item for item in candidates if item[1] == "MODERATE"), None)
    if chosen is None:
        return dict(_OVERLAY_NONE)
    flaw_id, rank = chosen
    out = dict(_OVERLAY_BY_FLAW[flaw_id])
    out["flaw_id"] = flaw_id
    out["rank"] = rank
    return out


def coaching_payload(report):
    return {
        "view": report["view"],
        "handedness": report["handedness"],
        "phases_ok": report["phases_ok"],
        "notes": report.get("notes", ""),
        "observed": report.get("observed", {}),
        "not_scored": report.get("not_scored", []),
    }


def build_insights(report):
    items = []
    observed = report.get("observed") or {}
    not_scored = report.get("not_scored") or []

    if report.get("notes"):
        items.append(("REVIEW SCOPE", report["notes"], "info"))

    if observed.get("head_sway") == "excessive":
        items.append(
            ("HEAD STABILITY", "Excessive lateral head sway. Keep the head quieter.", "warning")
        )
    elif observed.get("head_sway") == "moderate":
        items.append(
            ("HEAD STABILITY", "Some lateral head movement. Tighten it for consistency.", "warning")
        )
    elif "head_sway" in observed:
        items.append(("HEAD STABILITY", "Head stays stable through the swing.", "success"))

    lead = observed.get("lead_arm_at_top_deg")
    if lead is not None and lead < 150:
        items.append(
            (
                "LEAD ARM AT TOP",
                f"Lead arm is {lead:.0f}° at the top. Extend it more to keep width.",
                "warning",
            )
        )
    elif lead is not None:
        items.append(
            ("LEAD ARM AT TOP", f"Lead arm is {lead:.0f}° at the top. Solid extension.", "success")
        )

    if observed.get("connection") == "early_arm_lift":
        items.append(
            (
                "ARM STRUCTURE",
                "Hands lift before the torso turns. Keep the triangle and turn the chest.",
                "warning",
            )
        )
    elif observed.get("connection") == "connected":
        items.append(
            ("ARM STRUCTURE", "Arms stay connected to the body on the way up.", "success")
        )

    if observed.get("triangle_at_top") == "collapsed":
        items.append(
            (
                "TRIANGLE AT TOP",
                "The arm triangle narrows at the top. Keep width between hands and chest.",
                "warning",
            )
        )

    if report.get("view") == "dtl":
        if observed.get("ott") is True:
            items.append(
                (
                    "OVER THE TOP",
                    "Downswing hand path is outside the backswing. Drop the trail arm in.",
                    "warning",
                )
            )
        elif observed.get("plane") == "shallow":
            items.append(
                (
                    "SWING PLANE",
                    "Downswing hand path is inside the backswing (shallow / inside-out).",
                    "warning",
                )
            )
        elif observed.get("plane") == "on_plane":
            items.append(
                ("SWING PLANE", "Downswing hand path matches the backswing.", "success")
            )

    wrist = observed.get("lead_wrist_at_top")
    if wrist == "cupped":
        items.append(
            (
                "LEAD WRIST AT TOP",
                "Lead wrist looks cupped (broken inward). Feel a flatter wrist. Low confidence.",
                "warning",
            )
        )
    elif wrist == "bowed":
        items.append(
            (
                "LEAD WRIST AT TOP",
                "Lead wrist looks bowed. Fine if intentional; check the face. Low confidence.",
                "warning",
            )
        )
    elif wrist == "flat":
        items.append(
            ("LEAD WRIST AT TOP", "Lead wrist looks flat at the top. Low confidence.", "success")
        )

    if not_scored:
        skipped = ", ".join(f"{item['field']} ({item['reason']})" for item in not_scored)
        items.append(("NOT SCORED", skipped, "info"))

    return items
