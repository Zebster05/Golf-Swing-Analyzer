import json
import unittest

from analysis import (
    analyze_connection,
    analyze_plane,
    analyze_wrist,
    build_report,
    coaching_payload,
    detect_phases,
    extract_frame_pose,
    hands_mid,
    overlay_layers,
)


class _Lm:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _landmarks(overrides=None):
    pts = [(_Lm(0.5, 0.2))] * 33
    defaults = {
        0: (0.50, 0.18),
        7: (0.48, 0.18),
        8: (0.52, 0.18),
        11: (0.42, 0.32),
        12: (0.58, 0.32),
        13: (0.40, 0.50),
        14: (0.62, 0.48),
        15: (0.48, 0.72),
        16: (0.52, 0.72),
        17: (0.47, 0.75),
        19: (0.49, 0.75),
        18: (0.53, 0.75),
        20: (0.51, 0.75),
        23: (0.44, 0.62),
        24: (0.56, 0.62),
    }
    if overrides:
        defaults.update(overrides)
    for i, (x, y) in defaults.items():
        pts[i] = _Lm(x, y)
    return pts


def _frame(i, hands_y, hands_x=0.50, extra=None):
    overrides = {
        15: (hands_x - 0.02, hands_y),
        16: (hands_x + 0.02, hands_y),
        17: (hands_x - 0.03, hands_y + 0.02),
        19: (hands_x - 0.01, hands_y + 0.02),
    }
    if extra:
        overrides.update(extra)
    return extract_frame_pose(_landmarks(overrides), i, "right")


def _arc_swing(n=30, down_x=0.50):
    frames = []
    top = 14
    for i in range(n):
        if i <= top:
            t = i / top
            y = 0.78 - 0.50 * t
            x = 0.50
        else:
            t = (i - top) / (n - 1 - top)
            y = 0.28 + 0.52 * t
            x = 0.50 + (down_x - 0.50) * t
        frames.append(_frame(i + 1, y, x))
    return frames


def _tilt_shoulders(frames):
    """Mark torso turn so connection is not early_arm_lift."""
    for i, f in enumerate(frames):
        if i > 3:
            f["ls"] = (0.42, 0.36)
            f["rs"] = (0.58, 0.28)
    return frames


def _sway_heads(frames):
    for i, f in enumerate(frames):
        f["head_x"] = 0.35 if i % 2 == 0 else 0.55
    return frames


class DetectPhasesTests(unittest.TestCase):
    def test_finds_address_top_impact(self):
        phases = detect_phases(_arc_swing())
        self.assertTrue(phases["ok"])
        self.assertLess(phases["address"], phases["top"])
        self.assertLess(phases["top"], phases["mid_down"])
        self.assertLess(phases["mid_down"], phases["impact"])
        self.assertLess(hands_mid(_arc_swing()[phases["top"]])[1], 0.40)

    def test_flat_path_fails(self):
        frames = [_frame(i + 1, 0.70) for i in range(20)]
        phases = detect_phases(frames)
        self.assertFalse(phases["ok"])
        self.assertIsNotNone(phases["address"])

    def test_too_short_fails(self):
        self.assertFalse(detect_phases(_arc_swing(n=8))["ok"])

    def test_high_finish_uses_first_peak(self):
        frames = []
        top, impact, n = 14, 24, 40
        for i in range(n):
            if i <= top:
                y = 0.78 - 0.50 * (i / top)
            elif i <= impact:
                y = 0.28 + 0.50 * ((i - top) / (impact - top))
            else:
                y = 0.78 - 0.62 * ((i - impact) / (n - 1 - impact))
            frames.append(_frame(i + 1, y))
        phases = detect_phases(frames)
        self.assertTrue(phases["ok"])
        self.assertLess(phases["top"], 20)
        self.assertGreater(
            hands_mid(frames[phases["top"]])[1], hands_mid(frames[-1])[1]
        )


class ConnectionTests(unittest.TestCase):
    def test_early_arm_lift(self):
        frames = []
        for i in range(24):
            if i < 4:
                y, sh_y = 0.78, 0.32
            elif i < 10:
                y, sh_y = 0.40, 0.32
            else:
                t = (i - 10) / 13
                y = 0.28 + 0.50 * t
                sh_y = 0.32
            frames.append(
                _frame(i + 1, y, extra={11: (0.42, sh_y), 12: (0.58, sh_y)})
            )
        phases = detect_phases(frames)
        self.assertTrue(phases["ok"])
        conn = analyze_connection(frames, phases, "right")
        self.assertEqual(conn["connection"], "early_arm_lift")

    def test_collapsed_triangle(self):
        frames = _arc_swing()
        phases = detect_phases(frames)
        top = frames[phases["top"]]
        mid = hands_mid(top)
        top["ls"] = (mid[0] - 0.01, mid[1] - 0.01)
        top["rs"] = (mid[0] + 0.01, mid[1] - 0.01)
        conn = analyze_connection(frames, phases, "right")
        self.assertEqual(conn["triangle_at_top"], "collapsed")


class PlaneTests(unittest.TestCase):
    def test_face_on_skips_plane(self):
        frames = _arc_swing()
        phases = detect_phases(frames)
        out = analyze_plane(frames, phases, "face_on", "right")
        self.assertEqual(out["plane"], "n/a")
        self.assertIsNone(out["ott"])
        self.assertEqual(out["skip_reason"], "view_not_dtl")

    def test_outside_downswing_is_ott(self):
        frames = _arc_swing(down_x=0.72)
        phases = detect_phases(frames)
        out = analyze_plane(frames, phases, "dtl", "right")
        self.assertEqual(out["hand_path"], "outside_in")
        self.assertTrue(out["ott"])


class WristTests(unittest.TestCase):
    def test_cup_when_knuckles_break_toward_head(self):
        frames = _arc_swing()
        phases = detect_phases(frames)
        for i in range(phases["top"] - 2, phases["top"] + 3):
            f = frames[i]
            wrist = f["lw"]
            f["lead_index"] = (0.50, f["head_y"] + 0.02)
            f["lead_pinky"] = (0.51, f["head_y"] + 0.02)
            f["le"] = (wrist[0], wrist[1] + 0.18)
        out = analyze_wrist(frames, phases, "right")
        self.assertEqual(out["lead_wrist_at_top"], "cupped")
        self.assertEqual(out["wrist_confidence"], "low")


class PayloadTests(unittest.TestCase):
    def test_complete_swing_payload(self):
        frames = _arc_swing()
        phases = detect_phases(frames)
        report = build_report(frames, phases, "dtl", "right")
        payload = coaching_payload(report)
        self.assertEqual(
            set(payload),
            {"view", "handedness", "phases_ok", "notes", "observed", "not_scored"},
        )
        self.assertTrue(payload["phases_ok"])
        self.assertIn("head_sway", payload["observed"])
        self.assertIn("ott", payload["observed"])
        self.assertNotIn("ott", {item["field"] for item in payload["not_scored"]})
        json.dumps(payload)

    def test_partial_review_when_top_missing(self):
        frames = [_frame(i + 1, 0.70) for i in range(20)]
        phases = detect_phases(frames)
        report = build_report(frames, phases, "dtl", "right")
        payload = coaching_payload(report)
        self.assertFalse(payload["phases_ok"])
        self.assertIn("head_sway", payload["observed"])
        skipped = {item["field"] for item in payload["not_scored"]}
        self.assertIn("ott", skipped)
        self.assertIn("lead_wrist_at_top", skipped)
        self.assertNotIn("ott", payload["observed"])
        self.assertIsNone(report["ott"])
        json.dumps(payload)

    def test_face_on_does_not_invent_ott(self):
        frames = _arc_swing()
        phases = detect_phases(frames)
        report = build_report(frames, phases, "face_on", "right")
        payload = coaching_payload(report)
        self.assertIn("head_sway", payload["observed"])
        self.assertNotIn("ott", payload["observed"])
        self.assertIn("ott", {item["field"] for item in payload["not_scored"]})


class OverlayLayersTests(unittest.TestCase):
    def test_ott_uses_plane_and_handpath(self):
        frames = _arc_swing(down_x=0.72)
        report = build_report(frames, detect_phases(frames), "dtl", "right")
        self.assertTrue(report["observed"].get("ott"))
        layers = overlay_layers(report)
        self.assertEqual(layers["flaw_id"], "ott")
        self.assertTrue(layers["plane"])
        self.assertTrue(layers["handpath"])
        self.assertFalse(layers["triangle"])
        self.assertFalse(layers["skeleton"])
        self.assertFalse(layers["head"])
        self.assertEqual(layers["focus"], "over-the-top")
        self.assertEqual(layers["rank"], "CRITICAL")

    def test_head_excessive_without_ott(self):
        frames = _sway_heads(_tilt_shoulders(_arc_swing()))
        report = build_report(frames, detect_phases(frames), "dtl", "right")
        self.assertNotEqual(report["observed"].get("ott"), True)
        self.assertNotEqual(report["observed"].get("connection"), "early_arm_lift")
        self.assertEqual(report["observed"].get("head_sway"), "excessive")
        layers = overlay_layers(report)
        self.assertEqual(layers["flaw_id"], "head")
        self.assertTrue(layers["skeleton"])
        self.assertTrue(layers["head"])
        self.assertFalse(layers["handpath"])
        self.assertFalse(layers["triangle"])
        self.assertFalse(layers["plane"])

    def test_no_flaw_is_skeleton_only(self):
        layers = overlay_layers({"observed": {"head_sway": "stable", "ott": False}})
        self.assertIsNone(layers["flaw_id"])
        self.assertTrue(layers["skeleton"])
        self.assertFalse(layers["head"])
        self.assertFalse(layers["triangle"])
        self.assertFalse(layers["plane"])
        self.assertFalse(layers["handpath"])
        self.assertEqual(layers["focus"], "pose")
        self.assertIsNone(layers["rank"])

    def test_face_on_head_does_not_enable_plane(self):
        frames = _sway_heads(_tilt_shoulders(_arc_swing()))
        report = build_report(frames, detect_phases(frames), "face_on", "right")
        payload = coaching_payload(report)
        self.assertNotIn("ott", payload["observed"])
        self.assertIn("ott", {item["field"] for item in payload["not_scored"]})
        self.assertEqual(report["observed"].get("head_sway"), "excessive")
        layers = overlay_layers(report)
        self.assertEqual(layers["flaw_id"], "head")
        self.assertTrue(layers["head"])
        self.assertFalse(layers["plane"])
        self.assertFalse(layers["handpath"])

    def test_head_critical_beats_wrist_moderate(self):
        layers = overlay_layers(
            {
                "observed": {
                    "lead_wrist_at_top": "cupped",
                    "head_sway": "excessive",
                }
            }
        )
        self.assertEqual(layers["flaw_id"], "head")
        self.assertTrue(layers["head"])
        self.assertFalse(layers["handpath"])
        self.assertFalse(layers["triangle"])
        self.assertTrue(layers["skeleton"])

    def test_ott_beats_head_when_both_critical(self):
        layers = overlay_layers(
            {"observed": {"ott": True, "head_sway": "excessive"}}
        )
        self.assertEqual(layers["flaw_id"], "ott")
        self.assertTrue(layers["plane"])
        self.assertTrue(layers["handpath"])
        self.assertFalse(layers["head"])
        self.assertFalse(layers["triangle"])

    def test_steep_without_ott_uses_plane(self):
        layers = overlay_layers(
            {"observed": {"plane": "steep", "ott": False, "head_sway": "stable"}}
        )
        self.assertEqual(layers["flaw_id"], "steep")
        self.assertTrue(layers["plane"])
        self.assertTrue(layers["handpath"])
        self.assertFalse(layers["triangle"])
        self.assertFalse(layers["skeleton"])
        self.assertEqual(layers["rank"], "MODERATE")

    def test_payload_omits_overlay(self):
        frames = _arc_swing()
        report = build_report(frames, detect_phases(frames), "dtl", "right")
        report["overlay"] = overlay_layers(report)
        payload = coaching_payload(report)
        self.assertNotIn("overlay", payload)


if __name__ == "__main__":
    unittest.main()
