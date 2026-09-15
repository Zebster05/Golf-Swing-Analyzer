import json
import unittest

from gemini_coach import (
    GEMINI_COACH_MODELS,
    coach_user_profile,
    compose_golfer_note,
    generate_coaching_with_fallback,
    is_rate_limit_error,
    is_retryable_gemini_error,
    redact_secrets,
)


class _ApiError(Exception):
    def __init__(self, message, code=None, status=None, status_code=None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.status_code = status_code


class _Response:
    def __init__(self, payload):
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload


def _coaching_json():
    return {
        "summary": "Primary fault is over-the-top.",
        "positives": ["Stable head"],
        "negatives": ["Steep downswing"],
        "drills": [
            {
                "problem": "Over-the-top",
                "name": "Pump drill",
                "why": "Shallows the path.",
                "steps": ["Pause at top", "Drop the trail elbow"],
            }
        ],
        "pro_tip": "Feel the club drop behind you.",
    }


class ModelListTests(unittest.TestCase):
    def test_validated_free_tier_order(self):
        self.assertEqual(
            GEMINI_COACH_MODELS,
            (
                "gemini-3.6-flash",
                "gemini-3.5-flash",
                "gemini-3.8-flash",
                "gemini-3.7-flash",
                "gemini-flash-latest",
                "gemini-3-flash-preview",
                "gemini-3.5-flash-lite",
                "gemini-3.1-flash-lite",
                "gemini-flash-lite-latest",
            ),
        )
        for removed in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
            self.assertNotIn(removed, GEMINI_COACH_MODELS)


class ErrorClassificationTests(unittest.TestCase):
    def test_retryable_status_codes(self):
        for code in (429, 503, 404):
            self.assertTrue(is_retryable_gemini_error(_ApiError("fail", code=code)))

    def test_retryable_message_markers(self):
        markers = (
            "UNAVAILABLE",
            "high demand",
            "RESOURCE_EXHAUSTED",
            "NOT_FOUND",
            "no longer available",
        )
        for marker in markers:
            self.assertTrue(is_retryable_gemini_error(_ApiError(marker)))

    def test_rate_limit_detected(self):
        self.assertTrue(is_rate_limit_error(_ApiError("quota", code=429)))
        self.assertTrue(is_rate_limit_error(_ApiError("RESOURCE_EXHAUSTED")))
        self.assertFalse(is_rate_limit_error(_ApiError("not found", code=404)))

    def test_auth_error_is_not_retryable(self):
        exc = _ApiError("API key not valid", code=400)
        self.assertFalse(is_retryable_gemini_error(exc))
        self.assertFalse(is_rate_limit_error(exc))


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.sleeps = []
        self.calls = []

    def _sleep(self, seconds):
        self.sleeps.append(seconds)

    def _run(self, generate_content, **kwargs):
        return generate_coaching_with_fallback(
            "prompt",
            generate_content,
            sleep=self._sleep,
            fallback_delay=0.1,
            base_delay=2,
            **kwargs,
        )

    def test_success_attaches_model_used(self):
        payload = _coaching_json()

        def generate_content(model, prompt):
            self.calls.append(model)
            self.assertEqual(prompt, "prompt")
            return _Response(payload)

        result = self._run(generate_content)
        self.assertEqual(result["summary"], payload["summary"])
        self.assertEqual(result["positives"], payload["positives"])
        self.assertEqual(result["drills"][0]["name"], "Pump drill")
        self.assertEqual(result["_model_used"], GEMINI_COACH_MODELS[0])
        self.assertEqual(self.calls, [GEMINI_COACH_MODELS[0]])

    def test_503_falls_back_to_next_model(self):
        payload = _coaching_json()

        def generate_content(model, prompt):
            self.calls.append(model)
            if model == GEMINI_COACH_MODELS[0]:
                raise _ApiError("The model is overloaded", code=503, status="UNAVAILABLE")
            return _Response(payload)

        result = self._run(generate_content)
        self.assertEqual(result["_model_used"], GEMINI_COACH_MODELS[1])
        self.assertEqual(self.calls, [GEMINI_COACH_MODELS[0], GEMINI_COACH_MODELS[1]])
        self.assertEqual(self.sleeps, [0.1])

    def test_404_falls_back_to_next_model(self):
        payload = _coaching_json()

        def generate_content(model, prompt):
            self.calls.append(model)
            if model == GEMINI_COACH_MODELS[0]:
                raise _ApiError("model is no longer available", code=404, status="NOT_FOUND")
            return _Response(payload)

        result = self._run(generate_content)
        self.assertEqual(result["_model_used"], GEMINI_COACH_MODELS[1])
        self.assertEqual(self.calls, list(GEMINI_COACH_MODELS[:2]))

    def test_429_retries_same_model_then_falls_back(self):
        payload = _coaching_json()

        def generate_content(model, prompt):
            self.calls.append(model)
            if model == GEMINI_COACH_MODELS[0]:
                raise _ApiError("RESOURCE_EXHAUSTED", code=429)
            return _Response(payload)

        result = self._run(generate_content, rate_limit_retries=2)
        self.assertEqual(result["_model_used"], GEMINI_COACH_MODELS[1])
        self.assertEqual(
            self.calls,
            [GEMINI_COACH_MODELS[0]] * 3 + [GEMINI_COACH_MODELS[1]],
        )
        self.assertEqual(self.sleeps, [2, 4, 0.1])

    def test_non_retryable_returns_immediately(self):
        def generate_content(model, prompt):
            self.calls.append(model)
            raise _ApiError("API key not valid. Please pass a valid API key.", code=400)

        result = self._run(generate_content)
        self.assertIn("error", result)
        self.assertIn("API key not valid", result["error"])
        self.assertNotIn("All Gemini model fallbacks failed", result["error"])
        self.assertEqual(self.calls, [GEMINI_COACH_MODELS[0]])
        self.assertEqual(self.sleeps, [])

    def test_all_models_fail_returns_combined_error(self):
        def generate_content(model, prompt):
            self.calls.append(model)
            raise _ApiError(f"{model} high demand", code=503)

        result = self._run(generate_content)
        self.assertIn("error", result)
        self.assertIn("All Gemini model fallbacks failed", result["error"])
        self.assertIn(GEMINI_COACH_MODELS[0], result["error"])
        self.assertIn(GEMINI_COACH_MODELS[-1], result["error"])
        self.assertIn("high demand", result["error"])
        self.assertEqual(self.calls, list(GEMINI_COACH_MODELS))

    def test_does_not_leak_api_key(self):
        key = "secret-gemini-key"

        def generate_content(model, prompt):
            raise _ApiError(f"auth failed for {key}", code=401)

        result = self._run(generate_content, api_key=key)
        self.assertNotIn(key, result["error"])
        self.assertIn("[redacted]", result["error"])
        self.assertEqual(redact_secrets(f"token {key}", key), "token [redacted]")


class CoachUserProfileTests(unittest.TestCase):
    def test_composes_title_and_body(self):
        profile = coach_user_profile(
            "Mid (10-19)",
            "Slice (Right)",
            "Iron",
            "dtl",
            "right",
            post_title="  Why do I slice?  ",
            post_body="  I hang back and lose it right.  ",
        )
        self.assertEqual(profile["post_title"], "Why do I slice?")
        self.assertEqual(profile["post_body"], "I hang back and lose it right.")
        self.assertEqual(
            profile["golfer_note"],
            "Why do I slice?\n\nI hang back and lose it right.",
        )
        self.assertEqual(profile["handicap"], "Mid (10-19)")
        self.assertEqual(profile["common_miss"], "Slice (Right)")
        self.assertEqual(profile["club"], "Iron")

    def test_empty_fields_stay_empty(self):
        profile = coach_user_profile(
            "High (20+)", "Hook (Left)", "Driver", "face_on", "left"
        )
        self.assertEqual(profile["post_title"], "")
        self.assertEqual(profile["post_body"], "")
        self.assertEqual(profile["golfer_note"], "")

    def test_title_only_and_body_only(self):
        self.assertEqual(compose_golfer_note("Help", ""), "Help")
        self.assertEqual(compose_golfer_note("", "I chunk irons"), "I chunk irons")
        self.assertEqual(compose_golfer_note("   ", "\n"), "")

    def test_long_reddit_body_is_not_truncated(self):
        long_body = "I hang back. " * 80
        self.assertGreater(len(long_body), 400)
        profile = coach_user_profile(
            "Beginner",
            "Inconsistent",
            "Wedge",
            "dtl",
            "right",
            post_title="Range session thoughts",
            post_body=long_body,
        )
        self.assertEqual(profile["post_body"], long_body.strip())
        self.assertIn(long_body.strip(), profile["golfer_note"])
        self.assertGreater(len(profile["golfer_note"]), 400)


if __name__ == "__main__":
    unittest.main()
