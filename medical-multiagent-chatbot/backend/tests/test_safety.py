from backend.safety.postprocess import safety_postprocess


def test_safety_rewrites_diagnosis_and_dosing() -> None:
    text = "You have hypertension. Stop taking meds. 10 mg daily."
    result = safety_postprocess(text)
    assert "You have" not in result.text
    assert "Stop taking" not in result.text
    assert "[dosing removed]" in result.text

#smoke test to test if the safety guardrails are working