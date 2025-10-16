from sdk.core import SanctumAI

def test_basic():
    ai = SanctumAI()
    out = ai.generate_insight({"cycle_length_days": 29})
    assert isinstance(out, str) and len(out) > 10
