from eval.quality_ruler.arms import ARMS, LIVE_FLAG_NAMES


def test_every_arm_flag_is_applied():
    """apply_env sets only the flags in LIVE_FLAG_NAMES; an arm flag missing there is silently dropped and the arm
    runs as if the flag were off."""
    for name, arm in ARMS.items():
        missing = set(arm.get("live_flags") or {}) - set(LIVE_FLAG_NAMES)
        assert not missing, (name, missing)


def test_opener_metric():
    from eval.bench.score import RESTATE, opener_word

    assert RESTATE.match("So, you're saying the market wins?")
    assert RESTATE.match("You're suggesting we wait?")
    assert not RESTATE.match("Seventeen billion over budget.")
    assert opener_word("So, the dollar") == "so"
