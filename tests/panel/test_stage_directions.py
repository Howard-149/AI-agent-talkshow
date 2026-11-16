from agent.show.show_history import strip_stage_directions


def test_stage_directions_are_dropped():
    assert strip_stage_directions("Powerful image. [to the guest] You mentioned your friend.") == \
        "Powerful image. You mentioned your friend."
    assert strip_stage_directions("Sure [laughs], why not?") == "Sure, why not?"
    assert strip_stage_directions("No brackets here.") == "No brackets here."
    assert strip_stage_directions("A powerful image. [Guest], hearing Amy, does it change things?") == \
        "A powerful image. Hearing Amy, does it change things?"
    assert strip_stage_directions("[Guest], what do you think?") == "What do you think?"
    assert strip_stage_directions(None) is None
