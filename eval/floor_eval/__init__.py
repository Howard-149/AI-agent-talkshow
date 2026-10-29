"""Offline floor-control evaluation: planted-stimulus tests for interruption decisions.

A panelist's line is revealed one played sentence at a time to the listeners,
using the production intent prompts and LLM client. One sentence may carry a
planted problem (an overclaim, a factual error, or an off-topic drift). Arms
(poll schedule, rules on/off, moderator on/off) are simulated afterwards from
the same cached model answers, so they differ only in floor-control logic.

    python -m eval.floor_eval.stimuli    # generate + validate test lines
    python -m eval.floor_eval.run        # query listeners at every prefix
    python -m eval.floor_eval.analyze    # simulate arms, write summary
"""
