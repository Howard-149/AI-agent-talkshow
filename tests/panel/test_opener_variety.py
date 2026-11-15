from types import SimpleNamespace as NS

from agent.panel.think_prompt import opener_block


def test_opener_block_lists_own_recent_openers():
    h = NS(lines=[NS(role_id="guest", text="When I look at rents, I see families."),
                  NS(role_id="commentator", text="So the market wins?"),
                  NS(role_id="guest", text="It is not that simple.")])
    b = opener_block(h, "guest")
    assert '"When I look..."' in b and '"It is not..."' in b and "So the" not in b
    assert b.endswith("open this one differently.")


def test_opener_block_empty_before_speaking():
    assert opener_block(NS(lines=[NS(role_id="host", text="Welcome.")]), "guest") == ""


def test_demo_arm_flags():
    from eval.quality_ruler.arms import ARMS

    f = ARMS["demo"]["live_flags"]
    assert f["TALKSHOW_OPENER_VARIETY"] == "1" and f["TALKSHOW_FLOOR_JEV"] == "1"
    assert ARMS["demo"]["fact_pack"] is True
