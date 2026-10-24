"""Round router / path_map completeness (#1088).

Each debate's router runs at the end of a round and returns either all of the
debate's speakers, for another round, or its judge. A return missing from the
shared path map (`DEBATE_PATH_MAP` / `RISK_ANALYSIS_PATH_MAP`) would crash
LangGraph mid-run. The router reads only the turn count, so drift in the
speakers' labels (renames, translations) cannot change where it goes.
"""
import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import DEBATE_PATH_MAP, RISK_ANALYSIS_PATH_MAP


def _risk(count, latest_speaker=""):
    return {"risk_debate_state": {"latest_speaker": latest_speaker, "count": count}}


def _debate(count, current_response=""):
    return {"investment_debate_state": {"current_response": current_response, "count": count}}


def _targets(target):
    return {target} if isinstance(target, str) else set(target)


@pytest.mark.unit
@pytest.mark.parametrize("rounds", [1, 2, 5])
@pytest.mark.parametrize("count", range(0, 16))
def test_every_router_target_is_in_its_path_map(rounds, count):
    logic = ConditionalLogic(max_debate_rounds=rounds, max_risk_discuss_rounds=rounds)

    assert _targets(logic.should_continue_debate(_debate(count))) <= set(DEBATE_PATH_MAP)
    assert _targets(logic.should_continue_risk_analysis(_risk(count))) <= set(RISK_ANALYSIS_PATH_MAP)


@pytest.mark.unit
def test_the_routers_end_at_the_round_limit():
    logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    assert logic.should_continue_debate(_debate(2)) == "Research Manager"
    assert logic.should_continue_risk_analysis(_risk(3)) == "Portfolio Manager"


@pytest.mark.unit
@pytest.mark.parametrize("label", [
    "Bull Analyst: up", "Aggressive", "",
    "Aggressive Risk Analyst",   # drift: node renamed
    "Agresivo", "Optimista",     # drift: i18n / translated label
])
def test_a_speakers_label_does_not_change_where_the_router_goes(label):
    logic = ConditionalLogic(max_debate_rounds=1, max_risk_discuss_rounds=1)

    assert logic.should_continue_debate(_debate(0, label)) == ["Bull Researcher", "Bear Researcher"]
    assert logic.should_continue_risk_analysis(_risk(0, label)) == [
        "Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"]
