"""Debates run in simultaneous rounds.

Each round, every speaker answers the same transcript at the same time: round 1
is the opening statements, written blind to the other side, and each later round
answers what the others said in the round before. Nobody speaks first or last,
so no side gets the last word, and the managers read the debate as one
transcript in round order.
"""

from __future__ import annotations

import re

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from tests.test_graph_end_to_end import TRADE_DATE, ScriptedModel, _graph, offline  # noqa: F401
from tradingagents.graph.conditional_logic import ConditionalLogic

RESEARCH = ("Bull Researcher", "Bear Researcher")
RISK = ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst")


class DebateModel(ScriptedModel):
    """Answers each debate turn with a marker naming the speaker and its turn."""

    seen: list = Field(default_factory=list)    # (node, step, prompt), shared across copies

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        metadata = (run_manager.metadata or {}) if run_manager else {}
        node, step = metadata.get("langgraph_node"), metadata.get("langgraph_step")
        prompt = "\n".join(str(m.content) for m in messages)
        self.seen.append((node, step, prompt))
        if node not in RESEARCH + RISK:
            return super()._generate(messages, stop, run_manager, **kwargs)
        self._count()
        turn = sum(1 for n, _s, _p in self.seen if n == node)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=_turn(node, turn)))])

    def turns_of(self, node):
        return [(step, prompt) for n, step, prompt in self.seen if n == node]


def _turn(node: str, turn: int) -> str:
    return f"<{node} turn {turn}>"


@pytest.fixture
def two_rounds(tmp_path, monkeypatch, offline):  # noqa: F811
    model = DebateModel()
    graph = _graph(tmp_path, monkeypatch, model, max_debate_rounds=2, max_risk_discuss_rounds=2)
    state, _ = graph.propagate("NVDA", TRADE_DATE)
    return model, state


@pytest.mark.unit
@pytest.mark.parametrize("speakers", [RESEARCH, RISK], ids=["research", "risk"])
def test_the_speakers_of_a_round_answer_at_the_same_time(two_rounds, speakers):
    model, _ = two_rounds

    steps = {node: [step for step, _ in model.turns_of(node)] for node in speakers}

    assert all(len(s) == 2 for s in steps.values()), steps   # one turn per round, two rounds
    assert len({tuple(s) for s in steps.values()}) == 1, steps  # every speaker in the same steps


@pytest.mark.unit
@pytest.mark.parametrize("speakers", [RESEARCH, RISK], ids=["research", "risk"])
def test_an_opening_statement_is_written_blind_to_the_others(two_rounds, speakers):
    model, _ = two_rounds

    for node in speakers:
        (_, opening), (_, rebuttal) = model.turns_of(node)
        for other in speakers:
            if other != node:
                assert _turn(other, 1) not in opening, (node, other)
                assert _turn(other, 1) in rebuttal, (node, other)


@pytest.mark.unit
def test_a_rebuttal_does_not_see_the_same_rounds_other_answers(two_rounds):
    model, _ = two_rounds

    _, bull_rebuttal = model.turns_of("Bull Researcher")[1]

    assert _turn("Bear Researcher", 2) not in bull_rebuttal


@pytest.mark.unit
@pytest.mark.parametrize("judge,speakers", [("Research Manager", RESEARCH),
                                            ("Portfolio Manager", RISK)], ids=["research", "risk"])
def test_the_judge_reads_the_debate_as_a_transcript_in_round_order(two_rounds, judge, speakers):
    model, _ = two_rounds

    prompt = model.turns_of(judge)[0][1]
    expected = ["Round 1", *(_turn(s, 1) for s in speakers), "Round 2", *(_turn(s, 2) for s in speakers)]
    positions = [prompt.find(text) for text in expected]

    assert -1 not in positions, dict(zip(expected, positions, strict=True))
    assert positions == sorted(positions)


@pytest.mark.unit
def test_the_transcript_marks_the_openings_and_the_rebuttals(two_rounds):
    _, state = two_rounds

    history = state["investment_debate_state"]["history"]

    assert re.search(r"Round 1\b.*Opening statements", history)
    assert re.search(r"Round 2\b.*Rebuttals", history)
    assert state["investment_debate_state"]["count"] == 4
    assert state["risk_debate_state"]["count"] == 6


@pytest.mark.unit
def test_each_speakers_own_history_keeps_its_turns(two_rounds):
    _, state = two_rounds

    bull = state["investment_debate_state"]["bull_history"]
    neutral = state["risk_debate_state"]["neutral_history"]

    assert _turn("Bull Researcher", 1) in bull and _turn("Bull Researcher", 2) in bull
    assert _turn("Bear Researcher", 1) not in bull
    assert _turn("Neutral Analyst", 2) in neutral


# --- the router -------------------------------------------------------------


def _debate(count):
    return {"investment_debate_state": {"count": count}}


def _risk(count):
    return {"risk_debate_state": {"count": count}}


@pytest.mark.unit
def test_the_research_router_runs_every_round_with_both_speakers():
    logic = ConditionalLogic(max_debate_rounds=2)

    assert logic.should_continue_debate(_debate(0)) == list(RESEARCH)
    assert logic.should_continue_debate(_debate(2)) == list(RESEARCH)
    assert logic.should_continue_debate(_debate(4)) == "Research Manager"


@pytest.mark.unit
def test_the_risk_router_runs_every_round_with_all_three_speakers():
    logic = ConditionalLogic(max_risk_discuss_rounds=2)

    assert logic.should_continue_risk_analysis(_risk(0)) == list(RISK)
    assert logic.should_continue_risk_analysis(_risk(3)) == list(RISK)
    assert logic.should_continue_risk_analysis(_risk(6)) == "Portfolio Manager"


# --- the report ---------------------------------------------------------------

TRANSCRIPT = "## Round 1: Opening statements\n\n### Bull Analyst\n\nup\n\n### Bear Analyst\n\ndown"
RISK_TRANSCRIPT = "## Round 1: Opening statements\n\n### Aggressive Analyst\n\nmore\n\n### Neutral Analyst\n\nsame"


def _finished_state():
    return {
        "market_report": "MKT", "investment_plan": "RM PLAN", "trader_investment_plan": "TRADE",
        "final_trade_decision": "PM DECISION",
        "investment_debate_state": {"history": TRANSCRIPT, "bull_history": "\nBull Analyst: up",
                                    "bear_history": "\nBear Analyst: down"},
        "risk_debate_state": {"history": RISK_TRANSCRIPT, "aggressive_history": "\nAggressive Analyst: more"},
    }


@pytest.mark.unit
def test_the_report_saves_each_debate_as_one_transcript(tmp_path):
    from tradingagents.reporting import write_report_tree

    complete = write_report_tree(_finished_state(), "NVDA", tmp_path).read_text()

    assert (tmp_path / "2_research" / "debate.md").read_text() == TRANSCRIPT
    assert (tmp_path / "4_risk" / "debate.md").read_text() == RISK_TRANSCRIPT
    assert not [p.name for p in tmp_path.rglob("*.md")
                if p.name in {"bull.md", "bear.md", "aggressive.md", "conservative.md", "neutral.md"}]
    assert complete.index("### Research Debate") < complete.index("### Research Manager")
    assert "### Risk Debate" in complete


@pytest.mark.unit
def test_the_cli_report_shows_each_debate_as_one_transcript():
    from cli import display

    with display.console.capture() as captured:
        display.display_complete_report(_finished_state())
    shown = captured.get()

    assert "Research Debate" in shown and "Risk Debate" in shown
    assert "Bull Researcher" not in shown and "Aggressive Analyst" in shown   # a speaker inside the transcript
    assert shown.index("Bull Analyst") < shown.index("Bear Analyst")
