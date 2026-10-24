from tradingagents.agents.state import AgentState

# The speakers of each debate's rounds; every round runs all of them at once.
RESEARCH_SPEAKERS = ("Bull Researcher", "Bear Researcher")
RISK_SPEAKERS = ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst")


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_debate(self, state: AgentState) -> str | list[str]:
        """Another round with both researchers, or the Research Manager once the rounds are done."""
        if state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds:
            return "Research Manager"
        return list(RESEARCH_SPEAKERS)

    def should_continue_risk_analysis(self, state: AgentState) -> str | list[str]:
        """Another round with all three risk analysts, or the Portfolio Manager once the rounds are done."""
        if state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds:
            return "Portfolio Manager"
        return list(RISK_SPEAKERS)
