"""Debates in simultaneous rounds.

Every speaker of a round answers the same transcript at the same time: round 1
is the opening statements, written blind to the other side, and each later
round answers what the others said in the round before. Nobody speaks first or
last, so no side holds the last word.

A speaker writes only its own turn (``<speaker>_turn``); the round's end step
adds the turns to the transcript in a fixed order, the same every round.
"""

from __future__ import annotations

import re

_HEADING = re.compile(r"^(#{1,6})(?=\s)", re.MULTILINE)


def turn_key(speaker: str) -> str:
    """The debate-state key holding ``speaker``'s turn in the round in progress."""
    return f"{speaker.lower()}_turn"


def record_turn(debate_key: str, speaker: str, text: str) -> dict:
    """A speaker's state update: its turn in the round in progress, nothing else."""
    return {debate_key: {turn_key(speaker): text}}


def round_title(number: int) -> str:
    return "Opening statements" if number == 1 else "Rebuttals"


def _nest_headings(text: str) -> str:
    """Move a turn's own headings below the speaker's, so the transcript keeps its shape."""
    return _HEADING.sub(lambda m: "#" * min(len(m.group(1)) + 3, 6), text)


def create_round_end(debate_key: str, speakers: tuple[str, ...]):
    """The step that closes a round: add its turns to the transcript, speakers in order."""

    def round_end(state) -> dict:
        debate = state[debate_key]
        count = debate.get("count", 0)
        number = count // len(speakers) + 1
        turns = [(speaker, (debate.get(turn_key(speaker)) or "").strip() or "(no argument)")
                 for speaker in speakers]

        block = f"## Round {number}: {round_title(number)}\n\n" + "\n\n".join(
            f"### {speaker} Analyst\n\n{_nest_headings(text)}" for speaker, text in turns
        )
        history = debate.get("history", "")
        update = {
            "history": f"{history}\n\n{block}" if history else block,
            "current_response": block,
            "count": count + len(speakers),
        }
        for speaker, text in turns:
            name = speaker.lower()
            argument = f"{speaker} Analyst: {text}"
            update[f"{name}_history"] = f"{debate.get(f'{name}_history', '')}\n{argument}"
            update[f"current_{name}_response"] = argument
            update[turn_key(speaker)] = ""
        if "latest_speaker" in debate:
            update["latest_speaker"] = speakers[-1]
        return {debate_key: update}

    return round_end
