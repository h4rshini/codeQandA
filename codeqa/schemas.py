"""Step 3: the structured answer every agent run must end with."""
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class CitedFile(BaseModel):
    path: str = Field(description="Repo-relative file path.")
    lines: Annotated[list[int], Field(min_length=2, max_length=2)] = Field(
        description="[start_line, end_line], 1-indexed, inclusive.")


class AgentAnswer(BaseModel):
    answer: str
    cited_files: list[CitedFile]
    confidence: Literal["high", "medium", "low"]


# Shape-only: whether cited files/lines actually exist is checked by the eval
# (hallucination check), not here, so invented citations are counted, not hidden.
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "agent_answer", "schema": AgentAnswer.model_json_schema()},
}
