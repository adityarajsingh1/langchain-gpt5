"""
Structured output: get the model's answer back as a Pydantic object instead
of free text, so other code can use it without parsing prose.

Summary is the built-in example, but extract() works with any Pydantic model.
"""

from typing import Literal

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from gptwrap.assistant import AssistantError, _explain


class Summary(BaseModel):
    """A structured summary of a piece of text."""

    title: str = Field(description="a short title, at most 8 words")
    tldr: str = Field(description="the whole text in one sentence")
    key_points: list[str] = Field(description="the main points, 1 to 7 of them, one sentence each",
                                  min_length=1, max_length=7)
    action_items: list[str] = Field(default_factory=list,
                                    description="things someone has to do, if the text mentions any")
    sentiment: Literal["positive", "neutral", "negative", "mixed"] = Field(description="overall tone")

    def to_markdown(self) -> str:
        lines = [f"## {self.title}", "", self.tldr, "", *[f"- {p}" for p in self.key_points]]
        if self.action_items:
            lines += ["", "**To do**", *[f"- [ ] {a}" for a in self.action_items]]
        lines += ["", f"_tone: {self.sentiment}_"]
        return "\n".join(lines)


EXTRACT_SYSTEM = ("Fill in the requested structure from the user's text. Only use information that is in "
                  "the text, don't add facts of your own.")


def extract(model, text, schema, instructions=EXTRACT_SYSTEM, retries=1):
    """Ask `model` to turn `text` into an instance of the Pydantic `schema`.

    If the model returns something that doesn't fit the schema, it gets the
    validation error and one more try (`retries`), which fixes most slips.
    """
    if not text.strip():
        raise ValueError("there's no text to work with")
    structured = model.with_structured_output(schema)
    messages = [SystemMessage(instructions), HumanMessage(text)]
    for attempt in range(retries + 1):
        try:
            result = structured.invoke(messages)
        except openai.OpenAIError as e:
            raise AssistantError(_explain(e)) from e
        except (ValidationError, OutputParserException) as e:
            error = str(e)
        else:
            if isinstance(result, schema):
                return result
            error = "the model didn't return the requested structure"
        if attempt < retries:
            messages = messages + [HumanMessage(f"That didn't fit the structure: {error}\nPlease try again.")]
    raise AssistantError(f"couldn't get a valid {schema.__name__} out of the model: {error}")


def summarize(model, text):
    return extract(model, text, Summary)
