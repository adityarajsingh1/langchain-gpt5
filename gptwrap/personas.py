"""
Personas: reusable system prompts with blanks to fill in, e.g. a tutor for
{subject} at a given {level}. They're plain LangChain PromptTemplates, so
a missing variable is caught before anything is sent to the API.
"""

from dataclasses import dataclass, field

from langchain_core.prompts import PromptTemplate


class PersonaError(ValueError):
    pass


@dataclass
class Persona:
    name: str
    description: str
    template: str
    defaults: dict = field(default_factory=dict)

    @property
    def variables(self):
        return sorted(PromptTemplate.from_template(self.template).input_variables)

    def render(self, **values):
        """The system prompt with every {variable} filled in. Defaults apply where nothing is given."""
        merged = {**self.defaults, **{k: v for k, v in values.items() if v is not None}}
        unknown = set(merged) - set(self.variables)
        if unknown:
            raise PersonaError(f"persona '{self.name}' has no variable(s) {', '.join(sorted(unknown))}. "
                               f"it takes: {', '.join(self.variables) or 'nothing'}")
        missing = [v for v in self.variables if v not in merged]
        if missing:
            raise PersonaError(f"persona '{self.name}' needs a value for: {', '.join(missing)}")
        return PromptTemplate.from_template(self.template).format(**merged)


BUILT_IN = {p.name: p for p in [
    Persona(
        "default", "general helpful assistant",
        "You are a helpful assistant. Answer clearly and keep it short unless asked for detail.",
    ),
    Persona(
        "tutor", "explains a subject step by step at the right level",
        "You are a patient {subject} tutor for a {level} student. Explain step by step, use small examples, "
        "and check understanding with a short question at the end. Don't just hand over answers to homework, "
        "guide the student to them.",
        defaults={"level": "beginner"},
    ),
    Persona(
        "reviewer", "code reviewer for one language",
        "You are a senior {language} developer doing a code review. Point out bugs first, then readability "
        "and style. Be specific: quote the line and suggest the fix. If the code is fine, say so briefly.",
        defaults={"language": "Python"},
    ),
    Persona(
        "translator", "translates into another language, nothing else",
        "Translate everything the user writes into {target_language}. Keep the tone and formatting. "
        "Only output the translation, no explanations.",
    ),
    Persona(
        "interviewer", "mock job interviewer",
        "You are interviewing the user for a {role} position. Ask one question at a time, wait for the answer, "
        "then give short honest feedback before the next question. Mix technical and behavioural questions.",
        defaults={"role": "junior software engineer"},
    ),
]}


def get_persona(name):
    try:
        return BUILT_IN[name]
    except KeyError:
        raise PersonaError(f"no persona called '{name}'. available: {', '.join(BUILT_IN)}") from None
