"""
The Assistant: one system prompt plus whatever chat model you give it.
Takes a GPT-5 ChatOpenAI by default, but any LangChain chat model works,
which is how the tests run without an API key.
"""

import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

DEFAULT_SYSTEM = "You are a helpful assistant. Answer clearly and keep it short unless asked for detail."


class AssistantError(RuntimeError):
    pass


def _explain(err):
    # turn the openai SDK's exceptions into one line that says what to do about it
    if isinstance(err, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY."
    if isinstance(err, openai.NotFoundError):
        return "OpenAI doesn't know that model, or your key can't use it. Check OPENAI_MODEL."
    if isinstance(err, openai.RateLimitError):
        return "Hit OpenAI's rate limit (or ran out of credit). Wait a bit and try again."
    if isinstance(err, openai.APITimeoutError):
        return "OpenAI took too long to answer. Try again, or raise OPENAI_TIMEOUT."
    if isinstance(err, openai.APIConnectionError):
        return "Couldn't reach OpenAI. Check your internet connection (or OPENAI_BASE_URL)."
    return f"OpenAI returned an error: {err}"


class Assistant:
    def __init__(self, model=None, system=DEFAULT_SYSTEM):
        if model is None:
            from gptwrap.models import chat_model
            model = chat_model()
        self.model = model
        self.system = system
        prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{question}")])
        self.chain = prompt | model | StrOutputParser()

    def ask(self, question):
        question = question.strip()
        if not question:
            raise ValueError("ask() needs a question")
        try:
            return self.chain.invoke({"system": self.system, "question": question}).strip()
        except openai.OpenAIError as e:
            raise AssistantError(_explain(e)) from e
