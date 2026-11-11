"""Prompts for code summaries and code question answering.

Summary prompts are complete instructions: the code index calls the summarization model
with the filled-in prompt as its context, so the model should pass it through unchanged
(see :class:`raptor.code.models.PassthroughSummarizer`).
"""

from __future__ import annotations

_COMMON = (
    "Be factual and specific; only describe what the code shows. Mention concrete names "
    "(functions, classes, modules) so the summary can be found by search. Do not use "
    "markdown headings. Write at most {max_words} words."
)

SYMBOL_PROMPT = (
    "Summarize this {language} {symbol_kind} from `{path}` for a developer navigating "
    "the codebase. Cover: its purpose; inputs and outputs (parameters, return value, "
    "raised errors); side effects (I/O, mutation, network, global state); and notable "
    "dependencies it calls. " + _COMMON + "\n\n```{language}\n{code}\n```"
)

CLASS_PROMPT = (
    "Summarize the {language} class `{name}` defined in `{path}`. Cover: its "
    "responsibility; its public API (key methods and what they do); important state it "
    "holds; side effects; and how it relates to its base classes and collaborators. "
    + _COMMON
    + "\n\nClass header:\n```{language}\n{header}\n```\n\nMembers:\n{members}"
)

FILE_PROMPT = (
    "Summarize the source file `{path}`. Cover: the module's purpose; its public API "
    "(main classes and functions and what they do); inputs and outputs; side effects "
    "(I/O, network, global state); and its dependencies on other modules. " + _COMMON + "\n\n{body}"
)

DIR_PROMPT = (
    "Summarize the directory `{path}/` of a codebase from the summaries of its files "
    "and subdirectories below. Cover: what this part of the system is responsible for; "
    "its main components and how they fit together; the public entry points; and its "
    "dependencies on the rest of the codebase. " + _COMMON + "\n\n{children}"
)

REPO_PROMPT = (
    "Write an architectural overview of the repository `{name}` from the summaries of "
    "its top-level parts below. Cover: what the project does; its main components and "
    "how a typical request or call flows through them; the public entry points; and key "
    "external dependencies. " + _COMMON + "\n\n{readme}{children}"
)

CONCEPT_PROMPT = (
    "The following files and classes from one codebase were grouped together because "
    "they are related. Start with a short name for the cross-cutting concern they "
    "implement (for example 'authentication flow' or 'persistence layer') on its own "
    "line as `Concept: <name>`. Then explain how these pieces work together, which one "
    "does what, and where a developer should look first. " + _COMMON + "\n\n{children}"
)

CODE_QA_INSTRUCTIONS = (
    "You are answering a question about a codebase using the retrieved code and "
    "summaries below. Each snippet starts with its location as [path:start-end]. "
    "Answer directly and concretely, name the relevant functions and classes, and cite "
    "every claim with the location in square brackets, e.g. [raptor/builder.py:40-75]. "
    "If the context does not contain the answer, say what is missing."
)

CODE_QA_PROMPT = (
    CODE_QA_INSTRUCTIONS + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

JUDGE_PROMPT = (
    "You are grading an answer to a question about a codebase.\n\n"
    "Question: {question}\n\n"
    "Reference answer (written by a developer from the code):\n{reference}\n\n"
    "Candidate answer:\n{answer}\n\n"
    "Score the candidate from 1 to 5 on each criterion:\n"
    "- correctness: agrees with the reference and makes no false claims about the code\n"
    "- completeness: covers the key points of the reference\n"
    "- grounding: cites concrete files/functions that are actually relevant\n"
    'Reply with JSON only: {{"correctness": n, "completeness": n, "grounding": n, '
    '"comment": "<one sentence>"}}'
)
