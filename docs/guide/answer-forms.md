# Answers with a shape

This page goes on with the writer and the `count_words` code node of
[Steps and values](steps-and-values.md). An agent can answer in JSON with fixed fields.
The schema names each field with a type and, optionally, what it means.

```python
verdict = {
    "fits": {"type": "bool", "means": "the draft answers the question in under 50 words"},
    "action": {"type": "text", "one of": ["publish", "shorten"]},
}
judge = Agent(
    "judge",
    prompt="Question: {query}\nDraft, {count_words} words: {draft}\nJudge the draft.",
    inputs=["query", "draft", "count_words"],
    output="verdict",
    output_type="json",
    schema=verdict,
)
mas = MAS(
    [writer, judge],
    code_nodes=[Code("count", uses="count_words")],
    edges=[("writer", "count"), ("count", "judge")],
    tools=[count_words],
    llm=llm,
)

trace = mas.run("Why is the sky blue?")
print(trace["judge"][0].rendered_prompt)
```

For a four-word draft, the judge's model sees this prompt. The last lines come from the
schema:

```text
Question: Why is the sky blue?
Draft, 4 words: Sunlight scatters off air.
Judge the draft.

Answer with one JSON object and nothing else. Its fields:
- fits (true or false): the draft answers the question in under 50 words
- action (one of "publish", "shorten")
```

No optimizer can drop these format lines. An answer that does not fit gets exactly one
retry, with a note that says what was wrong. The record keeps the refused answer:

```python
print(trace.output)  # {"fits": true, "action": "publish"}
print(trace["judge"][0].rejected)  # the answer refused before the retry, if there was one
```

Two misses in a row end the run. Then `trace.status` is `failed` and `trace.error_kind`
is `bad_format`; calling `mas(...)` raises a `RuntimeError` instead.

Each field takes a `type`: `text`, `bool`, `int`, `number`, `json`, `object` or
`list of <type>`. It may add `means`, `optional`, `one of`, and `fields` for an object or
a list of objects.
