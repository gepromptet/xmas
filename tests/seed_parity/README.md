# Seed parity fixture

A trimmed copy of the seed implementation, kept so that
`xmas.optimization.attribution.ContrastiveAttributer` can be checked against the
code it came from rather than against a golden file this repository wrote
for itself.

**Source:** `mas_lib/attribution/{base,absolute}.py` at `tritthart/mas-optimization`
`main` (`2db997d`), which is itself adapted from
[`finitearth/cantante`](https://github.com/finitearth/cantante). The attribution
modules on `main` and on `t/feature/optimize-attribution` are identical once
docstrings are stripped, so either ref gives the same fixture.

**What was changed, and nothing else:**

- `src.prompt_utils.constrain_prompt_length` is stubbed as the identity. It
  truncates by `tiktoken` token count, and `tiktoken` is not an xmas dependency.
  The parity fixture stays well inside any budget, so the real function would
  also return its input unchanged.
- `logger` is a module-level no-op instead of the seed's configured logger.
- `AgentPromptBatch` is reduced to the one method the attributer calls,
  `get_agent_names()`.
- No name starts with an underscore, as in the rest of xmas: `_clean_state` is
  `clean_state`, `self._agent_names` is `self.agent_names`, and so on.

Everything that computes credit is verbatim but for those names: the grouping,
the seeded shuffle, the prompt payload, the response parsing and the clipping.

**Licensing:** neither seed repository carries a license, and this directory
copies from one of them. That is tolerable while xmas is private and Tom owns
both sides. It has to be resolved, or this directory deleted, before xmas is made
public.
