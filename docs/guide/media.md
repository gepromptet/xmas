# Pictures, documents and other files

A value can also be an `image`, `audio`, a `document`, a `table` or a `file`, or a list of
them. Such a value is a reference, a `Ref`: where the file is, its media type, its size
and a checksum. The bytes stay in the file until a model needs them.

Here a reviewer looks at the pages of a deck, and a step keeps the first page as the
cover. The pages come in as files on disk:

```python
from pathlib import Path
from xmas import MAS, APILLM, Agent, Image, Step
from xmas.tools import tool


@tool(writes={"cover": "image"})
def keep_cover(pages: list[Image]) -> Path:
    """Keep the first page as the deck's cover."""
    return Path(pages[0].location)


reviewer = Agent("reviewer", prompt="Review these slides.\n{pages}", inputs=["pages"], output="review")
vision = APILLM(model="vision-model", accepts=("text", "image"))
deck = MAS(
    [reviewer],
    steps=[Step("cover", uses="keep_cover")],
    inputs={"pages": "list of image"},
    outputs={"review": "text", "cover": "image"},
    tools=[keep_cover],
    llm=vision,
)

trace = deck.run_state({"pages": [Path("slide-1.png"), Path("slide-2.png")]})
print(trace["reviewer"][0].rendered_prompt)
print(trace.outputs["cover"].location)  # a copy in the store, named by its checksum
```

The model gets each page as a picture between the parts of the prompt's text. Each item
of a list gets a label, counted from 0 like `item`, so the model can name a page. The
record shows a marker in place of each picture, never its bytes:

```text
Review these slides.

[pages 0] [image slide-1.png, image/png, 182 KB]
[pages 1] [image slide-2.png, image/png, 96 KB]
```

`run_state` turns each `Path` into a reference to your file. The checksum is checked on
every read, so a file that changes during the run is refused. A step that returns a
`Path` writes a copy in the store instead, so a step that rewrites one file per item
leaves earlier values alone. A step may also return a `Ref`. Bytes are refused, since
`Ref.from_bytes(data, "image/png")` stores them. The store is a folder (see
[Configuration](bundles.md#configuration)), and the same bytes are stored once. In a trace, a reference is a small
dictionary, so `trace.to_dict()` stays plain JSON.

Each model states what it accepts and produces. On a model that takes only text, such
as the [quickstart](https://github.com/gepromptet/xmas#readme)'s `llm`, the check refuses the reviewer before any call with a
`ValueError`:

```python
MAS([reviewer], inputs={"pages": "list of image"}, llm=llm)
```

```text
The system does not check (1 problem):
- agent 'reviewer' reads image 'pages', but the default model accepts only text; give it accepts = ("text", "image") or convert it first with a step
```

What a model takes comes from the first of three places: the system's `models` entry for
the model's name, the model's own `accepts` and `produces`, or else text only. In a spec,
the `models` section names each model and says what it accepts and produces:

```yaml
models:
  vision-model: {accepts: [image]}
  painter-model: {produces: [image]}
```

`MAS(..., models={"vision-model": {"accepts": ["image"]}})` says the same in Python.
An entry may also set model parameters (see [Bundles](bundles.md)). Any other key is refused,
and so is a name no agent binds.
The rest of the rules:

- A model sees an image or audio as a content part. `APILLM` sends parts; another
  model class needs `generate_parts_with_meta` and `full_response = True`.
- A model never reads a document or a table directly. A step turns it into text or rows
  first, so the conversion shows in the graph. `xmas.convert` has two such functions:
  `document_text` reads text, Markdown and HTML, and `table_rows` reads CSV and TSV. For
  a PDF, write a step of your own.
- An agent with `output_type="image"` answers with a picture, on a model that produces
  images. The picture goes to the store, and the value is its reference. Such an agent
  takes no tools.
- An MCP tool may write a picture too: `MCPTools(..., writes={"paint": {"png": "image"}})`.
  A remote tool reads no media, so the check refuses a step that passes it one.
- A tool an agent calls reads no media either: the model would make up the reference.
  Run such a function as a step.
