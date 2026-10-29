# Drawing a system

`mas.describe()` returns the system as plain data, here the system of
[Conditions and loops](conditions-and-loops.md). Each node has its kind, model, typed
reads and writes, and whether it is detached. Each edge has its kind and condition. The
start and end wiring that xmas adds itself is marked `implicit`:

```python
picture = mas.describe()
print([(node["name"], node["kind"]) for node in picture["nodes"]])
# [('writer', 'agent'), ('count', 'code'), ('judge', 'agent'), ('publish', 'code'), ('sorry', 'code')]
print([edge["kind"] for edge in picture["edges"] if not edge.get("implicit")])
# ['plain', 'plain', 'while', 'switch', 'else', 'error']
mas.agents["judge"].prompt += "\nBe strict."
print(mas.describe()["id"] == picture["id"])  # True
```

The `id` is a hash of the description, and prompts are not part of it. So editing a
prompt keeps the id, and changing the wiring changes it. Any picture of the system comes
from the graph that runs, never from a hand-drawn copy.
