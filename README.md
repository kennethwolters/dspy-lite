# dspy-lite

DSPy with [litellm](https://github.com/BerriAI/litellm) replaced by [litelm](https://github.com/kennethwolters/litelm) and numpy made optional.

## What this is

- Same DSPy 3.1.3 -- signatures, modules (Predict, CoT, ReAct), optimizers (Bootstrap, MIPRO, SIMBA, GEPA, GRPO)
- `litellm` (100k+ LOC, 70+ transitive deps) replaced by `litelm` (2.6k LOC, 2 deps)
- `numpy` moved from core dep to optional `[embeddings]` extra
- **88% smaller install** (25 MB vs 213 MB, 30 packages vs 72)

## Install

```bash
pip install dspy-lite
```

With provider extras:

```bash
pip install dspy-lite[anthropic]     # Anthropic
pip install dspy-lite[bedrock]       # AWS Bedrock
pip install dspy-lite[embeddings]    # numpy for embeddings
pip install dspy-lite[all]           # everything
```

## Usage

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

predict = dspy.Predict("question -> answer")
result = predict(question="What is DSPy?")
print(result.answer)
```

Everything from DSPy works -- the import is still `import dspy`.

## What changed from DSPy

| | DSPy | dspy-lite |
|---|---|---|
| LM routing | litellm (34 MB, 201k LOC) | litelm (87 KB, 2.6k LOC) |
| numpy | core dep (60 MB) | optional `[embeddings]` |
| Install size | 213 MB, 72 packages | 25 MB, 30 packages |
| API | unchanged | unchanged |

## Documentation

Full DSPy documentation: [dspy.ai](https://dspy.ai)

## License

MIT (inherited from DSPy)
