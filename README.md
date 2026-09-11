# dspy-lite

DSPy with [litellm](https://github.com/BerriAI/litellm) replaced by [litelm](https://github.com/kennethwolters/litelm) and numpy made optional.

## What this is

- Same DSPy 3.2.1 -- signatures, modules (Predict, CoT, ReAct), optimizers (Bootstrap, MIPRO, SIMBA, GEPA, GRPO)
- `litellm` (100k+ LOC, 70+ transitive deps) replaced by `litelm` (2.6k LOC, 2 deps)
- `numpy` moved from core dep to optional `[embeddings]` extra
- **83% smaller clean install** (24 MB vs 144 MB, 36 packages vs 67)
- Unaffected by the [litellm supply chain compromise](https://github.com/kennethwolters/dspy-lite/issues/3) (CVE-2026-33634) — zero litellm code or dependencies

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

### Loading images, audio, and files

Resource validation never reads the filesystem or network implicitly. Load resources explicitly:

```python
image = dspy.Image.from_path("./image.png")
image = dspy.Image.from_url("https://example.com/image.png")  # explicit network request
image_reference = dspy.Image("https://example.com/image.png")  # provider-fetched URL; no local request

audio = dspy.Audio.from_path("./audio.wav")
file = dspy.File.from_path("./document.pdf")
```

Applications must validate or allowlist untrusted URLs before calling `from_url()`; explicit downloads do not provide SSRF filtering.

### Disk-cache safety

Disk caches use restricted pickle deserialization by default. Custom cached classes must be registered with `dspy.configure_cache(safe_types=[...])`. Setting `restrict_pickle=False` restores unrestricted pickle compatibility, emits a warning, and must only be used when the cache directory is fully trusted. Unsafe or incompatible legacy entries are treated as cache misses rather than retried with unrestricted deserialization.

## What changed from DSPy

| | DSPy | dspy-lite |
|---|---|---|
| LM routing | litellm (34 MB, 201k LOC) | litelm (87 KB, 2.6k LOC) |
| numpy | core dep (60 MB) | optional `[embeddings]` |
| Clean install size | 144 MB, 67 packages | 24 MB, 36 packages |
| API | unchanged | unchanged |

## Documentation

Full DSPy documentation: [dspy.ai](https://dspy.ai)

## Development disclosure

This project is maintained with AI-assisted coding. Through 2026-05-14, I used Claude Code with Claude Opus 4.6/4.7. As of 2026-05-14, I use pi with GPT-5.5. The human maintainer remains responsible for what is merged and released.

## License

MIT (inherited from DSPy)
