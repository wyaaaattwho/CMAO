# Data Layout

Place local dataset mirrors, caches, or small smoke-test files here.

Suggested conventions:

- `data/raw/`: untouched dataset dumps
- `data/processed/`: normalized JSON or JSONL files
- `data/cache/`: download cache if you do not want to rely on the global Hugging Face cache

The current implementation can load either:

- Hugging Face datasets by logical name such as `gsm8k`
- Local `.json` or `.jsonl` files via the dataset `path` field in the experiment config

