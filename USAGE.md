# Usage Guide

This repository provides a small semantic tagging pipeline implemented in Python.
It can be used in two ways:

1. **As a command line tool** for quick experiments.
2. **As a library** that can be imported into a larger application.

Float will eventually host the system, so there is no GUI included here.

## Command Line Interface

After installing the dependencies with `pip install -r requirements.txt`,
you can run the pipeline on a directory or single file containing
Markdown or JSON transcripts:

```bash
python -m semantic_tags.cli /path/to/transcripts --tags=tag1,tag2 --batch-size 16 --summary-out summary.json
```

The command now accepts a comma separated list of tags and additional options:

- `--model` – choose the embedding model (alias or path).
- `--list-models` – list recommended models. Can be used without a path to show the model directory.
- `--download-model` – download a model to the local `models/` directory.
- `--model-dir` – override the directory used to look for downloaded models.
- `--show-config` – display the configuration path and settings.
- Configuration values for `model_dir`, `batch_size`, `device` and `weaviate_url` are written to `model_config.json` so they persist across runs.
- `--batch-size` and `--device` – control the embedding step.
- `--tag-file` – load tags from a text file. If omitted the CLI will offer to use `default_tags.txt`.
- `--infer-topics` – automatically infer a tag for each cluster, optionally using OpenAI when an API key is provided.
- `--suggest-missing` – propose additional tags using a simple heuristic or OpenAI when `--openai-key` is supplied.
- `--weaviate-url` – persist the results to a running Weaviate instance.
- `--summary-out` – write a JSON summary of tag counts and inferred cluster labels.
  The summary now includes a `metadata` section recording the embedding model,
  batch size, device, chosen `k` and the Weaviate URL if used.
- `--tree` – print a concise topic summary per file.
- `--train-classifier` – fine tune a simple classifier from labelled nuggets.
- `--openai-key` – API key for OpenAI features (topic inference and missing tag suggestions).
- If omitted, the tool looks for an `OPENAI_API_KEY` environment variable.
- Each processed chunk includes speaker and emotion annotations.
- A progress bar displays embedding progress and the tool prints the model and device in use.

## Using as a Library

You can also import the `Pipeline` class and run it programmatically:

```python
from pathlib import Path
from semantic_tags.pipeline import Pipeline

pipeline = Pipeline()
graph = pipeline.run(Path('/path/to/transcripts'))
print(graph.graph.number_of_nodes(), graph.graph.number_of_edges())
```

This returns a `TagGraph` object from `semantic_tags.graph` that can be further
processed or saved. The library does not require any running server and works
fully offline as long as the embedding model is available.
