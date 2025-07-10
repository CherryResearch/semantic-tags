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

- `--model` – choose the embedding model.
- `--batch-size` and `--device` – control the embedding step.
- `--weaviate-url` – persist the results to a running Weaviate instance.
- `--summary-out` – write a JSON summary of tag counts and inferred cluster labels.
- `--tree` – print a concise topic summary per file.
- `--train-classifier` – fine tune a simple classifier from labelled nuggets.

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
