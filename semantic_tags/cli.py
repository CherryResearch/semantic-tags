import argparse
import os
from pathlib import Path

from .pipeline import Pipeline
from .weaviate_store import WeaviateStore
from .config import (
    AVAILABLE_MODELS,
    load_config,
    save_config,
    download_model,
    select_model,
    DEFAULT_CONFIG_PATH,
)


def main():
    parser = argparse.ArgumentParser(description="Run semantic tagging pipeline")
    parser.add_argument("path", nargs="?", type=Path, help="File or directory of transcripts")
    parser.add_argument("--model", type=str, help="Model alias or path")
    parser.add_argument("--model-dir", type=Path, help="Directory to store models")
    parser.add_argument("--download-model", type=str, help="Download model and exit")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--show-config", action="store_true", help="Display effective configuration")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--tags", type=str, help="Comma separated list of tags")
    parser.add_argument("--tag-file", type=Path, help="Path to text file with default tags")
    parser.add_argument(
        "--infer-topics", action="store_true", help="Automatically infer topic tags"
    )
    parser.add_argument(
        "--suggest-missing",
        action="store_true",
        help="Suggest additional tags using a heuristic or OpenAI",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str)
    parser.add_argument("--weaviate-url", type=str)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--openai-key", type=str, help="API key for OpenAI features")
    parser.add_argument(
        "--train-classifier",
        action="store_true",
        help="Fine tune a classifier from tagged nuggets",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Print a concise tree of topics per source",
    )
    args = parser.parse_args()

    # Load default API key from environment if available
    if not args.openai_key:
        args.openai_key = os.getenv("OPENAI_API_KEY")

    config = load_config(args.config)
    config_path = Path(os.getenv("SEMANTIC_TAGS_CONFIG", args.config or DEFAULT_CONFIG_PATH))
    if args.model_dir:
        config["model_dir"] = str(args.model_dir)

    if args.show_config:
        print(f"Configuration path: {config_path}")
        for key, val in config.items():
            print(f"{key}: {val}")
        if args.path is None:
            return

    if args.list_models:
        print(f"Model directory: {config['model_dir']}")
        if args.weaviate_url:
            print(f"Weaviate URL: {args.weaviate_url}")
        print("Recommended models:")
        for alias, name in AVAILABLE_MODELS.items():
            print(f"  {alias}: {name} (download with --download-model {alias})")
        print("Other sentence-transformers models are also supported.")
        return

    if args.download_model:
        model_name = select_model(args.download_model)
        path = download_model(model_name, Path(config["model_dir"]))
        print(f"Downloaded {model_name} to {path}")
        save_config(config, args.config)
        return

    if args.path is None:
        parser.error("the following arguments are required: path")

    if args.openai_key and not args.suggest_missing:
        resp = input("Use OpenAI to suggest missing tags? [y/N] ").strip().lower()
        if resp.startswith("y"):
            args.suggest_missing = True

    if not args.tags and args.tag_file is None:
        default_file = Path(__file__).with_name("default_tags.txt")
        if default_file.exists():
            resp = input(f"No tags provided. Use default list at {default_file}? [y/N] ")
            if resp.strip().lower().startswith("y"):
                args.tag_file = default_file

    if args.infer_topics and not args.openai_key:
        key = input(
            "OpenAI API key for topic inference (leave empty to use local heuristics): "
        ).strip()
        if key:
            args.openai_key = key

    tag_list = args.tags.split(",") if args.tags else None
    model_name = select_model(args.model) if args.model else config["default_model"]
    pipeline = Pipeline(
        model_name=model_name,
        batch_size=args.batch_size,
        device=args.device,
        tags=tag_list,
        tag_file=args.tag_file,
        model_dir=Path(config["model_dir"]),
    )

    print(f"Using model {pipeline.model_name} on device {pipeline.device}")
    if args.weaviate_url:
        print(f"Weaviate URL: {args.weaviate_url}")

    store = WeaviateStore(args.weaviate_url) if args.weaviate_url else None

    graph = pipeline.run(
        args.path,
        summary_path=args.summary_out,
        store=store,
        infer_topics=args.infer_topics,
        topic_api_key=args.openai_key if args.infer_topics else None,
    )
    print(
        f"Graph has {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges"
    )

    if args.suggest_missing:
        from .rag import suggest_missing_tags

        if not args.openai_key:
            key = input(
                "OpenAI API key for tag suggestion (leave empty to use heuristics): "
            ).strip()
            if key:
                args.openai_key = key

        try:
            suggestions = suggest_missing_tags(graph, args.openai_key)
            if suggestions:
                print("Possible missing tags:", ", ".join(suggestions))
        except Exception as e:
            print(f"Error during tag suggestion: {e}")

    if args.train_classifier:
        from .classifier import train_tag_classifier

        model = train_tag_classifier(graph)
        if model is None:
            print("Not enough labelled data to train classifier")
        else:
            _, clf = model
            print(f"Trained classifier on {len(clf.classes_)} tags")

    if args.tree:
        summary = graph.conversation_summary()

        def build_tree(summary_dict):
            tree: dict = {}
            for src, info in summary_dict.items():
                parts = Path(src).parts
                d = tree
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d[parts[-1]] = info
            return tree

        def print_tree(d, indent=0):
            for key in sorted(d):
                val = d[key]
                if isinstance(val, dict) and "topics" not in val:
                    print("    " * indent + f"{key}/")
                    print_tree(val, indent + 1)
                else:
                    topics = ", ".join(
                        f"{count} {topic}"
                        for topic, count in sorted(val["topics"].items(), key=lambda x: -x[1])
                    )
                    print("    " * indent + f"{key} ({val['nugget_count']} chunks; {topics})")

        print_tree(build_tree(summary))

    save_config(config, args.config)


if __name__ == "__main__":
    main()

