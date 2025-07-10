import argparse
from pathlib import Path

from .pipeline import Pipeline
from .weaviate_store import WeaviateStore


def main():
    parser = argparse.ArgumentParser(description="Run semantic tagging pipeline")
    parser.add_argument("path", type=Path, help="File or directory of transcripts")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
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
    pipeline = Pipeline(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        tags=tag_list,
        tag_file=args.tag_file,
    )

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


if __name__ == "__main__":
    main()
