import argparse
from pathlib import Path

from .pipeline import Pipeline
from .weaviate_store import WeaviateStore


def main():
    parser = argparse.ArgumentParser(description="Run semantic tagging pipeline")
    parser.add_argument("path", type=Path, help="File or directory of transcripts")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--tags", type=str, help="Comma separated list of tags")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str)
    parser.add_argument("--weaviate-url", type=str)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--openai-key", type=str, help="API key to suggest additional tags")
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

    tag_list = args.tags.split(",") if args.tags else None
    pipeline = Pipeline(
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        tags=tag_list,
    )

    store = WeaviateStore(args.weaviate_url) if args.weaviate_url else None

    graph = pipeline.run(
        args.path,
        summary_path=args.summary_out,
        store=store,
    )
    print(f"Graph has {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges")

    if args.openai_key:
        from .rag import suggest_missing_tags

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
                        f"{count} {topic}" for topic, count in sorted(val["topics"].items(), key=lambda x: -x[1])
                    )
                    print(
                        "    " * indent
                        + f"{key} ({val['nugget_count']} chunks; {topics})"
                    )

        print_tree(build_tree(summary))


if __name__ == "__main__":
    main()
