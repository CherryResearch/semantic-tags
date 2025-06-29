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
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Print nuggets grouped by source in a tree",
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

    if args.tree:
        from collections import defaultdict

        nuggets_by_source = defaultdict(list)
        for node, data in graph.graph.nodes(data=True):
            if data.get("type") == "nugget":
                nuggets_by_source[data["source"]].append(data["text"])

        def build_tree(mapping):
            tree = {}
            for src, nuggets in mapping.items():
                parts = Path(src).parts
                d = tree
                for part in parts[:-1]:
                    d = d.setdefault(part, {})
                d.setdefault(parts[-1], []).extend(nuggets)
            return tree

        def print_tree(d, indent=0):
            for key in sorted(d):
                val = d[key]
                if isinstance(val, dict):
                    print("    " * indent + f"{key}/")
                    print_tree(val, indent + 1)
                else:
                    print("    " * indent + key)
                    for n in val:
                        preview = n.strip().replace("\n", " ")
                        if len(preview) > 40:
                            preview = preview[:40] + "..."
                        print("    " * (indent + 1) + preview)

        tree = build_tree(nuggets_by_source)
        print_tree(tree)


if __name__ == "__main__":
    main()
