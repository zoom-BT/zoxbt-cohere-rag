from __future__ import annotations

import argparse
import json
import sys

from zoxbt_cohere_rag.eval_run import main as eval_main
from zoxbt_cohere_rag.retrieve import retrieve
from zoxbt_cohere_rag.embed_index import build_index
from zoxbt_cohere_rag.ingest import run_ingest

def cmd_ask(query: str, method: str) -> None:
    hits = retrieve(query, method=method)
    for i, h in enumerate(hits, 1):
        score = h.get("rerank_score", h.get("score", 0))
        print(f"\n--- [{i}] {h['title']} ({h['locale']}) score={score:.4f}")
        print(h["url"])
        print(h["text"][:400] + ("..." if len(h["text"]) > 400 else ""))
       
def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description="zoxbt-cohere-rag - blogfolio RAG for zoxbt.is-a.dev")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="ingest (if needed) + Cohere embed -> data/index/")

    bi = sub.add_parser("build-index", help="Ingest + Cohere embed -> data/index/")
    bi.add_argument("--reingest", action = "store_true")

    ev = sub.add_parser("eval", help="Run eval/questions.jsonl")
    ev.add_argument("--json", action = "store_true")

    ask = sub.add_parser("ask", help="Query the index")
    ask.add_argument("question")
    ask.add_argument(
        "--method", choices=("bm25", "embed", "embed_rerank"), 
        default="embed_rerank",
    )

    args = p.parse_args(argv)
    if args.cmd == "ingest":
        run_ingest()
    elif args.cmd == "build-index":
        build_index(reingest=args.reingest)
    elif args.cmd == "eval":
        if args.json:
            from zoxbt_cohere_rag.eval_run import run_eval

            print(json.dumps(run_eval(), indent=2))
        else:
            eval_main()
    elif args.cmd == "ask":
        cmd_ask(args.question, args.method)

if __name__ == "__main__":
    main(sys.argv[1:])