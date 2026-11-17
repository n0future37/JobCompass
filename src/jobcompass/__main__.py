import argparse
import json

from .settings import ROOT, Settings


def main():
    parser = argparse.ArgumentParser(description="JobCompass RAG")
    parser.add_argument("--env", choices=["dev", "test"], default="dev")
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preprocess")
    pre.add_argument("--output", default="data/processed/jobs_v2")
    pre.add_argument("--selection", help="只处理清单中的raw相对路径；原始文件均保留")
    commands.add_parser("provision")
    commands.add_parser("migrate")
    imp = commands.add_parser("import")
    imp.add_argument("--snapshot", default="data/processed/jobs_v2")
    commands.add_parser("build")
    query = commands.add_parser("query")
    query.add_argument("text")
    query.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    query.add_argument("--city", action="append", default=[])
    query.add_argument("--skill", action="append", default=[])
    query.add_argument("--no-rerank", action="store_true")
    query.add_argument("--no-generate", action="store_true")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    commands.add_parser("evaluate")
    commands.add_parser("audit")
    worker = commands.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = Settings(args.env)
    if args.command == "preprocess":
        from .preprocessing import preprocess

        selection = json.loads((ROOT / args.selection).read_bytes()) if args.selection else None
        result = preprocess(ROOT / "data/raw", ROOT / args.output, allowed_raw_files=selection)
        result = {k: v for k, v in result.items() if k not in {"raw_files", "raw_selection"}}
    elif args.command == "provision":
        from .provision import provision

        result = provision()
    elif args.command == "migrate":
        import os

        from alembic import command
        from alembic.config import Config

        os.environ["JOBCOMPASS_ENV"] = args.env
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        result = {"migration": "head", "environment": args.env}
    elif args.command == "import":
        from .storage import Store

        store = Store(settings)
        try:
            result = store.upsert_import(ROOT / args.snapshot)
        finally:
            store.close()
    elif args.command == "build":
        from .storage import Store
        from .worker import rebuild

        store = Store(settings)
        try:
            result = rebuild(settings, store)
        finally:
            store.close()
    elif args.command == "worker":
        from .worker import run

        result = run(settings, args.once)
    elif args.command == "query":
        from .conditions import SearchRequest
        from .service import RagService

        service = RagService(settings)
        try:
            result = service.query(
                SearchRequest(
                    query=args.text,
                    mode=args.mode,
                    cities=args.city,
                    skills_all=args.skill,
                    rerank=not args.no_rerank,
                    generate=not args.no_generate,
                )
            )
        finally:
            service.close()
    elif args.command == "serve":
        import os

        import uvicorn

        os.environ["JOBCOMPASS_ENV"] = args.env
        uvicorn.run("jobcompass.web:app", host="127.0.0.1", port=args.port, log_level="info")
        return
    elif args.command == "audit":
        from .audit import audit

        result = audit()
    else:
        from .evaluation import evaluate

        result = evaluate(settings)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
