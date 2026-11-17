from alembic import context

from jobcompass import career, identity, resumes, tasks  # noqa: F401
from jobcompass.settings import Settings
from jobcompass.storage import engine, metadata


def run():
    e = engine(Settings())
    try:
        with e.connect() as c:
            context.configure(connection=c, target_metadata=metadata, compare_type=True)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        e.dispose()


run()
