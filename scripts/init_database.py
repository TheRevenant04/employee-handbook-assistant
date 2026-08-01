"""Initialize database schemas and tables on container startup.

Creates the target database if it doesn't exist, then reads and
executes the SQL DDL from database/init.sql.
"""

import logging
import os
import time
from pathlib import Path

import psycopg
from psycopg import sql, OperationalError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INIT_SQL_PATH = Path(os.getenv("INIT_SQL_PATH", "database/init.sql"))
TARGET_DB = os.getenv("PGDATABASE", "employee_handbook")

def _validate_env():
    """Validate required environment variables."""
    required_vars = ["PGUSER", "PGPASSWORD", "PGHOST", "PGPORT"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

def _ensure_database():
    """Create target database if it doesn't exist."""
    conninfo = psycopg.conninfo.make_conninfo(
        dbname="postgres",
        user=os.getenv("PGUSER", "user"),
        password=os.getenv("PGPASSWORD", "password"),
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
    )
    with psycopg.connect(conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (TARGET_DB,)
            )
            if cur.fetchone() is None:
                logger.info("Creating database %s...", TARGET_DB)
                cur.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TARGET_DB))
                )
                logger.info("Database %s created.", TARGET_DB)
            else:
                logger.debug("Database %s already exists.", TARGET_DB)

def _execute_init_sql():
    """Connect to target database and run init.sql with retries."""
    if not INIT_SQL_PATH.exists():
        logger.error("Init SQL file not found at %s", INIT_SQL_PATH)
        raise FileNotFoundError(f"Init SQL file not found at {INIT_SQL_PATH}")
    
    sql_text = INIT_SQL_PATH.read_text(encoding="utf-8")
    if not sql_text.strip():
        logger.warning("Init SQL file is empty at %s, skipping", INIT_SQL_PATH)
        return

    conninfo = psycopg.conninfo.make_conninfo(
        dbname=TARGET_DB,
        user=os.getenv("PGUSER", "user"),
        password=os.getenv("PGPASSWORD", "password"),
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
    )
    
    logger.info("Initializing schemas and tables from %s...", INIT_SQL_PATH)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with psycopg.connect(conninfo, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql_text)
            logger.info("Database initialization complete.")
            return
        except OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning("Connection attempt %d failed: %s. Retrying in 2s...", attempt + 1, e)
                time.sleep(2)
            else:
                logger.error("Failed to connect after %d attempts", max_retries)
                raise

def main():
    _validate_env()
    
    if not INIT_SQL_PATH.exists():
        logger.warning("Init SQL file not found at %s, skipping", INIT_SQL_PATH)
        return

    _ensure_database()
    _execute_init_sql()

if __name__ == "__main__":
    main()