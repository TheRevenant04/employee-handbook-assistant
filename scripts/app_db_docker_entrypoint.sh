#!/usr/bin/env bash
set -e

echo "=== CUSTOM ENTRYPOINT START ==="
echo "POSTGRES_DB=$POSTGRES_DB"
echo "POSTGRES_USER=$POSTGRES_USER"
echo "PGPASSWORD set: ${PGPASSWORD:+yes}"

# Start postgres in background
echo "Starting postgres..."
docker-entrypoint.sh postgres &

# Wait for postgres to be ready on the default 'postgres' DB
echo "Waiting for postgres to be ready..."
until pg_isready -d postgres -U "$POSTGRES_USER" -q; do
  sleep 1
done
echo "Postgres is ready."

export PGPASSWORD="${PGPASSWORD:-$POSTGRES_PASSWORD}"

# Ensure the target database exists
echo "Ensuring database '$POSTGRES_DB' exists..."

# Try to connect; if it fails because DB doesn't exist, create it
if psql -d "$POSTGRES_DB" -U "$POSTGRES_USER" -c 'SELECT 1;' >/dev/null 2>&1; then
  echo "Database '$POSTGRES_DB' already exists."
else
  echo "Database '$POSTGRES_DB' does not exist; creating it..."
  psql -v ON_ERROR_STOP=1 \
    -d postgres \
    -U "$POSTGRES_USER" \
    -c "CREATE DATABASE \"$POSTGRES_DB\";"
  echo "Database '$POSTGRES_DB' created."
fi

# Run init.sql on every start (against the target DB)
echo "Looking for /docker-entrypoint-initdb.d/init.sql..."
if [ -f /docker-entrypoint-initdb.d/init.sql ]; then
  echo "Running init.sql against DB: $POSTGRES_DB"
  psql -v ON_ERROR_STOP=1 \
    -d "$POSTGRES_DB" \
    -U "$POSTGRES_USER" \
    -f /docker-entrypoint-initdb.d/init.sql
  echo "init.sql finished."
else
  echo "ERROR: /docker-entrypoint-initdb.d/init.sql not found!"
fi

echo "=== CUSTOM ENTRYPOINT END ==="

# Keep container running
wait