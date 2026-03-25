"""
Databricks I/O for entity resolution pipeline.

Reads/writes DataFrames from/to Databricks SQL warehouses.

Authentication (checked in order):
    1. OAuth M2M (service principal) — if DATABRICKS_CLIENT_ID and
       DATABRICKS_CLIENT_SECRET are set alongside DATABRICKS_HOST.
       Used in production containers.
    2. Databricks CLI / SDK default auth — falls back to ~/.databrickscfg,
       used for local development.

Required env vars (both modes):
    DATABRICKS_HTTP_PATH     — e.g. /sql/1.0/warehouses/abcdef1234567890

For OAuth M2M (production):
    DATABRICKS_HOST          — e.g. dbc-abc123.cloud.databricks.com
    DATABRICKS_CLIENT_ID     — service principal application (client) ID
    DATABRICKS_CLIENT_SECRET — service principal secret

For CLI auth (local dev):
    Run `databricks configure` or `databricks auth login` first.
    DATABRICKS_HOST is optional — will be read from CLI profile if not set.
"""

import os
import tempfile
import time
from dataclasses import dataclass

import pandas as pd
from databricks import sql as databricks_sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config, oauth_service_principal
from databricks.sql.client import Connection

# Pandas dtype -> Databricks SQL type mapping, since these don't get correctly
# mapped all the time
_DTYPE_MAP = {
    "int64": "BIGINT",
    "int32": "INT",
    "float64": "DOUBLE",
    "float32": "FLOAT",
    "bool": "BOOLEAN",
    "datetime64[ns]": "TIMESTAMP",
    "object": "STRING",
}


def is_databricks_fqn(value: str) -> bool:
    """Check if a string looks like a Databricks fully-qualified table name.

    (catalog.schema.table). Useful for determining whether the input is a CSV
    file or a table.
    """
    if os.sep in value or value.endswith(".csv"):
        return False
    parts = value.split(".")
    return len(parts) == 3 and all(p.strip() for p in parts)


def _strip_url_prefix(url: str) -> str:
    """Remove https:// or http:// prefix from a URL."""
    return url.removeprefix("https://").removeprefix("http://")


def _build_connect_kwargs() -> dict:
    """Resolve auth method and return kwargs for databricks_sql.connect().

    Tries OAuth M2M first (env vars), then falls back to Databricks SDK
    unified auth (CLI profile, Azure CLI, etc.).
    """
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    if not http_path:
        raise ValueError("DATABRICKS_HTTP_PATH env var is required")

    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
    host = os.environ.get("DATABRICKS_HOST", "")

    if client_id and client_secret:
        # Production: OAuth M2M via service principal
        hostname = _strip_url_prefix(host)
        if not hostname:
            raise ValueError(
                "DATABRICKS_HOST is required when using OAuth M2M "
                "(DATABRICKS_CLIENT_ID/SECRET are set)"
            )

        def credential_provider():
            config = Config(
                host=f"https://{hostname}",
                client_id=client_id,
                client_secret=client_secret,
            )
            return oauth_service_principal(config)

        print("Auth: OAuth M2M (service principal)")
        return {
            "server_hostname": hostname,
            "http_path": http_path,
            "credentials_provider": credential_provider,
        }
    else:
        # Local dev: Databricks SDK unified auth (CLI profile, PAT, etc.)
        config = Config(
            host=f"https://{_strip_url_prefix(host)}" if host else None,
        )
        hostname = _strip_url_prefix(config.host)
        print(f"Auth: Databricks CLI / SDK default ({hostname})")
        # credentials_provider must return a HeaderFactory (callable -> dict).
        # config.authenticate is itself a method that returns a dict, so we
        # wrap it in a lambda to match the expected () -> (() -> dict) signature.
        return {
            "server_hostname": hostname,
            "http_path": http_path,
            "credentials_provider": lambda: config.authenticate,
        }


def get_connection(
    max_retries: int = 5,
    retry_delay: int = 10,
) -> Connection:
    """Create a Databricks connection with cold-start retry."""
    connect_kwargs = _build_connect_kwargs()

    for attempt in range(max_retries):
        try:
            connection = databricks_sql.connect(**connect_kwargs)
            print("Databricks connection established")
            return connection
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(
                f"Connection attempt {attempt + 1}/{max_retries} failed: {e}. "
                f"Retrying in {retry_delay}s..."
            )
            time.sleep(retry_delay)
    raise RuntimeError("Unreachable")


@dataclass(frozen=True)
class TableFQN:
    """A fully-qualified Databricks table name (catalog.schema.table)."""

    catalog: str
    schema: str
    table: str

    @property
    def quoted(self) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{self.table}`"


def _parse_fqn(fqn: str) -> TableFQN:
    """Parse catalog.schema.table from a fully-qualified name."""
    parts = fqn.split(".")
    if len(parts) != 3:
        raise ValueError(f"Expected catalog.schema.table, got: {fqn}")
    return TableFQN(catalog=parts[0], schema=parts[1], table=parts[2])


def read_table(fqn: str) -> pd.DataFrame:
    """Read a Databricks table into a pandas DataFrame."""
    t = _parse_fqn(fqn)
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {t.quoted}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        cursor.close()
        df = pd.DataFrame(rows, columns=columns)
        print(f"Read {len(df):,} rows from {fqn}")
        return df
    finally:
        conn.close()


def _df_to_databricks_schema(df: pd.DataFrame) -> str:
    """Convert DataFrame dtypes to a Databricks CREATE TABLE column spec."""
    cols = []
    for col_name, dtype in df.dtypes.items():
        db_type = _DTYPE_MAP.get(str(dtype), "STRING")
        cols.append(f"`{col_name}` {db_type}")
    return ", ".join(cols)


def _get_workspace_client() -> WorkspaceClient:
    """Build a WorkspaceClient that respects the same env vars as the SQL connection."""
    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
    host = os.environ.get("DATABRICKS_HOST", "")

    if client_id and client_secret:
        return WorkspaceClient(
            host=f"https://{_strip_url_prefix(host)}",
            client_id=client_id,
            client_secret=client_secret,
        )
    return WorkspaceClient(
        host=f"https://{_strip_url_prefix(host)}" if host else None,
    )


def write_table(
    df: pd.DataFrame,
    fqn: str,
    overwrite: bool = False,
    staging_volume: str = "matcha_staging",
) -> None:
    """Write a pandas DataFrame to a Databricks table via parquet staging.

    Uploads a parquet file to a Unity Catalog Volume, then uses COPY INTO
    to load it into the target table.
    """
    t = _parse_fqn(fqn)
    schema_spec = _df_to_databricks_schema(df)

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Create or validate the target table
        if overwrite:
            cursor.execute(
                f"CREATE OR REPLACE TABLE {t.quoted} ({schema_spec})"
            )
        else:
            try:
                cursor.execute(
                    f"CREATE TABLE {t.quoted} ({schema_spec})"
                )
            except Exception as e:
                if "already exists" in str(e).lower():
                    raise RuntimeError(
                        f"Table {fqn} already exists. Use --overwrite to replace it."
                    ) from e
                raise
        print(f"Created table {fqn}")

        # Ensure the staging volume exists
        cursor.execute(
            f"CREATE VOLUME IF NOT EXISTS "
            f"`{t.catalog}`.`{t.schema}`.`{staging_volume}`"
        )

        # Upload parquet to a staging Volume, then COPY INTO
        volume_path = f"/Volumes/{t.catalog}/{t.schema}/{staging_volume}/{t.table}.parquet"
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            df.to_parquet(tmp.name, index=False)
            w = _get_workspace_client()
            with open(tmp.name, "rb") as f:
                w.files.upload(volume_path, f, overwrite=True)
            print(f"Uploaded parquet to {volume_path}")

        cursor.execute(
            f"COPY INTO {t.quoted} "
            f"FROM '{volume_path}' "
            f"FILEFORMAT = PARQUET "
            f"COPY_OPTIONS ('mergeSchema' = 'true')"
        )
        print(f"Wrote {len(df):,} rows to {fqn}")

        # Clean up staging file
        try:
            w.files.delete(volume_path)
        except Exception:
            pass  # best-effort cleanup

        cursor.close()
    finally:
        conn.close()
