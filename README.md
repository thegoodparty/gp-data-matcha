# Matcha: Multi-Source Entity Resolution

Splink-based probabilistic record linkage for matching records across
multiple data sources. Supports two entity types:

- **Candidacy** — matches candidacy stage records (BR, TS, DDHQ) by person + office + election date
- **Elected Official** — matches elected official records (BR, TS) by person + office (no election date)

The pipeline supports any number of sources discovered dynamically from the
`source_name` column. Entity-specific configuration (comparisons, blocking
rules, filters) lives in `scripts/configs/`.

## Quick start

### Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) for local development
- Docker for container-based runs
- [GitHub CLI](https://cli.github.com/) (`gh`) for pulling pre-built images

### Local (uv)

```bash
uv sync

# Candidacy matching (default)
uv run python -m scripts.cli match --entity-type candidacy --input input.csv

# Elected official matching
uv run python -m scripts.cli match --entity-type elected_official --input input.csv
```

### Docker (pre-built image)

Authenticate Docker with GHCR using the GitHub CLI:

```bash
# One-time: ensure gh has the packages scope
gh auth refresh -s read:packages

# Log Docker into GHCR
echo $(gh auth token) | docker login ghcr.io -u $(gh api user --jq .login) --password-stdin
```

Pull and run the latest image:

```bash
docker pull ghcr.io/thegoodparty/gp-data-matcha:latest

# Show help
docker run ghcr.io/thegoodparty/gp-data-matcha:latest match --help

# Run with a local CSV
docker run \
  -m 8g --cpus 4 \
  -v ~/path/to/input.csv:/app/data/input.csv \
  ghcr.io/thegoodparty/gp-data-matcha:latest \
  match --input /app/data/input.csv --output-dir /app/out
```

PR builds are tagged `pr-<number>` (e.g. `ghcr.io/thegoodparty/gp-data-matcha:pr-2`).

### Docker (local build)

```bash
docker build -t gp-data-matcha .
docker run gp-data-matcha match --help
```

### Databricks input + output

```bash
export DATABRICKS_HOST=dbc-abc123.cloud.databricks.com
export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/abcdef1234567890
export DATABRICKS_CLIENT_ID=<service-principal-client-id>
export DATABRICKS_CLIENT_SECRET=<service-principal-secret>

uv run python -m scripts.cli match \
  --entity-type candidacy \
  --input goodparty_data_catalog.dbt_dball.int__er_prematch_candidacy_stages \
  --output-cluster-table goodparty_data_catalog.er_source.er_clustered_candidacies \
  --output-pairwise-table goodparty_data_catalog.er_source.er_pairwise_predictions \
  --overwrite

# Elected officials
uv run python -m scripts.cli match \
  --entity-type elected_official \
  --input goodparty_data_catalog.dbt_dball.int__er_prematch_elected_officials \
  --output-cluster-table goodparty_data_catalog.er_source.er_clustered_elected_officials \
  --output-pairwise-table goodparty_data_catalog.er_source.er_pairwise_elected_officials \
  --overwrite
```

## CLI reference

```
Usage: cli.py match [OPTIONS]

Options:
  --entity-type [candidacy|elected_official]  Entity type to match (default: candidacy).
  --input TEXT                  Path to prematch CSV or Databricks FQN (catalog.schema.table). Required.
  --output-dir DIRECTORY        Directory for local results. Default: results/<entity-type>/
  --output-cluster-table TEXT   Databricks FQN to upload clustered results (catalog.schema.table).
  --output-pairwise-table TEXT  Databricks FQN to upload pairwise predictions (catalog.schema.table).
  --overwrite                   Overwrite existing Databricks output tables.
```

### Audit subcommands

```bash
uv run python -m scripts.cli audit summary --entity-type candidacy --results-dir results/candidacy/
uv run python -m scripts.cli audit low-confidence --entity-type candidacy --results-dir results/candidacy/ --sample 20
uv run python -m scripts.cli audit false-negatives --entity-type elected_official --results-dir results/elected_official/
```

When `--run-audit` is enabled (default), all three audits run automatically after
matching and log results. The standalone commands above are available for manual
re-runs against saved CSV results.

### Auditing with Claude Code

The `/audit-er-results` skill (`.claude/skills/audit-er-results/skill.md`)
provides a guided workflow for auditing match quality with Claude Code. It
runs through summary stats, low-confidence pair review, false negative
analysis, regression checking, and produces actionable recommendations. Use it
after a match run or when evaluating changes to blocking rules, comparisons,
or post-prediction filters:

```
/audit-er-results
```

**Input:** CSV file or Databricks table from `int__er_prematch_candidacy_stages` or `int__er_prematch_elected_officials`
**Output** (in `results/<entity-type>/`):
- `pairwise_predictions.csv` — all scored candidate pairs
- `clustered_candidacies.csv` or `clustered_elected_officials.csv` — all records with cluster assignments
- `match_weights_chart.html` — Splink match weight visualization
- `m_u_parameters_chart.html` — learned m/u probability visualization
- `audit_summary.csv` — match coverage stats per source
- `audit_low_confidence.csv` — most ambiguous pairs for review
- `audit_false_negatives.csv` — plausible matches the model missed

## Authentication

Two auth modes are supported, resolved automatically:

### Local development (Databricks CLI)

If you've authenticated with the Databricks CLI (`databricks configure` or
`databricks auth login`), the pipeline picks up your profile automatically.
Only one env var is needed:

```bash
export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/abcdef1234567890
```

`DATABRICKS_HOST` is optional — it will be read from your CLI profile if not set.

### Production (OAuth M2M service principal)

Used in Docker containers and Airflow. All four env vars are required:

| Variable | Description |
|----------|-------------|
| `DATABRICKS_HOST` | Databricks workspace hostname |
| `DATABRICKS_HTTP_PATH` | SQL warehouse HTTP path |
| `DATABRICKS_CLIENT_ID` | Service principal client ID |
| `DATABRICKS_CLIENT_SECRET` | Service principal secret |

The pipeline detects which mode to use based on whether `DATABRICKS_CLIENT_ID`
and `DATABRICKS_CLIENT_SECRET` are set.

## Design: candidate-level vs race-level attributes

Attributes are divided into two categories based on how they contribute to
scoring:

**Candidate-level attributes** (strongest Splink comparisons — drive the match
score):
- `last_name`, `first_name`, `party`, `email`, `phone`

**Race/election-level attributes** (used in both blocking rules and as Splink
comparisons, but guarded by post-prediction filters to prevent false positives):
- `state`, `official_office_name`, `election_date`, `district_identifier`
- `br_race_id` — used in blocking only

**Additional retained columns** (carried through for filtering and output but
not used as comparisons):
- `office_type`, `candidate_office`, `office_level`,
  `district_raw`, `seat_name`, `election_stage`, `br_candidacy_id`

### Why race-level attributes need post-prediction guards

Multiple candidates run in the same race. Race-level attributes like
`official_office_name`, `election_date`, and `state` produce positive Bayes
factors for *any* pair of candidates in the same race — which can overwhelm
name-mismatch penalties. The post-prediction filter (described below) catches
these cases by requiring name agreement and office/race consistency.

## How it works

The script uses [Splink 4](https://moj-analytical-services.github.io/splink/)
in `link_only` mode (cross-source matching, no within-source dedup) with
DuckDB as the backend. Sources are discovered dynamically from
`df["source_name"].unique()` and passed as a list to the Splink `Linker`.

### Preprocessing

Most data cleaning is handled upstream in the dbt prematch model
(`int__er_prematch_candidacy_stages`). The Python script performs only:

- **First name nicknames:** the dbt model maps each first name to an alias
  array via the `nicknames` seed (e.g. robert -> [robert, bob, bobby, rob,
  bert, ...]). The array always includes the original first name. The script
  parses these JSON arrays so Splink's `ArrayIntersectLevel` can check for
  overlap, recognizing "robert" and "bob" as potential matches without
  requiring exact string similarity.
- **Nulls:** literal `"null"` strings, empty strings, and `NaN` are all
  converted to `None` so Splink treats them as missing data

The following are handled in dbt (not in the Python script):
- **Names:** lowercased and trimmed
- **`official_office_name`:** lowercased and trimmed
- **`district_identifier`:** cast to int (normalizes leading zeros)
- **`br_race_id`:** cast to int (non-integer values like
  `ts_found_race_net_new` become null so the blocking rule only fires for
  records with a shared race ID)

### Blocking rules (which pairs to compare)

Blocking rules determine which record pairs are generated for scoring. Splink
unions the pairs from each rule, deduplicating. All rules enforce race-level
constraints so that only candidates plausibly in the same race are compared.

| Order | Rule | Purpose |
|-------|------|---------|
| 1 | `br_race_id` (exact) | High-cardinality first pass. Pairs records in the same race. Covers the majority of matches. |
| 2 | `state + election_date + official_office_name (JW >= 0.88) + last_name` (exact) | Catches cross-source office formatting differences for records without a shared race ID. |
| 3 | `state + last_name + election_date` (exact) | Broad catch-all for net-new records and cases not covered by race ID or office name. |
| 4 | `state + election_date + official_office_name (JW >= 0.88) + last_name (JW >= 0.88)` | Catches last name typos/variants across sources with different office formatting. |
| 5 | `phone` (exact) | Contact-info matches where names may differ. |
| 6 | `email` (exact) | Contact-info matches where names may differ. |

Rules 2 and 4 use DuckDB's `jaro_winkler_similarity` function via Splink's
`CustomRule` for fuzzy blocking.

### Comparisons (how pairs are scored)

All comparisons contribute Bayes factors to the match score:

| Column | Type | Levels | Notes |
|--------|------|--------|-------|
| `last_name` | Jaro-Winkler | exact, >= 0.95, >= 0.88, else | Term frequency adjusted (down-weights common surnames) |
| `first_name` | Custom | exact -> nickname -> JW >= 0.92 -> else | Nickname match via alias array intersection; TF adjusted on exact |
| `party` | Exact | match, else | |
| `email` | Exact | match, else | |
| `phone` | Exact | match, else | |
| `state` | Exact | match, else | |
| `election_date` | Exact | match, else | |
| `official_office_name` | Jaro-Winkler | exact, >= 0.95, >= 0.88, >= 0.75, else | 0.75 tier catches cross-source formatting |
| `district_identifier` | Exact | match, else | Numeric district; provides positive/negative Bayesian evidence |

### Training

Four EM passes with different blocking ensure all comparison columns get
trained. Each pass blocks on one or more columns (fixing them) and estimates
m probabilities for the rest:

1. Block on `last_name + state + election_date` -> trains first_name, party, email, phone, official_office_name, district_identifier
2. Block on `first_name` -> trains last_name, party, email, phone, state, election_date, official_office_name, district_identifier
3. Block on `email` -> trains last_name, first_name, party, phone, state, election_date, official_office_name, district_identifier
4. Block on `state + election_date` -> trains last_name, first_name, party, email, phone, official_office_name, district_identifier

u probabilities are estimated via random sampling (5M pairs) before EM.

### Post-prediction filters

After Splink scores all blocked pairs, three filters ensure we only cluster
true candidacy matches (same person + same office + same election):

1. **Person identity filter** — requires last name agreement (gamma > 0) AND
   first name agreement OR email/phone match. Removes same-race,
   different-candidate pairs.

2. **Race-level filter** — requires either `official_office_name` JW >= 0.75
   (gamma > 0) **or** a shared meaningful locality token between the two office
   names. The JW threshold catches most cross-source formatting differences
   (e.g. "durham school board" vs "durham county board of education", JW 0.87).
   The locality-token fallback handles cases where the overall JW is low but
   both names reference the same place — e.g. "mayor of brodhead" vs
   "brodhead city mayor" (JW 0.557, but shared token "brodhead"). Common
   structural words (city, county, board, council, district, school, etc.)
   are excluded from the token overlap check so that only place names and
   other distinctive tokens count.

3. **Race ID filter** — excludes pairs where both sides have a known integer
   `br_race_id` and they differ, **unless** the office names match well
   (JW >= 0.88). Sources sometimes assign different race IDs to the same
   race, so a strong office name match overrides the race ID disagreement.

### Thresholds

- **Prediction threshold: 0.01** — low threshold to capture all plausible pairs
  for the post-prediction filters to evaluate
- **Clustering threshold: 0.95** — high confidence required to cluster, since
  the unit of matching is a *candidacy* (person + office + election date), not
  just a person

## Edge cases this handles

### Last name typos across sources

The fuzzy last name blocking rule (JW >= 0.88) ensures these pairs are
generated even when names don't match exactly:

| BR record | TS record | Match prob |
|-----------|-----------|------------|
| phillip **whitaker** (fort smith school board - zone 1) | phillip **whiteaker** (fort smith public school district zone 1) | 0.92 |
| joe **montelone** (green park city mayor) | joe **monteleone** (green park city mayor) | 0.72 |
| bob **feidler** (st. croix county board - dist 9) | bob **fiedler** (chenequa village board) | 0.83 |
| amanda **fuerst** (wauwatosa city council - dist 10) | amanda **fuers** (wauwatosa city council - dist 10) | 0.84 |
| emily **bassham** (mountainburg school board - zone 2) | emily **basham** (mountainburg school district, zone 2) | 0.88 |

### Cross-source office name formatting

Sources often format the same office differently. The fuzzy office blocking
rule (JW >= 0.88) handles most cases. The 0.75 JW tier in the comparison
catches reformatted office names that fall below 0.88. For extreme formatting
differences (JW < 0.75), the locality-token fallback catches pairs that share
a place name:

| Source L | Source R | JW | Mechanism |
|----------|----------|-----|-----------|
| `fort smith school board - zone 1` (BR) | `fort smith public school district zone 1` (TS) | 0.89 | Office JW >= 0.88 |
| `durham school board - district 4` (BR) | `durham county board of education district 04` (TS) | 0.87 | Office JW >= 0.75 |
| `lake mills city council - district 1` (BR) | `city of lake mills council member- district 1` (TS) | 0.79 | Office JW >= 0.75 |
| `city of racine alderperson 2` (DDHQ) | `racine city council - district 2` (TS) | 0.66 | Locality token "racine" |
| `mayor of brodhead` (DDHQ) | `brodhead city mayor` (TS) | 0.56 | Locality token "brodhead" |
| `city of seminole councilmember 3` (DDHQ) | `seminole city council - ward 3` (TS) | 0.64 | Locality token "seminole" |

### First name nicknames

The alias array intersection catches nickname matches that string similarity
would miss:

| BR name | TS name | Mechanism |
|---------|---------|-----------|
| robert smith | bob smith | alias arrays both contain "bob" and "robert" |
| william jones | bill jones | alias intersection |
| james wilson | jim wilson | alias intersection |

### Same person, different office (correctly separated)

The race-level filter prevents matching a person who runs for two different
offices (e.g. mayor and city council) in the same city:

| Candidate A | Candidate B | Matched? |
|-------------|-------------|----------|
| dean isgrigg, gerald city council - ward 2 | dean isgrigg, gerald city mayor | No (City Council != Mayor) |

Note: candidates running for multiple offices in the same election can still
end up in the same cluster if intermediate pairs chain them together. For
example, John Muraski's howard village board and howard village president
records are clustered together via transitive links through cross-source pairs.

### Same race, different candidates (correctly separated)

Two different candidates running in the same race share office, state, date,
and district — but the person identity filter separates them:

| Candidate A | Candidate B | Matched? |
|-------------|-------------|----------|
| joel straub, marathon county board dist 15 | timothy sondelski, marathon county board dist 25 | No (different names) |
| clark rinehart, raleigh city council | sana siddiqui, raleigh city council | No (different names) |

### Known false negatives

A small number of true matches are systematically missed:

1. **Office names with no shared locality token and JW < 0.75 (~2-3 pairs):**
   The race-level filter requires either JW >= 0.75 or a shared meaningful
   locality token. A few true matches fall through when the source uses a
   county name while the other uses a city name for the same jurisdiction
   (e.g. "louisville metro council" vs "jefferson-dist 9 legislative council"
   — Louisville is in Jefferson County, but neither name contains the other's
   locality token). State names used as locality tokens can also cause a small
   number of false positives (e.g. "university of nebraska board of regents"
   matching "nebraska member of state board of education" via shared
   "nebraska").

2. **Uncommon nicknames not in the nicknames seed (~60 pairs sharing
   `br_race_id` + `last_name`):** The nickname alias table doesn't cover
   informal or uncommon variants. Examples:
   - `barb` / `barbara`, `samara` / `sammie`, `keisha` / `lakeisha`
   - `a.j.` / `a.` (initial/period handling)
   - `fee fee` / `iphenia`, `clutch` / `claude` (exotic nicknames)

## Testing

### Unit tests

```bash
uv run pytest tests/ -v
```

### Integration tests (requires Databricks)

Integration tests run the full CLI pipeline against a live Databricks SQL
warehouse. They are automatically skipped when `DATABRICKS_HTTP_PATH` is not
set.

```bash
# Local dev (Databricks CLI auth)
export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/abcdef1234567890

# CI / production (OAuth M2M)
export DATABRICKS_HOST=dbc-abc123.cloud.databricks.com
export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/abcdef1234567890
export DATABRICKS_CLIENT_ID=<service-principal-client-id>
export DATABRICKS_CLIENT_SECRET=<service-principal-secret>

uv run pytest tests/ -m integration -v
```

The integration fixture creates an ephemeral Databricks schema, uploads the
dummy test data, runs the match pipeline, and tears down the schema on
completion.

## CI/CD

The `.github/workflows/er_container.yml` workflow builds multi-arch (amd64 + arm64) images:
- **On PR:** builds and pushes `ghcr.io/thegoodparty/gp-data-matcha:pr-<number>`
- **On push to main:** pushes `:latest` and `:<sha>` tags
- **On PR close:** cleans up the PR-specific image tag
