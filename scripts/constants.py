# scripts/constants.py
"""Shared constants for entity resolution configs."""

OFFICE_STOP_WORDS = (
    "'city','of','the','county','board','council','school','district',"
    "'mayor','alderperson','trustee','at','large','zone','ward','seat',"
    "'position','commission','precinct','town','village','member',"
    "'councilmember','supervisor','supervisors','commissioner','judge',"
    "'branch','education','unified','public','elementary','consolidated',"
    "'central','special','independent','office','clerk','treasurer',"
    "'coroner','sheriff','magistrate','property','value','administrator',"
    "'emergency','services','director','justice','peace','representative',"
    "'house','representatives','legislature','legislative','metro',"
    "'president','attorney','executive','municipal','assessor','auditor',"
    "'recorder','register','surveyor','constable','marshal','comptroller',"
    "'controller','prosecutor','councilor','councilman','councilwoman',"
    "'alderman','alderwoman','selectman','selectperson','freeholder',"
    "'and','for','no.','odd','unexpired'"
)


def _office_locality_tokens(side: str) -> str:
    """DuckDB expression for the distinct meaningful (locality) tokens of an
    office name on one side of a pair. Drops stop words, single chars, and
    pure-digit tokens, leaving the locality/distinguishing tokens.
    """
    return (
        "list_distinct(list_filter("
        f"string_split(lower(official_office_name_{side}), ' '), "
        "x -> len(x) > 1 "
        f"AND NOT list_contains([{OFFICE_STOP_WORDS}], x) "
        "AND NOT regexp_matches(x, '^\\d+$')))"
    )


# Shared post-prediction filter: requires name + identity signal + office overlap.
# Each config can extend this with entity-specific clauses.
BASE_POST_PREDICTION_FILTER = f"""
    gamma_last_name > 0
      AND (gamma_first_name > 0 OR gamma_email > 0 OR gamma_phone > 0)
      AND (
        gamma_official_office_name > 0
        OR list_has_any(
          list_filter(
            string_split(lower(official_office_name_l), ' '),
            x -> len(x) > 1
              AND NOT list_contains([{OFFICE_STOP_WORDS}], x)
              AND NOT regexp_matches(x, '^\\d+$')
          ),
          list_filter(
            string_split(lower(official_office_name_r), ' '),
            x -> len(x) > 1
              AND NOT list_contains([{OFFICE_STOP_WORDS}], x)
              AND NOT regexp_matches(x, '^\\d+$')
          )
        )
      )
"""

# EO-specific post-prediction filter: adds contact-info bypass and office_type
# fallback for cross-source office title synonyms. Contact-confirmed pairs
# (email or phone match) skip office checks entirely since identity is established.
# Does NOT include the candidacy-specific br_race_id guard.
EO_POST_PREDICTION_FILTER = f"""
    gamma_last_name > 0
      AND (gamma_first_name > 0 OR gamma_email > 0 OR gamma_phone > 0)
      AND (
        gamma_email > 0
        OR gamma_phone > 0
        OR gamma_official_office_name > 0
        OR list_has_any(
          list_filter(
            string_split(lower(official_office_name_l), ' '),
            x -> len(x) > 1
              AND NOT list_contains([{OFFICE_STOP_WORDS}], x)
              AND NOT regexp_matches(x, '^\\d+$')
          ),
          list_filter(
            string_split(lower(official_office_name_r), ' '),
            x -> len(x) > 1
              AND NOT list_contains([{OFFICE_STOP_WORDS}], x)
              AND NOT regexp_matches(x, '^\\d+$')
          )
        )
        OR gamma_office_type > 0
        OR gamma_ballotready_position_id > 0
      )
"""

# Race-level post-prediction filter for election_stage ER. No person fields
# (no first_name/last_name/email/phone), so race identity must be carried
# entirely by geography + office + election cycle. A pair is kept only when it
# agrees on:
#   - state, election_date, election_stage. An election stage is a distinct
#     entity: a primary and a general for the same office must not cluster.
#   - office identity: a near-exact full office name (>=0.95 JW tier), OR the
#     same normalized candidate_office AND a shared locality token. Requiring
#     candidate_office stops different offices in one county ("X county clerk"
#     vs "X county mayor") from merging on the shared "X" locality token, and
#     requiring the locality token stops same-office different-locality races
#     ("nelson village president" vs "suamico village president") from merging
#     on the shared generic office suffix.
#   - district_identifier and seat_name, when both sides expose them, so
#     "... district 1" and "... district 2" do not merge.
#
# state and election_date are exact-equality keys in the blocking rules / EM
# blocks, so Splink never trains their m and drops the gamma_<col>. Reference
# the retained raw _l/_r columns for those instead of gamma_*. The
# locality-token overlap mirrors BASE_POST_PREDICTION_FILTER's stop-word
# treatment; OFFICE_STOP_WORDS strips generic office nouns so the locality is
# the discriminating token.
_es_tok_l = _office_locality_tokens("l")
_es_tok_r = _office_locality_tokens("r")
ELECTION_STAGE_POST_PREDICTION_FILTER = f"""
    state_l = state_r
      AND election_date_l = election_date_r
      AND election_stage_l = election_stage_r
      AND (
        gamma_official_office_name >= 3
        OR (
          candidate_office_l = candidate_office_r
          -- Require locality-token SET agreement (one side's tokens are a
          -- subset of the other's), not just any shared token. Any-overlap let
          -- common geographic words ("grand", "port", "north") chain distinct
          -- towns ("grand prairie" vs "grand saline") on generic offices.
          AND len(list_intersect({_es_tok_l}, {_es_tok_r})) > 0
          AND (
            len(list_intersect({_es_tok_l}, {_es_tok_r})) = len({_es_tok_l})
            OR len(list_intersect({_es_tok_l}, {_es_tok_r})) = len({_es_tok_r})
          )
        )
      )
      AND (
        district_identifier_l IS NULL
        OR district_identifier_r IS NULL
        OR district_identifier_l = district_identifier_r
      )
      AND (
        seat_name_l IS NULL
        OR seat_name_r IS NULL
        OR seat_name_l = seat_name_r
      )
"""
