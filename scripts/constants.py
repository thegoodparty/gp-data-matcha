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
    "'and','for','no.','odd','unexpired'"
)

# Shared post-prediction filter: requires name + identity signal + office overlap.
# Each config can extend this with entity-specific clauses.
# The locality-token-overlap branch is guarded by a district_identifier check —
# locality overlap alone (e.g. shared city/county name) is insufficient when both
# sides have a district_identifier that disagrees, since that signals different
# races in the same locality (city council vs county board, district 8 vs 14, etc.).
BASE_POST_PREDICTION_FILTER = f"""
    gamma_last_name > 0
      AND (gamma_first_name > 0 OR gamma_email > 0 OR gamma_phone > 0)
      AND (
        gamma_official_office_name > 0
        OR (
          list_has_any(
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
          AND (
            district_identifier_l IS NULL
            OR district_identifier_r IS NULL
            OR district_identifier_l = district_identifier_r
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
        OR (
          list_has_any(
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
          AND (
            district_identifier_l IS NULL
            OR district_identifier_r IS NULL
            OR district_identifier_l = district_identifier_r
          )
        )
        OR gamma_office_type > 0
        OR gamma_ballotready_position_id > 0
      )
"""
