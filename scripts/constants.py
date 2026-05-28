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
# (no first_name/last_name/email), so the filter requires either a hard
# state+date+office signal, or a state+position-FK fast path. The office
# overlap clause mirrors BASE_POST_PREDICTION_FILTER so we get the same
# locality-token + r-N school-district tolerance.
ELECTION_STAGE_POST_PREDICTION_FILTER = f"""
    gamma_state > 0
      AND (
        gamma_ballotready_position_id > 0
        OR gamma_election_date > 0
      )
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
