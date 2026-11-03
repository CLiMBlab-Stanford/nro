---
section: Overview
context_version: 1
requires: [observed, runs, participants, spaces]
---
{% if observed %}
A total of {{ runs }} blood-oxygen-level-dependent (BOLD) {{ runs | plural("run") }}
from {{ participants }} {{ participants | plural("participant") }} were preprocessed.
{% endif %}
Each BOLD run was preprocessed once and published in the {{ spaces | join_and }}
spaces. Slice-timing correction was not performed.
