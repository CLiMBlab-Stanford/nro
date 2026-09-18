"""Immutable scientific migrations, stored one file per destination version."""

from nro.orchestration.migrations.core import load_chain

# Schema 6 is the first supported in-place migration baseline. Add v0007.py,
# then one consecutively numbered module for each later persisted change.
MIGRATIONS = load_chain(__name__, baseline_version=6)
