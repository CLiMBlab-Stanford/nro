# Definitions agent guidance

This package owns user-managed definitions, their schemas, store inheritance,
authoring, migration, and runtime resolution. Keep definition parsing deterministic
and independent of scheduler state. Protected installation settings belong in
`nro.site`; scientific file and image primitives belong in `nro.engine`.

Run `nro dev test --sphere definitions` for focused validation. Changes to compiled
scientific values also require the affected module spheres.
