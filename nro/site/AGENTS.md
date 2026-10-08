# Site agent guidance

This package owns protected site settings, installation bootstrap, dependency
acquisition, shared maintenance, launchers, and upgrade rehearsal. Site operations
may coordinate with orchestration, but scientific algorithms and definition-store
formats belong in `nro.engine` and `nro.definitions` respectively.

Run `nro dev test --sphere installation` for focused validation. Changes to scheduler
or registry behavior also require the orchestration sphere.
