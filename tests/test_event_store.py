"""Event catalog integrity, conservative name matching, and ingestion snapshots."""

from io import StringIO
import hashlib

import pytest
import yaml

from nro.configuration.events import EventStore
from nro.engine.events import validate_events


def add_task(root, task, variants, *, names=None):
    directory = root / task
    directory.mkdir(parents=True)
    files = {}
    for variant, text in variants.items():
        (directory / f'{variant}.tsv').write_text(text)
        files[variant] = {'path': f'{task}/{variant}.tsv', 'source_names': []}
    (directory / 'index.yml').write_text(yaml.safe_dump({'tasks': names or [task], 'files': files}))


EVENTS = 'onset\tduration\ttrial_type\n0\t1\tA\n'


def test_complete_task_names_not_prefixes(tmp_path):
    add_task(tmp_path, 'langlocSN', {'set1_run1': EVENTS, 'set1_run2': EVENTS})
    add_task(tmp_path, 'langlocSNW', {'main': EVENTS})
    store = EventStore(tmp_path)
    assert [e.identifier for e in store.candidates('LANGLOC-SN')] == ['langlocSN/set1_run1', 'langlocSN/set1_run2']
    assert store.candidates('langloc') == []
    assert len(store.candidates('langlocSNW')) == 1
    assert store.candidates('') == []


@pytest.mark.parametrize('identifier', ['../x', '/langlocSN/main', 'langlocSN/../main', 'unknown/main'])
def test_ids_cannot_escape_store(tmp_path, identifier):
    with pytest.raises(ValueError):
        EventStore(tmp_path).resolve(identifier)


def test_index_cannot_reference_external_files(tmp_path):
    add_task(tmp_path, 'task', {'main': EVENTS})
    index = tmp_path / 'task/index.yml'
    value = yaml.safe_load(index.read_text())
    value['files']['main']['path'] = '../external.tsv'
    index.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='outside'):
        EventStore(tmp_path).candidates('task')


def test_identical_events_in_distinct_tasks_are_independent(tmp_path):
    add_task(tmp_path, 'langlocSN', {'main': EVENTS})
    add_task(tmp_path, 'sentenceCompletion', {'main': EVENTS})
    store = EventStore(tmp_path)
    first = store.resolve('langlocSN/main')
    second = store.resolve('sentenceCompletion/main')
    assert first.path != second.path
    first.path.write_text(EVENTS.replace('A', 'B'))
    assert second.snapshot()[0] == EVENTS


@pytest.mark.parametrize('symlink', [False, True])
def test_index_cannot_share_another_tasks_file(tmp_path, symlink):
    add_task(tmp_path, 'task', {'main': EVENTS})
    add_task(tmp_path, 'other', {'main': EVENTS})
    index = tmp_path / 'task/index.yml'
    value = yaml.safe_load(index.read_text())
    if symlink:
        (tmp_path / 'task/shared.tsv').symlink_to(tmp_path / 'other/main.tsv')
        value['files']['main']['path'] = 'task/shared.tsv'
    else:
        value['files']['main']['path'] = 'other/main.tsv'
    index.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='task directory'):
        EventStore(tmp_path).resolve('task/main')


def test_wizard_chooses_id_and_retains_snapshot_provenance(tmp_path, monkeypatch):
    from nro.bidsify.review import _events
    add_task(tmp_path, 'task', {'one': EVENTS, 'two': EVENTS.replace('A', 'B')})
    item = {'entities': {'task': 'task', 'run': '01'}, 'events': None}
    prompts = []
    answers = iter(['task/two', 'y'])
    def input_value(prompt):
        prompts.append(prompt)
        return next(answers)
    monkeypatch.setattr('builtins.input', input_value)
    text = _events({'config': {'event_store': str(tmp_path), 'event_rules': []}}, item)
    assert '[task/one]' not in prompts[0]
    assert text == EVENTS.replace('A', 'B')
    assert item['events_source'] == {'catalog_id': 'task/two', 'sha256': hashlib.sha256(text.encode()).hexdigest()}
    assert item['events'] is None


def test_catalog_and_explicit_rule_candidates(tmp_path):
    from nro.bidsify.review import event_candidates
    add_task(tmp_path, 'task', {'main': EVENTS})
    other = tmp_path / 'local.tsv'
    other.write_text(EVENTS)
    config = {'event_store': str(tmp_path), 'event_rules': [{'task': 'task', 'pattern': str(other)}]}
    entries, paths = event_candidates(config, 'task')
    assert [entry.identifier for entry in entries] == ['task/main']
    assert paths == [str(other)]


def test_snapshot_is_validated_at_selection(tmp_path):
    add_task(tmp_path, 'task', {'main': EVENTS})
    entry = EventStore(tmp_path).resolve('task/main')
    assert entry.snapshot()[0] == EVENTS
    entry.path.write_text('onset\tduration\n0\t-1\n')
    with pytest.raises(ValueError, match='nonnegative'):
        entry.snapshot()


def test_imported_catalog_integrity_and_shared_contents():
    store = EventStore()
    paths = set()
    identifiers = []
    for index in store.root.glob('*/index.yml'):
        for entry in store._entries(index)[1]:
            text, provenance = store.resolve(entry.identifier).snapshot()
            validate_events(StringIO(text))
            assert provenance['catalog_id'] == entry.identifier
            paths.add(entry.path)
            identifiers.append(entry.identifier)
    assert len(identifiers) == len(set(identifiers))
    assert len(paths) < len(identifiers)
    assert paths == set(store.root.glob('*/*.tsv'))
    digests = [(path.parent.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
    assert len(digests) == len(set(digests))
    assert store.candidates('unassigned') == []
