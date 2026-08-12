# Ensembl Gene Symbols — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user turn a dataset whose `.var` holds only Ensembl IDs into one with official gene symbols, in one action, offline.

**Architecture:** A pure `xcell/gene_symbols.py` loads the shipped GENCODE tables and maps IDs; the adaptor writes `.var['gene_symbol']` and optionally calls the *existing* `swap_var_index` to promote it. No new renaming path.

**Tech Stack:** numpy, pandas, gzip (stdlib), pytest, React + TypeScript with inline styles.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-12-ensembl-gene-symbols-design.md`.
- Shipped tables: `xcell/data/ensembl_symbols_{human,mouse}.tsv.gz`, `id<TAB>symbol`, no header. Built by `xcell/data/build_gene_symbol_maps.py`.
- Species detected from ID prefixes (`ENSG`→human, `ENSMUSG`→mouse) by majority; anything else is refused with the species named.
- Version suffixes stripped on both sides (`ENSG00000141510.15` → `ENSG00000141510`).
- Unmapped IDs keep their own ID. Never blank, never `NaN`, never dropped.
- Duplicate symbols are reported, never a refusal.
- Preview mutates nothing and is JSON-serializable: no `NaN`, no `inf`.
- Tests: `backend/tests/`, no `conftest.py`, module-level `_adata()`/`_adaptor()` factories, seeded, `csr_matrix` + `float32`.
- Frontend: inline styles. Palette — panel `#16213e`, border `#0f3460`, inset `#0f1625`, accent `#4ecdc4`, alert `#e94560`, warning `#e9a23b`, muted `#aaa`.
- A feature creating a new `.var` state must refresh: `refreshSchema()` and `fetchVarIdentifierColumns()`.

---

### Task 1: The pure lookup module

**Files:**
- Create: `backend/xcell/gene_symbols.py`
- Test: `backend/tests/test_gene_symbols.py`

**Interfaces:**
- Produces: `SPECIES_PREFIXES: dict[str, str]`; `strip_version(gene_id: str) -> str`; `detect_species(ids: list[str]) -> dict` returning `{'species', 'counts', 'n_recognized', 'fraction'}`; `load_table(species: str) -> dict[str, str]` (cached); `map_ids(ids, table) -> dict` returning `{'symbols': list[str], 'n_mapped', 'n_unmapped', 'n_duplicate_symbols', 'examples'}`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest

from xcell.gene_symbols import (
    detect_species, load_table, map_ids, strip_version,
)

TABLE = {'ENSMUSG00000002603': 'Tgfb1', 'ENSMUSG00000051951': 'Xkr4',
         'ENSMUSG00000000001': 'Gnai3', 'ENSMUSG00000000002': 'Gnai3'}


def test_strip_version_removes_only_the_suffix():
    assert strip_version('ENSG00000141510.15') == 'ENSG00000141510'
    assert strip_version('ENSG00000141510') == 'ENSG00000141510'
    assert strip_version('Tgfb1') == 'Tgfb1'


def test_detect_species_reads_the_prefix():
    assert detect_species(['ENSG00000141510', 'ENSG00000012048'])['species'] == 'human'
    assert detect_species(['ENSMUSG00000002603'])['species'] == 'mouse'


def test_detect_species_takes_the_majority_and_reports_the_split():
    out = detect_species(['ENSMUSG1', 'ENSMUSG2', 'ENSMUSG3', 'ENSG1'])
    assert out['species'] == 'mouse'
    assert out['counts']['mouse'] == 3 and out['counts']['human'] == 1


def test_detect_species_names_an_unsupported_species():
    out = detect_species(['ENSDARG00000000001', 'ENSDARG00000000002'])
    assert out['species'] is None
    assert 'zebrafish' in (out['unsupported'] or '')


def test_detect_species_reports_symbols_as_not_ensembl():
    out = detect_species(['Tgfb1', 'Xkr4', 'Shh'])
    assert out['species'] is None
    assert out['n_recognized'] == 0


def test_map_ids_maps_and_keeps_unmapped_ids_as_themselves():
    out = map_ids(['ENSMUSG00000002603', 'ENSMUSG99999999999'], TABLE)
    assert out['symbols'] == ['Tgfb1', 'ENSMUSG99999999999']
    assert out['n_mapped'] == 1 and out['n_unmapped'] == 1


def test_map_ids_ignores_the_version_suffix():
    out = map_ids(['ENSMUSG00000002603.7'], TABLE)
    assert out['symbols'] == ['Tgfb1']


def test_map_ids_counts_duplicate_symbols_without_raising():
    out = map_ids(['ENSMUSG00000000001', 'ENSMUSG00000000002'], TABLE)
    assert out['symbols'] == ['Gnai3', 'Gnai3']
    assert out['n_duplicate_symbols'] == 1


def test_map_ids_never_produces_a_blank_symbol():
    out = map_ids(['ENSMUSG00000002603', 'weird', ''], TABLE)
    assert all(s != '' and s is not None for s in out['symbols'][:2])


def test_the_shipped_tables_load_and_are_large():
    for species, probe, expect in [('human', 'ENSG00000141510', 'TP53'),
                                   ('mouse', 'ENSMUSG00000002603', 'Tgfb1')]:
        table = load_table(species)
        assert len(table) > 40_000, f'{species} table looks truncated'
        assert table[probe] == expect
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_gene_symbols.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'xcell.gene_symbols'`

- [ ] **Step 3: Write the minimal implementation**

```python
"""Map Ensembl gene IDs to official symbols, from tables shipped with xcell.

Pure: takes lists of strings, returns plain dicts. The tables are loaded from
``xcell/data`` and cached per species, so a session pays the read once.

Shipped rather than fetched so this works with no network, returns instantly,
and maps the same dataset the same way in a year -- which a live API cannot
promise.
"""
from __future__ import annotations

import gzip
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

_DATA = Path(__file__).resolve().parent / 'data'

# Ensembl encodes the species in the prefix, so it never has to be asked for.
SPECIES_PREFIXES = {'ENSG': 'human', 'ENSMUSG': 'mouse'}

# Named only so an unsupported dataset gets a message that says which animal it
# is, rather than "unsupported".
_OTHER_PREFIXES = {
    'ENSDARG': 'zebrafish', 'ENSRNOG': 'rat', 'ENSGALG': 'chicken',
    'ENSXETG': 'frog', 'ENSCAFG': 'dog', 'ENSSSCG': 'pig',
    'ENSMMUG': 'macaque', 'ENSBTAG': 'cow',
}


def strip_version(gene_id: str) -> str:
    """``ENSG00000141510.15`` -> ``ENSG00000141510``.

    Whether a dataset keeps the version suffix depends on which tool wrote it,
    not on the biology, so it is removed on both sides of every lookup.
    """
    gid = str(gene_id)
    head = gid.split('.', 1)[0]
    return head if head.startswith('ENS') else gid


def _prefix_of(gene_id: str) -> str | None:
    gid = strip_version(gene_id)
    # Longest first: ENSMUSG must win over ENSG-as-a-prefix-of-nothing.
    for pfx in sorted(list(SPECIES_PREFIXES) + list(_OTHER_PREFIXES),
                      key=len, reverse=True):
        if gid.startswith(pfx):
            return pfx
    return None


def detect_species(ids: list[str]) -> dict[str, Any]:
    """Which species these Ensembl IDs are, by majority, or None."""
    counts: Counter[str] = Counter()
    other: Counter[str] = Counter()
    for gid in ids:
        pfx = _prefix_of(gid)
        if pfx in SPECIES_PREFIXES:
            counts[SPECIES_PREFIXES[pfx]] += 1
        elif pfx is not None:
            other[_OTHER_PREFIXES[pfx]] += 1

    n_recognized = sum(counts.values())
    total = max(len(ids), 1)
    species = counts.most_common(1)[0][0] if counts else None
    unsupported = other.most_common(1)[0][0] if (not counts and other) else None
    return {
        'species': species,
        'counts': dict(counts),
        'n_recognized': n_recognized,
        'fraction': n_recognized / total,
        'unsupported': unsupported,
    }


@lru_cache(maxsize=4)
def load_table(species: str) -> dict[str, str]:
    """Ensembl ID -> symbol for one species, read once per session."""
    if species not in SPECIES_PREFIXES.values():
        raise ValueError(
            f"no symbol table for '{species}' — xcell ships human and mouse")
    path = _DATA / f'ensembl_symbols_{species}.tsv.gz'
    if not path.exists():
        raise ValueError(
            f'the {species} symbol table is missing from this install '
            f'({path.name}). Rebuild it with '
            '`python xcell/data/build_gene_symbol_maps.py`.')
    table: dict[str, str] = {}
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            gid, _, sym = line.rstrip('\n').partition('\t')
            if gid and sym:
                table[gid] = sym
    return table


def map_ids(ids: list[str], table: dict[str, str]) -> dict[str, Any]:
    """Map each ID to a symbol, keeping the ID itself where there is no match."""
    symbols: list[str] = []
    examples: list[dict[str, str]] = []
    n_mapped = 0
    for gid in ids:
        sym = table.get(strip_version(gid))
        if sym:
            n_mapped += 1
            if len(examples) < 5:
                examples.append({'id': str(gid), 'symbol': sym})
        # An unmapped gene keeps its own id: blank would break identifier
        # detection and make the gene unaddressable, which is worse than ugly.
        symbols.append(sym or str(gid))

    dupes = [s for s, n in Counter(symbols).items() if n > 1]
    return {
        'symbols': symbols,
        'n_mapped': n_mapped,
        'n_unmapped': len(ids) - n_mapped,
        'n_duplicate_symbols': len(dupes),
        'examples': examples,
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd backend && pixi run -e dev pytest tests/test_gene_symbols.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/gene_symbols.py backend/tests/test_gene_symbols.py backend/xcell/data/
git commit -m "feat(genes): map Ensembl ids to symbols from shipped GENCODE tables"
```

---

### Task 2: Adaptor — preview and apply

**Files:**
- Modify: `backend/xcell/adaptor.py` (new methods near `swap_var_index`, `adaptor.py:1567`)
- Test: `backend/tests/test_gene_symbols_adaptor.py`

**Interfaces:**
- Consumes: Task 1's `detect_species`, `load_table`, `map_ids`.
- Produces: `preview_gene_symbol_mapping() -> dict` with keys `species`, `n_genes`, `n_recognized`, `n_mapped`, `n_unmapped`, `n_duplicate_symbols`, `examples`, `existing_symbol_column`, `applicable`, `reason`; and `map_gene_symbols(*, column='gene_symbol', set_as_index=False) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor

MOUSE = ['ENSMUSG00000002603', 'ENSMUSG00000051951', 'ENSMUSG00000000001']


def _adata(names, n=12, seed=0):
    rng = np.random.default_rng(seed)
    X = csr_matrix(rng.poisson(1.0, size=(n, len(names))).astype(np.float32))
    obs = pd.DataFrame(index=[f'c{i}' for i in range(n)])
    var = pd.DataFrame(index=pd.Index(names, name=None))
    return ad.AnnData(X=X, obs=obs, var=var)


def _adaptor(a):
    return DataAdaptor('x.h5ad', adata=a)


def test_preview_detects_mouse_and_mutates_nothing():
    a = _adata(MOUSE)
    d = _adaptor(a)
    before = list(a.var_names)
    out = d.preview_gene_symbol_mapping()
    assert out['species'] == 'mouse'
    assert out['applicable'] is True
    assert out['n_mapped'] >= 2
    assert list(a.var_names) == before          # untouched
    assert 'gene_symbol' not in a.var.columns
    json.dumps(out)                              # crosses the API


def test_preview_says_it_is_not_needed_when_names_are_already_symbols():
    d = _adaptor(_adata(['Tgfb1', 'Xkr4', 'Shh']))
    out = d.preview_gene_symbol_mapping()
    assert out['applicable'] is False
    assert 'symbol' in out['reason'].lower() or 'ensembl' in out['reason'].lower()


def test_preview_names_an_unsupported_species():
    d = _adaptor(_adata(['ENSDARG00000000001', 'ENSDARG00000000002']))
    out = d.preview_gene_symbol_mapping()
    assert out['applicable'] is False
    assert 'zebrafish' in out['reason']


def test_map_writes_the_column_and_leaves_the_index_alone():
    a = _adata(MOUSE)
    d = _adaptor(a)
    out = d.map_gene_symbols()
    assert 'gene_symbol' in a.var.columns
    assert list(a.var_names) == MOUSE            # index untouched
    assert a.var['gene_symbol'].iloc[0] == 'Tgfb1'
    assert out['n_mapped'] >= 2
    json.dumps(out)


def test_map_with_set_as_index_promotes_symbols_and_keeps_them_unique():
    a = _adata(MOUSE)
    d = _adaptor(a)
    d.map_gene_symbols(set_as_index=True)
    assert 'Tgfb1' in list(a.var_names)
    assert len(set(a.var_names)) == len(a.var_names)


def test_map_refuses_an_unsupported_species():
    d = _adaptor(_adata(['ENSDARG00000000001', 'ENSDARG00000000002']))
    with pytest.raises(ValueError, match='zebrafish'):
        d.map_gene_symbols()


def test_unmapped_genes_keep_their_id_rather_than_going_blank():
    a = _adata(['ENSMUSG00000002603', 'ENSMUSG99999999999'])
    d = _adaptor(a)
    d.map_gene_symbols()
    vals = list(a.var['gene_symbol'])
    assert vals[1] == 'ENSMUSG99999999999'
    assert all(v and not pd.isna(v) for v in vals)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_gene_symbols_adaptor.py -q`
Expected: FAIL — `AttributeError: 'DataAdaptor' object has no attribute 'preview_gene_symbol_mapping'`

- [ ] **Step 3: Write the minimal implementation**

Add to `adaptor.py` immediately after `swap_var_index`:

```python
    # Columns whose name says "this is already a symbol", checked before
    # offering to create another one.
    _SYMBOL_COLUMNS = ('gene_symbol', 'gene_symbols', 'symbol', 'SYMBOL',
                       'gene_name', 'gene_names', 'feature_name')

    def _gene_symbol_state(self) -> tuple[dict[str, Any], str | None]:
        """(species detection over var_names, an existing symbol column or None)."""
        from xcell import gene_symbols as gs

        ids = [str(n) for n in self.adata.var_names]
        existing = next(
            (c for c in self._SYMBOL_COLUMNS if c in self.adata.var.columns), None)
        return gs.detect_species(ids), existing

    def preview_gene_symbol_mapping(self) -> dict[str, Any]:
        """What mapping Ensembl ids to symbols would do here. Mutates nothing."""
        from xcell import gene_symbols as gs

        detection, existing = self._gene_symbol_state()
        ids = [str(n) for n in self.adata.var_names]
        out: dict[str, Any] = {
            'species': detection['species'],
            'n_genes': int(len(ids)),
            'n_recognized': int(detection['n_recognized']),
            'existing_symbol_column': existing,
            'n_mapped': 0, 'n_unmapped': 0, 'n_duplicate_symbols': 0,
            'examples': [], 'applicable': False, 'reason': '',
        }

        if detection['species'] is None:
            if detection['unsupported']:
                out['reason'] = (
                    f"These look like {detection['unsupported']} Ensembl ids. "
                    'xcell ships symbol tables for human and mouse only.')
            else:
                out['reason'] = (
                    'These gene names are not Ensembl ids, so there is nothing '
                    'to map — they are already symbols or another identifier.')
            return out

        mapped = gs.map_ids(ids, gs.load_table(detection['species']))
        out.update({
            'n_mapped': int(mapped['n_mapped']),
            'n_unmapped': int(mapped['n_unmapped']),
            'n_duplicate_symbols': int(mapped['n_duplicate_symbols']),
            'examples': mapped['examples'],
            'applicable': True,
        })
        if existing:
            # Not a refusal: the column may be stale or partial. Say it exists
            # so the user can use the Gene IDs picker instead if it is fine.
            out['reason'] = (
                f"'.var[{existing!r}]' already looks like symbols — you may not "
                'need this; the Gene IDs picker can switch to it.')
        return out

    def map_gene_symbols(
        self, *, column: str = 'gene_symbol', set_as_index: bool = False,
    ) -> dict[str, Any]:
        """Write official symbols for Ensembl ids into ``.var[column]``.

        Unmapped ids keep their own id, so no gene becomes unaddressable.
        Duplicate symbols are reported rather than refused: unlike a rename
        pattern, a collision here is a fact about the annotation and there is
        nothing for the user to correct.
        """
        from xcell import gene_symbols as gs

        if not column:
            raise ValueError('column must be a non-empty name')
        detection, _ = self._gene_symbol_state()
        if detection['species'] is None:
            if detection['unsupported']:
                raise ValueError(
                    f"These look like {detection['unsupported']} Ensembl ids; "
                    'xcell ships symbol tables for human and mouse only.')
            raise ValueError(
                'These gene names are not Ensembl ids, so there is nothing to '
                'map.')

        ids = [str(n) for n in self.adata.var_names]
        mapped = gs.map_ids(ids, gs.load_table(detection['species']))
        self.adata.var[column] = pd.Categorical(mapped['symbols'])

        result = {
            'species': detection['species'],
            'column': column,
            'n_genes': int(len(ids)),
            'n_mapped': int(mapped['n_mapped']),
            'n_unmapped': int(mapped['n_unmapped']),
            'n_duplicate_symbols': int(mapped['n_duplicate_symbols']),
            'examples': mapped['examples'],
            'set_as_index': bool(set_as_index),
        }
        self._log_action('map_gene_symbols',
                         {'column': column, 'set_as_index': set_as_index},
                         result)
        if set_as_index:
            # The one implementation of this operation, cache invalidation and
            # all. Re-implementing it here is how the caches get out of step.
            self.swap_var_index(column)
        return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && pixi run -e dev pytest tests/test_gene_symbols_adaptor.py tests/ -q`
Expected: PASS, no regressions in the existing suite.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/adaptor.py backend/tests/test_gene_symbols_adaptor.py
git commit -m "feat(genes): preview and apply Ensembl symbol mapping"
```

---

### Task 3: Routes

**Files:**
- Modify: `backend/xcell/api/routes.py` (beside the var-identifier routes, `routes.py:2475`)
- Test: `backend/tests/test_gene_symbols_routes.py`

**Interfaces:**
- Consumes: Task 2's adaptor methods.
- Produces: `GET /api/genes/symbol_mapping_preview` and `POST /api/genes/map_symbols` with body `{column?: str, set_as_index?: bool, dataset?: str}`.

- [ ] **Step 1: Write the failing test**

```python
import anndata as ad
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app

MOUSE = ['ENSMUSG00000002603', 'ENSMUSG00000051951', 'ENSMUSG00000000001']


def _install(monkeypatch, names):
    rng = np.random.default_rng(0)
    X = csr_matrix(rng.poisson(1.0, size=(10, len(names))).astype(np.float32))
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=[f'c{i}' for i in range(10)]),
                   var=pd.DataFrame(index=names))
    adaptor = DataAdaptor('x.h5ad', adata=a)
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: adaptor)
    return adaptor


def test_preview_route_reports_the_species(monkeypatch):
    _install(monkeypatch, MOUSE)
    r = TestClient(app).get('/api/genes/symbol_mapping_preview')
    assert r.status_code == 200
    assert r.json()['species'] == 'mouse'


def test_map_route_writes_the_column(monkeypatch):
    a = _install(monkeypatch, MOUSE)
    r = TestClient(app).post('/api/genes/map_symbols', json={})
    assert r.status_code == 200
    assert r.json()['n_mapped'] >= 2
    assert 'gene_symbol' in a.adata.var.columns


def test_map_route_maps_a_bad_species_to_400(monkeypatch):
    _install(monkeypatch, ['ENSDARG00000000001', 'ENSDARG00000000002'])
    r = TestClient(app).post('/api/genes/map_symbols', json={})
    assert r.status_code == 400
    assert 'zebrafish' in r.json()['detail']
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd backend && pixi run -e dev pytest tests/test_gene_symbols_routes.py -q`
Expected: FAIL — 404 on both routes.

- [ ] **Step 3: Write the minimal implementation**

Add beside the var-identifier routes:

```python
@router.get("/genes/symbol_mapping_preview")
def genes_symbol_mapping_preview(dataset: str | None = Query(None)):
    """What mapping Ensembl ids to symbols would do here. Mutates nothing."""
    adaptor = get_adaptor(dataset)
    try:
        return adaptor.preview_gene_symbol_mapping()
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class MapGeneSymbolsRequest(BaseModel):
    column: str = "gene_symbol"
    set_as_index: bool = False
    dataset: str | None = None


@router.post("/genes/map_symbols")
def genes_map_symbols(request: MapGeneSymbolsRequest,
                      dataset: str | None = Query(None)):
    """Write official gene symbols into .var, optionally as the gene index."""
    adaptor = get_adaptor(request.dataset or dataset)
    try:
        return adaptor.map_gene_symbols(
            column=request.column, set_as_index=request.set_as_index)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd backend && pixi run -e dev pytest tests/ -q`
Expected: PASS, full suite.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/api/routes.py backend/tests/test_gene_symbols_routes.py
git commit -m "feat(api): routes for previewing and applying symbol mapping"
```

---

### Task 4: The Genes panel action

**Files:**
- Create: `frontend/src/components/GeneSymbolModal.tsx`
- Modify: `frontend/src/components/GenePanel.tsx` (OverflowMenu at `:1842`)

**Interfaces:**
- Consumes: Task 3's two routes.
- Produces: a modal opened from the `⋯` menu item "Add gene symbols…".

- [ ] **Step 1: Build the modal**

A local-`useState` modal in the owning panel, per the house pattern. On open it
GETs the preview and shows: detected species, `n_mapped` of `n_genes`, the
unmapped count, the duplicate-symbol count, up to five worked examples
(`ENSMUSG00000002603 → Tgfb1`), and any `reason`. A checkbox **"Use symbols as
gene names"** defaults to **checked**. The apply button is disabled when
`applicable` is false. Click-outside and Escape close it; the card
`stopPropagation`s.

- [ ] **Step 2: Wire the menu item**

```tsx
<OverflowMenu
  items={[
    { label: MESSAGES.geneMask.menuItem, onClick: () => setGeneMaskModalOpen(true) },
    { label: 'Add gene symbols…', onClick: () => setSymbolModalOpen(true) },
  ]}
/>
```

- [ ] **Step 3: Refresh both channels after applying**

The mapping creates new `.var` state, so on success call **both**
`refreshSchema()` and `fetchVarIdentifierColumns()` — the first so the gene
count and index are current, the second so the **Gene IDs:** dropdown offers the
new column immediately. Missing either leaves the UI showing stale identifiers.

- [ ] **Step 4: Verify**

Run: `cd frontend && npx tsc --noEmit && npx vitest run`
Expected: tsc clean, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/GeneSymbolModal.tsx frontend/src/components/GenePanel.tsx
git commit -m "feat(ui): add gene symbols from the Genes panel"
```

---

### Task 5: Verify on real data, document

**Files:**
- Modify: `CHANGELOG.md` (under `## [Unreleased]` → `### Added`)

- [ ] **Step 1: Find or make a real Ensembl-id dataset**

Scan the user's h5ad files for one whose `var_names` start with `ENSMUSG`/`ENSG`.
If none exists, derive a temporary one by renaming a known dataset's genes to
their Ensembl ids, and say so in the notes.

- [ ] **Step 2: Verify in the browser**

Start both servers, load that dataset, open Genes → `⋯` → **Add gene symbols…**,
confirm the preview reports the right species and a plausible mapped fraction,
apply with the checkbox on, then confirm gene search finds a symbol (e.g.
`Shh`) that previously returned nothing. Stop both servers.

- [ ] **Step 3: Write the CHANGELOG entry and commit**

House voice: lead with what the user sees, give the measured numbers (table
sizes, mapped fraction on the real dataset), and name what was rejected and why
(live API lookups; refusing on duplicate symbols).

```bash
git add CHANGELOG.md
git commit -m "docs: gene symbols for Ensembl-only datasets"
```

---

## Self-Review

**Spec coverage.** Shipped tables → Task 1 + the build script. Species detection → Task 1. Version stripping → Task 1. Preview-before-acting → Tasks 2, 3, 4. Unmapped keep their id → Tasks 1, 2. Duplicates tolerated → Tasks 1, 2. Optional index promotion → Task 2 (delegating to `swap_var_index`). UI location → Task 4. Testing list → Tasks 1–3.

**Placeholders.** None: every step carries code or an exact command. Task 5 step 1 is conditional by necessity and both branches are specified.

**Type consistency.** `detect_species` returns `species`/`counts`/`n_recognized`/`fraction`/`unsupported`, consumed under those names in Task 2. `map_ids` returns `symbols`/`n_mapped`/`n_unmapped`/`n_duplicate_symbols`/`examples`, consumed under those names in Task 2. `preview_gene_symbol_mapping` returns `applicable`/`reason`/`existing_symbol_column`, consumed in Tasks 3 and 4. `MapGeneSymbolsRequest.column`/`set_as_index` match `map_gene_symbols`'s keyword-only parameters.
