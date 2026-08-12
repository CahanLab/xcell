# Export Save Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** File → Export writes to a directory and filename the user picks, instead of dropping a fixed name into the browser's Downloads folder.

**Architecture:** Extract the save picker `PyscnModal` already contains into `SavePathPicker`, generalize `savePath.ts` off `.pkl`, and add one backend route that writes h5ad / metadata / gene sets to a chosen path.

**Tech Stack:** FastAPI, anndata, pytest, React + TypeScript with inline styles.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-12-export-save-path-design.md`.
- The streaming `GET /api/export/h5ad` **stays** — it is right when the backend is not on your machine, and it is a working URL today.
- `h5ad` and `metadata` are produced by the backend. `gene_sets` content comes from the client, because the flattening that produces today's file shape lives in the frontend and duplicating it server-side would let the two drift.
- Errors: bad directory / not-a-file / unknown kind → `400` with a readable message. `except HTTPException: raise` before any catch-all.
- Overwrites must be visible **before** the button is pressed.
- Tests: `backend/tests/`, no `conftest.py`, module-level factories, seeded, `csr_matrix` + `float32`. JSON-serializable responses.
- Frontend: inline styles. Palette — panel `#16213e`, border `#0f3460`, inset `#0f1625`, accent `#4ecdc4`, alert `#e94560`, warning `#e9a23b`, muted `#aaa`/`#888`.

---

### Task 1: `savePath.ts` becomes extension-agnostic

**Files:**
- Modify: `frontend/src/lib/savePath.ts`, `frontend/src/lib/savePath.test.ts`
- Modify: `frontend/src/components/PyscnModal.tsx` (3 call sites)

**Interfaces:**
- Produces: `composeSavePath(dir: string, filename: string, ext: string): string`. `ext` includes the dot (`'.h5ad'`); `''` means "append nothing". `splitSavePath` is unchanged.

- [ ] **Step 1: Update the tests to the explicit extension**

Every existing case passes `'.pkl'` explicitly. Add:

```typescript
it('appends the extension it is given, not a hardcoded one', () => {
  expect(composeSavePath('/data', 'export', '.h5ad')).toBe('/data/export.h5ad')
})

it('leaves a name that already has the extension alone', () => {
  expect(composeSavePath('/data', 'export.h5ad', '.h5ad')).toBe('/data/export.h5ad')
})

it('matches the extension case-insensitively', () => {
  expect(composeSavePath('/data', 'EXPORT.H5AD', '.h5ad')).toBe('/data/EXPORT.H5AD')
})

it('appends nothing when the extension is empty', () => {
  expect(composeSavePath('/data', 'export', '')).toBe('/data/export')
})
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd frontend && npx vitest run src/lib/savePath.test.ts`
Expected: FAIL — extra argument ignored, `.pkl` appended.

- [ ] **Step 3: Implement**

```typescript
export function composeSavePath(dir: string, filename: string, ext: string): string {
  const d = dir.trim()
  const name = filename.trim()
  if (!d || !name) return ''
  // No default extension: a silent fallback is exactly how a .h5ad ends up
  // named .pkl.
  const has = !ext || name.toLowerCase().endsWith(ext.toLowerCase())
  const withExt = has ? name : `${name}${ext}`
  const base = d.endsWith('/') ? d.slice(0, -1) : d
  return `${base}/${withExt}`
}
```

Update the three `PyscnModal` call sites to pass `'.pkl'`; `npx tsc --noEmit` finds any missed.

- [ ] **Step 4: Verify** — `npx vitest run && npx tsc --noEmit`, both clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/savePath.ts frontend/src/lib/savePath.test.ts frontend/src/components/PyscnModal.tsx
git commit -m "refactor(ui): composeSavePath takes the extension it should append"
```

---

### Task 2: The backend write route

**Files:**
- Modify: `backend/xcell/api/routes.py` (`BROWSE_KINDS` at `:96`; new route beside `/export/h5ad` at `:3074`)
- Test: `backend/tests/test_export_save.py`

**Interfaces:**
- Produces: `POST /api/export/save` body `{path: str, kind: 'h5ad'|'metadata'|'gene_sets', content: str | None}` → `{path, kind, n_bytes}`.
- `BROWSE_KINDS` gains `'tabular': ('.tsv', '.csv', '.txt')` and `'geneset': ('.json', '.gmt')`.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from xcell.adaptor import DataAdaptor
from xcell.api import routes
from xcell.main import app


def _install(monkeypatch):
    rng = np.random.default_rng(0)
    X = csr_matrix(rng.poisson(1.0, size=(12, 6)).astype(np.float32))
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame({'cell_type': ['a', 'b'] * 6},
                         index=[f'c{i}' for i in range(12)]),
        var=pd.DataFrame(index=[f'g{i}' for i in range(6)]),
    )
    adaptor = DataAdaptor('x.h5ad', adata=a)
    monkeypatch.setattr(routes, 'get_adaptor', lambda dataset=None: adaptor)
    return adaptor


def test_saves_an_h5ad_that_reads_back(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = tmp_path / 'exported.h5ad'
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(out), 'kind': 'h5ad'})
    assert r.status_code == 200, r.text
    assert out.exists() and r.json()['n_bytes'] > 0
    back = ad.read_h5ad(out)
    assert back.shape == (12, 6)
    assert 'cell_type' in back.obs.columns


def test_saves_metadata_as_tsv(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = tmp_path / 'meta.tsv'
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(out), 'kind': 'metadata'})
    assert r.status_code == 200, r.text
    text = out.read_text()
    assert '\t' in text and 'cell_type' in text


def test_saves_gene_sets_from_client_content(monkeypatch, tmp_path):
    _install(monkeypatch)
    out = tmp_path / 'sets.json'
    payload = json.dumps([{'name': 'skin', 'genes': ['Krt5']}])
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(out), 'kind': 'gene_sets',
                                   'content': payload})
    assert r.status_code == 200, r.text
    assert json.loads(out.read_text())[0]['name'] == 'skin'


def test_gene_sets_without_content_is_400(monkeypatch, tmp_path):
    _install(monkeypatch)
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(tmp_path / 's.json'),
                                   'kind': 'gene_sets'})
    assert r.status_code == 400
    assert 'content' in r.json()['detail'].lower()


def test_a_missing_directory_is_400_naming_it(monkeypatch, tmp_path):
    _install(monkeypatch)
    bad = tmp_path / 'nope' / 'x.h5ad'
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(bad), 'kind': 'h5ad'})
    assert r.status_code == 400
    assert 'nope' in r.json()['detail']


def test_a_directory_as_the_path_is_400(monkeypatch, tmp_path):
    _install(monkeypatch)
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(tmp_path), 'kind': 'h5ad'})
    assert r.status_code == 400


def test_an_unknown_kind_is_400_listing_the_valid_ones(monkeypatch, tmp_path):
    _install(monkeypatch)
    r = TestClient(app).post('/api/export/save',
                             json={'path': str(tmp_path / 'x'), 'kind': 'nonsense'})
    assert r.status_code == 400
    assert 'h5ad' in r.json()['detail']


def test_browse_kinds_cover_the_new_shapes():
    assert set(routes.BROWSE_KINDS) >= {'data', 'classifier', 'tabular', 'geneset'}
```

- [ ] **Step 2: Run to verify it fails** — 404 on the route.

- [ ] **Step 3: Implement**

```python
_EXPORT_SAVE_KINDS = ('h5ad', 'metadata', 'gene_sets')


class ExportSaveRequest(BaseModel):
    path: str
    kind: str = "h5ad"
    # Only gene sets carry content: the flattening that produces the file's
    # shape lives in the frontend, and duplicating it here would let the two
    # drift. h5ad and metadata the backend can produce itself.
    content: str | None = None


@router.post("/export/save")
def export_save(request: ExportSaveRequest, dataset: str | None = Query(None)):
    """Write an export to a path the user picked, rather than to Downloads."""
    if request.kind not in _EXPORT_SAVE_KINDS:
        raise HTTPException(
            status_code=400,
            detail=f"kind must be one of {list(_EXPORT_SAVE_KINDS)}, "
                   f"got '{request.kind}'")

    target = Path(request.path).expanduser()
    if target.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"That path is a directory, not a file: {target}")
    if not target.parent.is_dir():
        raise HTTPException(
            status_code=400, detail=f"No such directory: {target.parent}")

    adaptor = get_adaptor(dataset)
    try:
        if request.kind == 'h5ad':
            adaptor.prepare_export_with_lines().write_h5ad(target)
        elif request.kind == 'metadata':
            target.write_text(adaptor.export_annotations_tsv())
        else:
            if request.content is None:
                raise HTTPException(
                    status_code=400,
                    detail="gene_sets needs 'content' — the sets live in the "
                           "browser session")
            target.write_text(request.content)
    except HTTPException:
        raise
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"path": str(target), "kind": request.kind,
            "n_bytes": int(target.stat().st_size)}
```

`BROWSE_KINDS` gains the two entries. If `export_annotations_tsv` does not exist
on the adaptor under that name, use whatever `POST /api/annotations/export`
already calls — do not add a second implementation.

- [ ] **Step 4: Verify** — `pixi run -e dev pytest tests/ -q`, full suite green.

- [ ] **Step 5: Commit**

```bash
git add backend/xcell/api/routes.py backend/tests/test_export_save.py
git commit -m "feat(api): write an export to a path the user picked"
```

---

### Task 3: `SavePathPicker`, and PyscnModal onto it

**Files:**
- Create: `frontend/src/components/SavePathPicker.tsx`
- Modify: `frontend/src/components/PyscnModal.tsx`

**Interfaces:**
- Produces:

```typescript
interface SavePathPickerProps {
  kind: BrowseKind          // which files to show, and which directory to remember
  ext: string               // '.h5ad' — appended when the typed name lacks it
  value: string             // the composed absolute path
  onChange: (path: string) => void
  defaultName?: string
  label?: string
}
```

The component owns the `FileBrowser`, the filename field and the composed path,
and reports `exists` state so the caller can warn about an overwrite. It tracks
the browsed directory's entries via `FileBrowser`'s `onNavigate(dir, entries)`
and sets an overwrite warning when the composed basename matches one.

- [ ] **Step 1: Build the component**, styled with the existing tokens: browser
  in an inset panel, filename input, the full path shown beneath in `#888`
  monospace, and the overwrite line in `#e9a23b` when it applies.

- [ ] **Step 2: Refactor PyscnModal onto it** with `kind="classifier"` and
  `ext=".pkl"`, deleting the inline browser/filename/compose wiring. If the
  usage turns out not to be a clean drop-in, stop and leave PyscnModal alone
  rather than destabilising a working feature — and say so.

- [ ] **Step 3: Verify** — `npx tsc --noEmit && npx vitest run`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/SavePathPicker.tsx frontend/src/components/PyscnModal.tsx
git commit -m "refactor(ui): one save-path picker, shared"
```

---

### Task 4: The Export modal, verification, docs

**Files:**
- Modify: `frontend/src/App.tsx` (export handlers `:1173-1262`, modal `:2388`)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Rework the modal** — each of the three formats gets a
  `SavePathPicker` (`data`/`.h5ad`, `tabular`/`.tsv`, `geneset`/`.json`) with a
  default name derived from the loaded dataset's filename stem plus `_xcell`,
  and a Save button that POSTs to `/api/export/save`. On success show the path
  written and its size rather than closing silently, so the user knows where it
  went.

- [ ] **Step 2: Keep the download escape hatch** — a small "download instead"
  link per format, preserving today's behaviour for anyone on a remote backend.

- [ ] **Step 3: Verify in the browser** on an isolated backend: export an h5ad
  to a chosen directory, confirm the file appears on disk with a plausible size
  and reads back in Python, confirm the overwrite warning appears on a second
  export to the same name, and confirm a bad directory shows the 400 message.
  Stop the servers afterwards.

- [ ] **Step 4: CHANGELOG and commit** — house voice: what the user sees, the
  measured file sizes from the real check, and what was rejected (the native
  save sheet, and why).

---

## Self-Review

**Spec coverage.** Extracted picker → Task 3. Extension-agnostic `savePath` →
Task 1. Server-side write route → Task 2. Default names and overwrite warning →
Tasks 3, 4. Browse kinds → Task 2. Streaming route kept → Task 4 step 2.
Testing list → Tasks 1, 2.

**Placeholders.** None. Task 2 names the one uncertainty explicitly (the
adaptor's TSV method name) and says what to do rather than leaving it open.

**Type consistency.** `composeSavePath(dir, filename, ext)` is used with an
explicit `ext` in Tasks 1, 3. `SavePathPickerProps.kind` is the `BrowseKind`
union `FileBrowser` already exports. `ExportSaveRequest.path/kind/content`
match the client's POST body in Task 4.
