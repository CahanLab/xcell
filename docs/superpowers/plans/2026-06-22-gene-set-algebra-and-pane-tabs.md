# Gene-set algebra + pane tabs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users turn `.var` boolean columns into gene sets and combine gene sets/columns with set operations (union/intersection/difference/symmetric-difference), and de-crowd the Genes and Cells panes with tabs.

**Architecture:** One new backend endpoint resolves a boolean `.var` column to gene names; everything else is frontend. Gene sets stay frozen name-lists (`GeneSet`); set math is a pure frontend util; new sets land in the Manual category. The Genes pane gets `Sets | Color` tabs and the Cells pane gets `Categorical | Continuous` tabs.

**Tech Stack:** Python (FastAPI, pandas/anndata, pytest), React/TypeScript (Zustand store, Vite). No frontend test runner exists — frontend is verified via `tsc --noEmit`, `vite build`, and a Playwright smoke on an isolated stack.

**Spec:** `docs/superpowers/specs/2026-06-22-gene-set-algebra-and-pane-tabs-design.md`

**Conventions:**
- Backend tests: `cd backend && pixi run -e dev python -m pytest tests/ -q`
- Frontend checks: `cd frontend && npx tsc --noEmit && npm run build`
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Isolated smoke stack (never touch the user's :8000/:5173): backend
  `XCELL_DATA_PATH=<abs>/backend/xcell/data/toy_spatial_ligrec.h5ad pixi run -e dev uvicorn xcell.main:app --host 127.0.0.1 --port 8001`
  + frontend `XCELL_BACKEND=http://127.0.0.1:8001 npx vite --port 5180 --strictPort`. Clean up servers + screenshots after.

## File map
- `backend/xcell/adaptor.py` — add `column_to_gene_names` (near `get_var_boolean_columns`, ~:3287).
- `backend/xcell/api/routes.py` — add `GET /var/column_genes` (near the other `/var/*` routes, ~:2037).
- `backend/tests/test_var_column_genes.py` — new.
- `frontend/src/hooks/useData.ts` — add `fetchVarBooleanColumns`, `fetchVarColumnGenes`.
- `frontend/src/lib/geneSetOps.ts` — new pure set-op util.
- `frontend/src/components/CombineGeneSetsModal.tsx` — new.
- `frontend/src/components/VarColumnsSection.tsx` — new (.var-columns list in the Sets tab).
- `frontend/src/components/GenePanel.tsx` — `Sets | Color` tabs + mount the new pieces.
- `frontend/src/components/CellPanel.tsx` — `Categorical | Continuous` tabs.
- `frontend/src/store.ts` — `combineModalOpen` flag + setter.

---

## Phase 1 — Pane reorganization (UI only)

### Task 1: Cells pane — Categorical | Continuous tabs

**Files:** Modify `frontend/src/components/CellPanel.tsx`

- [ ] **Step 1: Add tab state**

After the existing `const [continuousExpanded, setContinuousExpanded] = useState(true)` (~:902) add:

```tsx
  const [cellTab, setCellTab] = useState<'categorical' | 'continuous'>('categorical')
```

- [ ] **Step 2: Replace the two collapsible sections with a tab strip + active list**

In the render (`CellPanel.tsx` ~:1321–1391), replace the entire **"Categorical Columns - collapsible"** block AND the **"Continuous Columns - collapsible"** block (the two `{categoricalColumns.length > 0 && (…)}` / `{continuousColumns.length > 0 && (…)}` siblings) with this single block:

```tsx
        {/* Categorical | Continuous tabs */}
        <div style={{ display: 'flex', gap: '4px', margin: '4px 0 8px' }}>
          {([
            ['categorical', `Categorical (${categoricalColumns.length})`],
            ['continuous', `Continuous (${continuousColumns.length})`],
          ] as const).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setCellTab(key)}
              style={{
                flex: 1, padding: '5px 8px', fontSize: '11px', cursor: 'pointer',
                backgroundColor: cellTab === key ? '#0f3460' : 'transparent',
                color: cellTab === key ? '#4ecdc4' : '#888',
                border: '1px solid ' + (cellTab === key ? '#4ecdc4' : '#1a1a2e'),
                borderRadius: '4px',
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {cellTab === 'categorical' && (
          categoricalColumns.length > 0 ? (
            <div style={styles.section}>
              {categoricalColumns.map((summary) => (
                <CategoryColumn
                  key={summary.name}
                  summary={summary}
                  displayName={getDisplayName(summary.name)}
                  isActive={colorMode === 'metadata' && selectedColorColumn === summary.name}
                  onColorBy={() => handleColorBy(summary.name)}
                  onSelectCells={(categoryValue) => handleSelectCellsByCategory(summary.name, categoryValue)}
                  onHighlightCells={(categoryValue) => handleHighlightCategory(summary.name, categoryValue)}
                  checkedCategories={comparisonCheckedColumn === summary.name ? comparisonCheckedCategories : new Set<string>()}
                  onToggleCategory={(category) => handleCheckboxToggle(summary.name, category)}
                  onHide={() => hideColumn(summary.name)}
                  onRename={(newName) => setColumnDisplayName(summary.name, newName)}
                  onRenameLabel={(oldLabel, newLabel) => handleRenameLabel(summary.name, oldLabel, newLabel)}
                  onMergeLabels={() => setMergeModalColumn(summary.name)}
                  onTransferLabels={() => setTransferModalColumn(summary.name)}
                  labelsShown={embeddingLabelColumn === summary.name}
                  onToggleLabels={() => handleToggleLabels(summary.name)}
                  selectedCategorySource={selectedCategorySource}
                />
              ))}
            </div>
          ) : (
            <div style={styles.emptyState}>No categorical columns</div>
          )
        )}

        {cellTab === 'continuous' && (
          continuousColumns.length > 0 ? (
            <div style={styles.section}>
              {continuousColumns.map((summary) => (
                <ContinuousColumn
                  key={summary.name}
                  summary={summary}
                  displayName={getDisplayName(summary.name)}
                  isActive={colorMode === 'metadata' && selectedColorColumn === summary.name}
                  onColorBy={() => handleColorBy(summary.name)}
                  onHide={() => hideColumn(summary.name)}
                  onRename={(newName) => setColumnDisplayName(summary.name, newName)}
                />
              ))}
            </div>
          ) : (
            <div style={styles.emptyState}>No continuous columns</div>
          )
        )}
```

(Leave "Add Annotation" above and the "Hidden Columns" block + empty-state below unchanged. `categoricalExpanded`/`continuousExpanded` become unused — remove those two `useState` lines to keep `tsc` clean.)

- [ ] **Step 3: Remove now-unused state**

Delete the lines `const [categoricalExpanded, setCategoricalExpanded] = useState(true)` and `const [continuousExpanded, setContinuousExpanded] = useState(true)`.

- [ ] **Step 4: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: no errors (no unused-var errors for the removed state).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/CellPanel.tsx
git commit -m "$(printf 'frontend: Cells pane Categorical | Continuous tabs\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 2: Genes pane — Sets | Color tabs

**Files:** Modify `frontend/src/components/GenePanel.tsx`

- [ ] **Step 1: Add tab state**

In `GenePanel()` near the other `useState`s (~:1699, where `showBrowse` is declared) add:

```tsx
  const [geneTab, setGeneTab] = useState<'sets' | 'color'>('sets')
```

- [ ] **Step 2: Add the tab strip at the top of the content area**

In the render, immediately AFTER `<div style={styles.content}>` (~:1918) and BEFORE the `{/* New Set Input … */}` block, insert:

```tsx
        {/* Sets | Color tabs */}
        <div style={{ display: 'flex', gap: '4px', marginBottom: '10px' }}>
          {([['sets', 'Sets'], ['color', 'Color']] as const).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setGeneTab(key)}
              style={{
                flex: 1, padding: '5px 8px', fontSize: '11px', cursor: 'pointer',
                backgroundColor: geneTab === key ? '#0f3460' : 'transparent',
                color: geneTab === key ? '#4ecdc4' : '#888',
                border: '1px solid ' + (geneTab === key ? '#4ecdc4' : '#1a1a2e'),
                borderRadius: '4px',
              }}
            >
              {label}
            </button>
          ))}
        </div>

        {geneTab === 'sets' && (
        <>
```

- [ ] **Step 3: Close the Sets group and open the Color group between the two content regions**

The "Sets" content runs from the New-Set-Input block through the **Find Similar Genes** section. The "Color" content is the **Bivariate Mode Section** + **HighlightOverlayPanel**. Between the closing of the Find-Similar block (the `)}` ending the `{hasGeneNeighbors && ( … )}` at ~:2063) and the `{/* Bivariate Mode Section */}` comment (~:2065), insert:

```tsx
        </>
        )}

        {geneTab === 'color' && (
        <>
```

- [ ] **Step 4: Close the Color group before the Hidden footer**

After the `<HighlightOverlayPanel … />` closing `/>` (~:2142) and BEFORE `<HiddenCategoriesFooter />` (~:2144), insert:

```tsx
        </>
        )}
```

(Net effect: New-Set/Folder inputs + gene-set categories + Find Similar render under **Sets**; Bivariate + Highlight render under **Color**; `HiddenCategoriesFooter` and the status bars stay always-visible. `GeneSearch` stays in the header, always visible.)

- [ ] **Step 5: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/GenePanel.tsx
git commit -m "$(printf 'frontend: Genes pane Sets | Color tabs (de-crowd Bivariate + Highlight)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Phase 2 — `.var` columns as gene sets

### Task 3: Backend `/var/column_genes`

**Files:**
- Modify `backend/xcell/adaptor.py`, `backend/xcell/api/routes.py`
- Create `backend/tests/test_var_column_genes.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_var_column_genes.py`:

```python
"""Resolve a boolean .var column to its gene names."""
import numpy as np
import anndata
import pytest
from scipy.sparse import csr_matrix
from fastapi.testclient import TestClient

from xcell.adaptor import DataAdaptor
from xcell.main import app
from xcell.api import routes


def _adata():
    rng = np.random.default_rng(0)
    ad = anndata.AnnData(X=csr_matrix(rng.random((20, 6)).astype(np.float32)))
    ad.var_names = [f"g{i}" for i in range(6)]
    ad.var["highly_variable"] = [True, False, True, True, False, False]
    ad.var["means"] = rng.random(6)  # numeric, not boolean
    return ad


def test_column_to_gene_names_returns_true_genes_in_order():
    a = DataAdaptor("x.h5ad", adata=_adata())
    assert a.column_to_gene_names("highly_variable") == ["g0", "g2", "g3"]


def test_column_to_gene_names_rejects_missing_column():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError):
        a.column_to_gene_names("nope")


def test_column_to_gene_names_rejects_non_boolean_column():
    a = DataAdaptor("x.h5ad", adata=_adata())
    with pytest.raises(ValueError):
        a.column_to_gene_names("means")


def test_route_var_column_genes(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.get("/api/var/column_genes", params={"column": "highly_variable"})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"column": "highly_variable", "genes": ["g0", "g2", "g3"]}


def test_route_var_column_genes_bad_column(monkeypatch):
    a = DataAdaptor("x.h5ad", adata=_adata())
    monkeypatch.setattr(routes, "get_adaptor", lambda dataset=None: a)
    client = TestClient(app)
    resp = client.get("/api/var/column_genes", params={"column": "means"})
    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_var_column_genes.py -q`
Expected: FAIL (`DataAdaptor` has no attribute `column_to_gene_names`).

- [ ] **Step 3: Implement the adaptor method**

In `backend/xcell/adaptor.py`, immediately after `get_var_boolean_columns` (ends ~:3314) add:

```python
    def column_to_gene_names(self, column: str) -> list[str]:
        """Gene names where a boolean .var column is True (in .var order).

        Raises ValueError if the column is absent or not boolean-like (the
        allow-list is exactly what get_var_boolean_columns reports).
        """
        valid = {c['name'] for c in self.get_var_boolean_columns()}
        if column not in valid:
            raise ValueError(
                f"'{column}' is not a boolean .var column"
            )
        mask = self._column_to_bool_array(column)
        return self.adata.var_names[mask].tolist()
```

- [ ] **Step 4: Implement the route**

In `backend/xcell/api/routes.py`, after the `get_var_boolean_column_values` route (~:2074) add:

```python
@router.get("/var/column_genes")
def get_var_column_genes(
    column: str = Query(...), dataset: str | None = Query(None)
):
    """Gene names where a boolean .var column is True."""
    adaptor = get_adaptor(dataset)
    try:
        genes = adaptor.column_to_gene_names(column)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"column": column, "genes": genes}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && pixi run -e dev python -m pytest tests/test_var_column_genes.py -q`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/xcell/adaptor.py backend/xcell/api/routes.py backend/tests/test_var_column_genes.py
git commit -m "$(printf 'backend: GET /var/column_genes resolves a boolean .var column to gene names\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 4: Frontend data hooks

**Files:** Modify `frontend/src/hooks/useData.ts`

- [ ] **Step 1: Add the two fetch helpers**

Append near the other exported fetchers in `frontend/src/hooks/useData.ts`:

```ts
export interface VarBooleanColumn {
  name: string
  n_true: number
  n_total: number
}

export async function fetchVarBooleanColumns(
  slot?: DatasetSlot,
): Promise<VarBooleanColumn[]> {
  const res = await fetch(appendDataset(`${API_BASE}/var/boolean_columns`, slot))
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function fetchVarColumnGenes(
  column: string,
  slot?: DatasetSlot,
): Promise<string[]> {
  const url = appendDataset(`${API_BASE}/var/column_genes?column=${encodeURIComponent(column)}`, slot)
  const res = await fetch(url)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  const body = await res.json()
  return body.genes as string[]
}
```

Note: confirm `appendDataset` supports a query-string-bearing path; if it appends `?dataset=` with `?`, switch the column param to `&` form or use the existing pattern in this file (grep `appendDataset(` for an example that already includes a `?`). If none exists, build the URL as `appendDataset(\`${API_BASE}/var/column_genes\`, slot)` then append `column` with the correct separator based on whether the result already contains `?`.

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useData.ts
git commit -m "$(printf 'frontend: useData fetchers for .var boolean columns + column genes\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 5: `.var` columns section in the Sets tab

**Files:**
- Create `frontend/src/components/VarColumnsSection.tsx`
- Modify `frontend/src/components/GenePanel.tsx`

- [ ] **Step 1: Create the section component**

Create `frontend/src/components/VarColumnsSection.tsx`:

```tsx
import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { fetchVarBooleanColumns, fetchVarColumnGenes, type VarBooleanColumn } from '../hooks/useData'

/** Lists boolean .var columns; each can be materialized as a frozen gene set. */
export default function VarColumnsSection() {
  const addGeneSetToCategory = useStore((s) => s.addGeneSetToCategory)
  const [columns, setColumns] = useState<VarBooleanColumn[]>([])
  const [expanded, setExpanded] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  useEffect(() => {
    fetchVarBooleanColumns()
      .then(setColumns)
      .catch(() => setColumns([]))
  }, [])

  if (columns.length === 0) return null

  const addColumn = async (name: string) => {
    setBusy(name)
    setError(null)
    try {
      const genes = await fetchVarColumnGenes(name)
      addGeneSetToCategory('manual', name, genes)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div style={{ marginBottom: '12px' }}>
      <div
        onClick={() => setExpanded(!expanded)}
        style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '11px', color: '#888', marginBottom: '4px' }}
      >
        <span>{expanded ? '▼' : '▶'}</span>
        <span>.var columns ({columns.length})</span>
      </div>
      {expanded && (
        <div style={{ backgroundColor: '#0f3460', borderRadius: '4px', padding: '6px 8px' }}>
          {columns.map((c) => (
            <div key={c.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px', color: '#ccc', padding: '2px 0' }}>
              <span title={`${c.n_true} of ${c.n_total} genes`}>{c.name} <span style={{ color: '#888' }}>({c.n_true})</span></span>
              <button
                onClick={() => addColumn(c.name)}
                disabled={busy === c.name}
                title="Add as a gene set in Manual"
                style={{ padding: '1px 8px', fontSize: '11px', backgroundColor: '#16213e', color: '#4ecdc4', border: '1px solid #4ecdc4', borderRadius: '3px', cursor: busy === c.name ? 'wait' : 'pointer' }}
              >
                {busy === c.name ? '…' : '+'}
              </button>
            </div>
          ))}
          {error && <div style={{ color: '#e94560', fontSize: '10px', marginTop: '4px' }}>{error}</div>}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: Mount it in the Sets tab**

In `frontend/src/components/GenePanel.tsx`, add the import near the other component imports (top of file):

```tsx
import VarColumnsSection from './VarColumnsSection'
```

Then, inside the `{geneTab === 'sets' && (<>` group, immediately AFTER the gene-set categories `.map(...)` block closes (`})}` at ~:1982) and BEFORE the `{/* Find Similar Genes Section */}` block, insert:

```tsx
        <VarColumnsSection />
```

- [ ] **Step 3: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/VarColumnsSection.tsx frontend/src/components/GenePanel.tsx
git commit -m "$(printf 'frontend: .var columns section materializes columns as gene sets\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Phase 3 — Combine modal (set operations)

### Task 6: Pure set-operation util

**Files:** Create `frontend/src/lib/geneSetOps.ts`

- [ ] **Step 1: Create the util**

Create `frontend/src/lib/geneSetOps.ts`:

```ts
/** Set operations over gene-name arrays. Each returns a de-duplicated array
 *  with stable order = first appearance in A, then new members of B.
 *  Matching is exact-string (gene names already match the .var index). */

export type SetOp = 'union' | 'intersection' | 'difference' | 'symmetric'

const dedupe = (xs: string[]): string[] => Array.from(new Set(xs))

export function union(a: string[], b: string[]): string[] {
  return dedupe([...a, ...b])
}

export function intersection(a: string[], b: string[]): string[] {
  const sb = new Set(b)
  return dedupe(a).filter((g) => sb.has(g))
}

export function difference(a: string[], b: string[]): string[] {
  const sb = new Set(b)
  return dedupe(a).filter((g) => !sb.has(g))
}

export function symmetricDifference(a: string[], b: string[]): string[] {
  const sa = new Set(a)
  const sb = new Set(b)
  return [...dedupe(a).filter((g) => !sb.has(g)), ...dedupe(b).filter((g) => !sa.has(g))]
}

export function applyOp(op: SetOp, a: string[], b: string[]): string[] {
  switch (op) {
    case 'union': return union(a, b)
    case 'intersection': return intersection(a, b)
    case 'difference': return difference(a, b)
    case 'symmetric': return symmetricDifference(a, b)
  }
}

export const OP_SYMBOL: Record<SetOp, string> = {
  union: '∪', intersection: '∩', difference: '−', symmetric: '△',
}

export const OP_LABEL: Record<SetOp, string> = {
  union: 'Union (A ∪ B)',
  intersection: 'Intersection (A ∩ B)',
  difference: 'Difference (A − B)',
  symmetric: 'Symmetric difference (A △ B)',
}
```

- [ ] **Step 2: Typecheck**

Run: `cd frontend && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/geneSetOps.ts
git commit -m "$(printf 'frontend: pure gene-set operation util (union/intersect/diff/symdiff)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 7: Combine modal + store flag + button

**Files:**
- Create `frontend/src/components/CombineGeneSetsModal.tsx`
- Modify `frontend/src/store.ts`, `frontend/src/components/GenePanel.tsx`

- [ ] **Step 1: Add the store flag**

In `frontend/src/store.ts`, mirror the existing `isImportModalOpen` flag (the
convention is an `is*` boolean field + a `set*` setter):

- Interface (near `isImportModalOpen: boolean` ~:651) add:

```ts
  isCombineModalOpen: boolean
  setCombineModalOpen: (open: boolean) => void
```

- Initial state (near `isImportModalOpen: false,` ~:1024) add:

```ts
  isCombineModalOpen: false,
```

- Action (near `setImportModalOpen:` ~:2035) add:

```ts
  setCombineModalOpen: (open) => set({ isCombineModalOpen: open }),
```

- [ ] **Step 2: Create the modal**

Create `frontend/src/components/CombineGeneSetsModal.tsx`:

```tsx
import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { fetchVarBooleanColumns, fetchVarColumnGenes, type VarBooleanColumn } from '../hooks/useData'
import { applyOp, OP_LABEL, OP_SYMBOL, type SetOp } from '../lib/geneSetOps'

type Operand = { kind: 'set'; id: string; label: string; genes: string[] }
             | { kind: 'col'; name: string; label: string }

export default function CombineGeneSetsModal() {
  const open = useStore((s) => s.isCombineModalOpen)
  const setOpen = useStore((s) => s.setCombineModalOpen)
  const categories = useStore((s) => s.geneSetCategories)
  const addGeneSetToCategory = useStore((s) => s.addGeneSetToCategory)

  const [columns, setColumns] = useState<VarBooleanColumn[]>([])
  const [aKey, setAKey] = useState('')
  const [bKey, setBKey] = useState('')
  const [op, setOp] = useState<SetOp>('intersection')
  const [name, setName] = useState('')
  const [nameEdited, setNameEdited] = useState(false)
  const [colCache, setColCache] = useState<Record<string, string[]>>({})
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // All operands: every gene set (any category) + every .var boolean column.
  const operands = useMemo<Operand[]>(() => {
    const out: Operand[] = []
    for (const cat of Object.values(categories)) {
      const collect = (sets: { id: string; name: string; genes: string[] }[]) => {
        for (const gs of sets) out.push({ kind: 'set', id: gs.id, label: `${cat.name}: ${gs.name} (${gs.genes.length})`, genes: gs.genes })
      }
      collect(cat.geneSets)
      for (const f of cat.folders) collect(f.geneSets)
    }
    for (const c of columns) out.push({ kind: 'col', name: c.name, label: `.var: ${c.name} (${c.n_true})` })
    return out
  }, [categories, columns])

  useEffect(() => {
    if (!open) return
    fetchVarBooleanColumns().then(setColumns).catch(() => setColumns([]))
    setAKey(''); setBKey(''); setOp('intersection'); setName(''); setNameEdited(false); setError(null)
  }, [open])

  const keyOf = (o: Operand) => (o.kind === 'set' ? `set:${o.id}` : `col:${o.name}`)
  const findOperand = (key: string) => operands.find((o) => keyOf(o) === key) || null

  const resolve = async (o: Operand | null): Promise<string[]> => {
    if (!o) return []
    if (o.kind === 'set') return o.genes
    if (colCache[o.name]) return colCache[o.name]
    const genes = await fetchVarColumnGenes(o.name)
    setColCache((c) => ({ ...c, [o.name]: genes }))
    return genes
  }

  const a = findOperand(aKey)
  const b = findOperand(bKey)

  const [result, setResult] = useState<string[]>([])
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      if (!a || !b) { setResult([]); return }
      const [ga, gb] = [await resolve(a), await resolve(b)]
      if (!cancelled) setResult(applyOp(op, ga, gb))
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aKey, bKey, op, columns])

  // Auto-suggest a name from operands + operator unless the user edited it.
  const shortLabel = (o: Operand | null) => !o ? '?' : (o.kind === 'col' ? o.name : o.label.split(': ').slice(1).join(': ').replace(/\s*\(\d+\)$/, ''))
  const suggested = `${shortLabel(a)} ${OP_SYMBOL[op]} ${shortLabel(b)}`
  useEffect(() => { if (!nameEdited) setName(a && b ? suggested : '') }, [aKey, bKey, op]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!open) return null

  const canCreate = !!a && !!b && name.trim().length > 0 && !busy

  const handleCreate = async () => {
    if (!a || !b) return
    setBusy(true); setError(null)
    try {
      const genes = applyOp(op, await resolve(a), await resolve(b))
      addGeneSetToCategory('manual', name.trim(), genes)
      setOpen(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const selStyle = { width: '100%', padding: '6px 8px', fontSize: '12px', backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: '4px' } as const

  return (
    <div onClick={() => setOpen(false)} style={{ position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ backgroundColor: '#16213e', border: '1px solid #0f3460', borderRadius: '8px', padding: '20px 24px', minWidth: '420px', maxWidth: '520px', color: '#eee' }}>
        <div style={{ fontSize: '15px', fontWeight: 600, marginBottom: '12px' }}>Combine gene sets</div>

        <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>A</label>
        <select value={aKey} onChange={(e) => setAKey(e.target.value)} style={{ ...selStyle, marginBottom: '10px' }}>
          <option value="">Select…</option>
          {operands.map((o) => <option key={keyOf(o)} value={keyOf(o)}>{o.label}</option>)}
        </select>

        <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>Operator</label>
        <select value={op} onChange={(e) => setOp(e.target.value as SetOp)} style={{ ...selStyle, marginBottom: '10px' }}>
          {(['union', 'intersection', 'difference', 'symmetric'] as SetOp[]).map((k) => <option key={k} value={k}>{OP_LABEL[k]}</option>)}
        </select>

        <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>B</label>
        <select value={bKey} onChange={(e) => setBKey(e.target.value)} style={{ ...selStyle, marginBottom: '12px' }}>
          <option value="">Select…</option>
          {operands.map((o) => <option key={keyOf(o)} value={keyOf(o)}>{o.label}</option>)}
        </select>

        <div style={{ fontSize: '12px', color: '#4ecdc4', marginBottom: '12px' }}>
          Result: <span style={{ color: '#eee' }}>{a && b ? `${result.length} genes` : '—'}</span>
        </div>

        <label style={{ display: 'block', fontSize: '11px', color: '#888', marginBottom: '4px' }}>New set name</label>
        <input value={name} onChange={(e) => { setName(e.target.value); setNameEdited(true) }} placeholder="name…" style={{ ...selStyle, marginBottom: '6px' }} />

        {error && <div style={{ color: '#e94560', fontSize: '11px', marginBottom: '8px' }}>{error}</div>}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '10px' }}>
          <button onClick={() => setOpen(false)} style={{ padding: '6px 12px', fontSize: '12px', backgroundColor: 'transparent', color: '#888', border: '1px solid #888', borderRadius: '4px', cursor: 'pointer' }}>Cancel</button>
          <button onClick={handleCreate} disabled={!canCreate} style={{ padding: '6px 14px', fontSize: '12px', fontWeight: 500, backgroundColor: canCreate ? '#4ecdc4' : '#1a1a2e', color: canCreate ? '#000' : '#666', border: 'none', borderRadius: '4px', cursor: canCreate ? 'pointer' : 'not-allowed' }}>Create</button>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 3: Mount the modal + add the Combine button**

In `frontend/src/components/GenePanel.tsx`:

Add the import (top of file):

```tsx
import CombineGeneSetsModal from './CombineGeneSetsModal'
```

Pull the setter from the store (in the existing `useStore()` destructure near `setImportModalOpen`, add `setCombineModalOpen`):

```tsx
  const setCombineModalOpen = useStore((s) => s.setCombineModalOpen)
```

(If the panel reads the store via a single `useStore()` destructure object rather than selectors, add `setCombineModalOpen` to that destructure instead — match the file's existing pattern.)

Mount the modal next to `<ImportModal />` (~:1892):

```tsx
      <CombineGeneSetsModal />
```

Add the **Combine** button inside the `{geneTab === 'sets' && (<>` group, right after `<VarColumnsSection />`:

```tsx
        <button
          onClick={() => setCombineModalOpen(true)}
          style={{ width: '100%', padding: '7px', fontSize: '12px', marginBottom: '12px', backgroundColor: '#0f3460', color: '#4ecdc4', border: '1px solid #1a1a2e', borderRadius: '4px', cursor: 'pointer' }}
        >
          ⨂ Combine sets…
        </button>
```

- [ ] **Step 4: Typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/store.ts frontend/src/components/CombineGeneSetsModal.tsx frontend/src/components/GenePanel.tsx
git commit -m "$(printf 'frontend: Combine gene sets modal (set operations over sets + .var columns)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

### Task 8: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Full backend suite**

Run: `cd backend && pixi run -e dev python -m pytest tests/ -q`
Expected: all green (previous total + 5 new).

- [ ] **Step 2: Frontend typecheck + build**

Run: `cd frontend && npx tsc --noEmit && npm run build`
Expected: clean.

- [ ] **Step 3: Playwright smoke on isolated stack**

Start the isolated backend (:8001, toy dataset) + vite (:5180) per the Conventions block. Then with Playwright MCP at `http://127.0.0.1:5180`:
- Cells pane: confirm `Categorical | Continuous` tabs switch the list.
- Genes pane: confirm `Sets | Color` tabs; Bivariate Coloring + Highlight appear only under **Color**; the `.var columns` list and `⨂ Combine sets…` appear under **Sets**.
- Click `+` on a `.var` column (e.g. a boolean column present in the toy data; if none, add `highly_variable` via Scanpy HVG first, or use a dataset that has one) → a new Manual gene set named after the column appears.
- Open Combine, pick two operands, an operator → the Result count updates; Create → a new Manual set appears.

Stop both servers and delete any screenshot PNGs / `.playwright-mcp` artifacts.

- [ ] **Step 4: Append an implemented note to the spec**

Add a short "Implemented (2026-06-22)" line near the top of
`docs/superpowers/specs/2026-06-22-gene-set-algebra-and-pane-tabs-design.md`.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "$(printf 'gene-set algebra + pane tabs: verified end-to-end\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-review notes
- **Spec coverage:** `.var` columns as sets (Tasks 3–5) ✓; set operations (Tasks 6–7) ✓; Genes tabs (Task 2) ✓; Cells tabs (Task 1) ✓; frozen snapshots → Manual (Tasks 5, 7) ✓; boolean-only (Task 3 allow-list) ✓; testing (Tasks 3, 8) ✓.
- **Naming consistency:** `column_to_gene_names` (adaptor), `GET /var/column_genes`, `fetchVarBooleanColumns`/`fetchVarColumnGenes` + `VarBooleanColumn`, `geneSetOps` (`union`/`intersection`/`difference`/`symmetricDifference`/`applyOp`/`SetOp`/`OP_LABEL`/`OP_SYMBOL`), `combineModalOpen`/`setCombineModalOpen`, `VarColumnsSection`, `CombineGeneSetsModal` — used identically across tasks.
- **Backend shape:** `get_var_boolean_columns()` returns `{name, n_true, n_total}` (matches `VarBooleanColumn`).
- **Risk note:** Task 4 Step 1 flags the `appendDataset` + query-string interaction to confirm before relying on it.
- **Out of scope (untouched):** live/dynamic sets; numeric-threshold columns; backend set math.
