/** Choosing which genes an analysis runs on: boolean `.var` columns, or gene
 *  sets from the Gene Panel.
 *
 *  Shared by the single- and multi-line association modals, which had the
 *  column half of this duplicated verbatim.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { useStore, type GeneSet } from '../store'
import { appendDataset } from '../hooks/useData'
import { flattenGeneSets } from './GenePanel'
import {
  resolveGeneSubset,
  geneSubsetLabel,
  type GeneSubsetOperation,
  type GeneSubsetSelection,
  type GeneSubsetSpec,
} from '../lib/geneSubset'

interface BooleanColumn {
  name: string
  n_true: number
  n_total: number
}

/** What the modal keeps. Gene sets are held by id and resolved against the
 *  live store on every render, so a set edited while the modal is open is
 *  sent as it now stands rather than as it was when clicked. */
export interface GeneSubsetChoice {
  columns: string[]
  geneSetIds: string[]
  operation: GeneSubsetOperation
}

export interface GeneSubsetControl {
  choice: GeneSubsetChoice
  setChoice: (choice: GeneSubsetChoice) => void
  /** The choice resolved against the store — what `resolveGeneSubset` reads. */
  selection: GeneSubsetSelection
  /** Ready to send as the `gene_subset` request field. */
  spec: GeneSubsetSpec
  /** One-line description of the current choice. */
  label: string
  availableGeneSets: GeneSet[]
}

const EMPTY: GeneSubsetChoice = { columns: [], geneSetIds: [], operation: 'intersection' }

export function useGeneSubset(): GeneSubsetControl {
  const geneSetCategories = useStore((s) => s.geneSetCategories)
  const [choice, setChoice] = useState<GeneSubsetChoice>(EMPTY)

  const availableGeneSets = useMemo(
    () => flattenGeneSets(geneSetCategories).filter((gs) => gs.genes.length > 0),
    [geneSetCategories],
  )

  const selection = useMemo<GeneSubsetSelection>(() => ({
    columns: choice.columns,
    // Panel order, not click order: it matches what the label reads back.
    geneSets: availableGeneSets.filter((gs) => choice.geneSetIds.includes(gs.id)),
    operation: choice.operation,
  }), [choice, availableGeneSets])

  // Memoized: resolveGeneSubset builds a fresh array, and callers put `spec`
  // in useCallback dependency lists.
  const spec = useMemo(() => resolveGeneSubset(selection), [selection])
  const label = useMemo(() => geneSubsetLabel(selection), [selection])

  return { choice, setChoice, selection, spec, label, availableGeneSets }
}

const dark = {
  title: { fontSize: '11px', color: '#888', marginBottom: '6px' },
  pills: { display: 'flex', flexWrap: 'wrap' as const, gap: '4px' },
  pill: {
    padding: '3px 8px',
    fontSize: '10px',
    borderRadius: '10px',
    cursor: 'pointer',
    border: '1px solid #1a1a2e',
  },
  toggle: {
    padding: '3px 10px',
    fontSize: '10px',
    borderRadius: '4px',
    cursor: 'pointer',
    border: '1px solid #1a1a2e',
  },
  group: { marginTop: '8px' },
  note: { marginTop: '6px', fontSize: '10px', color: '#e9a23b' },
}

const selected = (on: boolean) => ({
  backgroundColor: on ? '#4ecdc4' : '#0f3460',
  color: on ? '#000' : '#aaa',
  borderColor: on ? '#4ecdc4' : '#1a1a2e',
  fontWeight: on ? 600 : 400,
})

function OperationToggle({
  operation,
  onChange,
}: {
  operation: GeneSubsetOperation
  onChange: (op: GeneSubsetOperation) => void
}) {
  return (
    <div style={{ marginTop: '6px', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px' }}>
      <span style={{ color: '#888' }}>Combine:</span>
      {(['intersection', 'union'] as const).map((op) => (
        <button
          key={op}
          onClick={() => onChange(op)}
          style={{ ...dark.toggle, ...selected(operation === op) }}
          title={op === 'intersection' ? 'Genes in every selection' : 'Genes in any selection'}
        >
          {op === 'intersection' ? 'AND' : 'OR'}
        </button>
      ))}
    </div>
  )
}

export default function GeneSubsetPicker({ control }: { control: GeneSubsetControl }) {
  const { choice, setChoice, selection, label, availableGeneSets } = control
  const scanpyActionHistory = useStore((s) => s.scanpyActionHistory)
  const [columns, setColumns] = useState<BooleanColumn[]>([])

  useEffect(() => {
    fetch(appendDataset('/api/var/boolean_columns'))
      .then((res) => res.json())
      .then(setColumns)
      .catch(() => setColumns([]))
  }, [scanpyActionHistory])

  // Columns and gene sets are exclusive — the backend has no way to express
  // "this signature AND highly_variable", and narrowing a curated set by a
  // .var flag would drop the genes the user came to test.
  const toggleColumn = useCallback((name: string) => {
    const on = choice.columns.includes(name)
    setChoice({
      ...choice,
      columns: on ? choice.columns.filter((c) => c !== name) : [...choice.columns, name],
      geneSetIds: [],
    })
  }, [choice, setChoice])

  const toggleGeneSet = useCallback((id: string) => {
    const on = choice.geneSetIds.includes(id)
    setChoice({
      ...choice,
      geneSetIds: on ? choice.geneSetIds.filter((g) => g !== id) : [...choice.geneSetIds, id],
      columns: [],
    })
  }, [choice, setChoice])

  const setOperation = useCallback(
    (operation: GeneSubsetOperation) => setChoice({ ...choice, operation }),
    [choice, setChoice],
  )

  const nSelected = choice.columns.length + choice.geneSetIds.length
  const emptyCombination = Array.isArray(control.spec) && control.spec.length === 0

  if (columns.length === 0 && availableGeneSets.length === 0) return null

  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={dark.title}>Genes: <span style={{ color: '#ccc' }}>{label}</span></div>

      {columns.length > 0 && (
        <div style={dark.pills}>
          {columns.map((col) => (
            <button
              key={col.name}
              onClick={() => toggleColumn(col.name)}
              style={{ ...dark.pill, ...selected(choice.columns.includes(col.name)) }}
              title={`${col.n_true.toLocaleString()} of ${col.n_total.toLocaleString()} genes`}
            >
              {col.name} ({col.n_true.toLocaleString()})
            </button>
          ))}
        </div>
      )}

      {availableGeneSets.length > 0 && (
        <div style={columns.length > 0 ? dark.group : undefined}>
          {columns.length > 0 && (
            <div style={{ ...dark.title, marginBottom: '4px' }}>Gene sets</div>
          )}
          <div style={dark.pills}>
            {availableGeneSets.map((gs) => (
              <button
                key={gs.id}
                onClick={() => toggleGeneSet(gs.id)}
                style={{ ...dark.pill, ...selected(choice.geneSetIds.includes(gs.id)) }}
                title={`${gs.genes.length.toLocaleString()} genes — genes absent from this dataset are reported with the result`}
              >
                {gs.name} ({gs.genes.length.toLocaleString()})
              </button>
            ))}
          </div>
        </div>
      )}

      {nSelected >= 2 && <OperationToggle operation={selection.operation} onChange={setOperation} />}

      {emptyCombination && (
        <div style={dark.note}>
          No genes in common — switch to OR, or drop a set.
        </div>
      )}
    </div>
  )
}
