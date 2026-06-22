import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { fetchVarBooleanColumns, fetchVarColumnGenes, type VarBooleanColumn } from '../hooks/useData'
import { applyOp, OP_LABEL, OP_SYMBOL, type SetOp } from '../lib/geneSetOps'

type Operand =
  | { kind: 'set'; id: string; label: string; genes: string[] }
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
  const [result, setResult] = useState<string[]>([])
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
    setAKey(''); setBKey(''); setOp('intersection'); setName(''); setNameEdited(false); setError(null); setResult([])
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

  // Live result count.
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
  const shortLabel = (o: Operand | null) =>
    !o ? '?' : (o.kind === 'col' ? o.name : o.label.split(': ').slice(1).join(': ').replace(/\s*\(\d+\)$/, ''))
  useEffect(() => {
    if (!nameEdited) setName(a && b ? `${shortLabel(a)} ${OP_SYMBOL[op]} ${shortLabel(b)}` : '')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aKey, bKey, op])

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
