import { useDeferredValue, useEffect, useMemo, useRef, useState, type CSSProperties, type DragEvent } from 'react'
import {
  useStore,
  generateFolderId,
  generateGeneSetId,
  type GeneSet,
  type GeneSetCategoryType,
  type GeneSetFolder,
  type GeneSetManagerSource,
  type GeneSetManagerTab,
} from '../store'
import { appendDataset } from '../hooks/useData'
import ConfirmDialog from './ConfirmDialog'
import {
  EMPTY_SELECTION,
  MANAGER_ORDER,
  NEW_FOLDER,
  addFolder,
  countGeneSets,
  deleteSelection,
  destinationFromValue,
  destinationOptions,
  destinationValue,
  filterTree,
  findGeneSet,
  folderKey,
  gatherSelectedSets,
  insertGeneSet,
  makeGeneIndex,
  mergeGeneSets,
  mergeSetsInto,
  moveSets,
  remapSelection,
  renameFolder,
  parseGeneText,
  resolveGenes,
  selectAllShown,
  selectionSummary,
  setKey,
  shownState,
  suggestMergeName,
  toggleSet,
  toggleShown,
  updateGeneSet,
  type Categories,
  type Destination,
  type FilteredCategory,
  type GeneIndex,
  type GeneResolution,
  type MergeRule,
  type Selection,
  type TickState,
} from '../lib/geneSetManager'

const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`

const styles: Record<string, CSSProperties> = {
  overlay: {
    position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.6)', zIndex: 1000,
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  card: {
    backgroundColor: '#16213e', border: '1px solid #0f3460', borderRadius: '8px', color: '#eee',
    width: 'min(1040px, 95vw)', height: 'min(700px, 90vh)', display: 'flex', flexDirection: 'column',
    boxShadow: '0 10px 40px rgba(0,0,0,0.5)',
  },
  header: {
    display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px',
    borderBottom: '1px solid #0f3460',
  },
  body: { flex: 1, display: 'flex', minHeight: 0 },
  left: { flex: '1 1 55%', display: 'flex', flexDirection: 'column', borderRight: '1px solid #0f3460', minWidth: 0 },
  right: { flex: '1 1 45%', display: 'flex', flexDirection: 'column', minWidth: 0 },
  input: {
    width: '100%', boxSizing: 'border-box', padding: '6px 8px', fontSize: '12px',
    backgroundColor: '#0f3460', color: '#eee', border: '1px solid #1a1a2e', borderRadius: '4px',
  },
  textarea: {
    width: '100%', boxSizing: 'border-box', padding: '6px 8px', fontSize: '12px', resize: 'vertical',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
    backgroundColor: '#0f1625', color: '#eee', border: '1px solid #0f3460', borderRadius: '4px',
  },
  label: { display: 'block', fontSize: '11px', color: '#888', margin: '10px 0 4px' },
  smallButton: {
    padding: '3px 8px', fontSize: '11px', backgroundColor: '#0f3460', color: '#ccc',
    border: '1px solid #1a1a2e', borderRadius: '3px', cursor: 'pointer', whiteSpace: 'nowrap',
  },
  primaryButton: {
    padding: '7px 16px', fontSize: '12px', fontWeight: 600, backgroundColor: '#4ecdc4', color: '#000',
    border: 'none', borderRadius: '4px', cursor: 'pointer', whiteSpace: 'nowrap', flex: '0 0 auto',
  },
  disabledButton: {
    padding: '7px 16px', fontSize: '12px', fontWeight: 600, backgroundColor: '#1a1a2e', color: '#666',
    border: 'none', borderRadius: '4px', cursor: 'not-allowed', whiteSpace: 'nowrap', flex: '0 0 auto',
  },
  row: {
    display: 'flex', alignItems: 'center', gap: '6px', padding: '3px 8px', fontSize: '12px',
    borderRadius: '3px', minHeight: '22px',
  },
  count: { marginLeft: 'auto', fontSize: '10px', color: '#888', whiteSpace: 'nowrap', paddingLeft: '8px' },
  hint: { fontSize: '12px', color: '#888', lineHeight: 1.5, padding: '16px 4px' },
}

function TriCheckbox({ state, onChange, title }: { state: TickState; onChange: () => void; title?: string }) {
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = state === 'some'
  }, [state])
  return (
    <input
      ref={ref}
      type="checkbox"
      checked={state === 'all'}
      onChange={onChange}
      onClick={(e) => e.stopPropagation()}
      title={title}
      style={{ margin: 0, cursor: 'pointer', flex: '0 0 auto' }}
    />
  )
}

const geneCountLabel = (gs: GeneSet) =>
  gs.genesDown && gs.genesDown.length > 0 ? `↑${gs.genes.length} ↓${gs.genesDown.length}` : `${gs.genes.length}`

/** What a pasted list resolves to against the dataset, in one or two lines. */
function ResolutionSummary({
  res, duplicates, keepMissing, onKeepMissing, keepLabel,
}: {
  res: GeneResolution
  duplicates: number
  keepMissing: boolean
  onKeepMissing: (v: boolean) => void
  keepLabel: string
}) {
  const [showRecased, setShowRecased] = useState(false)
  if (!res.checked) {
    return (
      <div style={{ fontSize: '11px', color: '#888', marginTop: '6px' }}>
        {plural(res.genes.length, 'gene')} · no dataset loaded, so they are kept as typed
      </div>
    )
  }
  const typed = res.found + res.missing.length
  if (typed === 0) return null
  return (
    <div style={{ fontSize: '11px', marginTop: '6px', lineHeight: 1.6 }}>
      <span style={{ color: '#4ecdc4' }}>✓ {res.found.toLocaleString()} in this dataset</span>
      {duplicates > 0 && <span style={{ color: '#888' }}> · {plural(duplicates, 'duplicate')} dropped</span>}
      {res.recased.length > 0 && (
        <>
          <span style={{ color: '#888' }}> · </span>
          <button
            onClick={() => setShowRecased((v) => !v)}
            style={{ background: 'none', border: 'none', padding: 0, color: '#e9a23b', cursor: 'pointer', fontSize: '11px' }}
            title="Matched ignoring case; stored in the dataset's spelling"
          >
            {res.recased.length} case-corrected {showRecased ? '▾' : '▸'}
          </button>
        </>
      )}
      {showRecased && (
        <div style={{ color: '#aaa', maxHeight: '60px', overflowY: 'auto' }}>
          {res.recased.map((r) => `${r.from} → ${r.to}`).join(', ')}
        </div>
      )}
      {res.missing.length > 0 && (
        <div>
          <span style={{ color: '#e94560' }}>✗ {res.missing.length} not found:</span>{' '}
          <span style={{ color: '#aaa' }}>
            {res.missing.slice(0, 12).join(', ')}{res.missing.length > 12 ? `, … ${res.missing.length - 12} more` : ''}
          </span>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#aaa', cursor: 'pointer', marginTop: '2px' }}>
            <input type="checkbox" checked={keepMissing} onChange={(e) => onKeepMissing(e.target.checked)} style={{ margin: 0 }} />
            {keepLabel}
          </label>
        </div>
      )}
    </div>
  )
}

// Sets dragged inside the manager. Its own type, so a drag from the Genes
// pane (or anywhere else) is never mistaken for one of these.
const MANAGER_DRAG_TYPE = 'application/x-xcell-manager-sets'

// A small "3 sets" badge as the drag image when several ticked sets travel together.
function setDragBadge(e: DragEvent, text: string) {
  const el = document.createElement('div')
  el.textContent = text
  Object.assign(el.style, {
    position: 'fixed', top: '-100px', left: '-100px', padding: '3px 10px', background: '#0f3460',
    color: '#4ecdc4', border: '1px solid #4ecdc4', borderRadius: '10px', font: '12px sans-serif',
  })
  document.body.appendChild(el)
  e.dataTransfer.setDragImage(el, 12, 12)
  window.setTimeout(() => el.remove(), 0)
}

function DestinationSelect({
  cats, value, onChange, newFolderName, onNewFolderName,
}: {
  cats: Categories
  value: string
  onChange: (v: string) => void
  newFolderName: string
  onNewFolderName: (v: string) => void
}) {
  const options = destinationOptions(cats, MANAGER_ORDER)
  return (
    <>
      <select value={value} onChange={(e) => onChange(e.target.value)} style={styles.input}>
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
      {value === NEW_FOLDER && (
        <input
          value={newFolderName}
          onChange={(e) => onNewFolderName(e.target.value)}
          placeholder="New folder name…"
          style={{ ...styles.input, marginTop: '6px' }}
        />
      )}
    </>
  )
}

/** Resolve a destination choice, making the new folder if that was picked.
 *  Returns the categories to build on and where to put the set. */
function prepareDestination(
  cats: Categories, value: string, newFolderName: string,
): { cats: Categories; dest: Destination; label: string } {
  const dest = destinationFromValue(value)
  if (dest) {
    const c = cats[dest.cat]
    const folder = dest.folderId ? c.folders.find((f) => f.id === dest.folderId) : null
    return { cats, dest, label: folder ? `${c.name} › ${folder.name}` : c.name }
  }
  const name = newFolderName.trim()
  if (!name) throw new Error('Name the new folder.')
  const folder: GeneSetFolder = { id: generateFolderId(), name, expanded: true, createdAt: new Date().toISOString(), geneSets: [] }
  return { cats: addFolder(cats, 'manual', folder), dest: { cat: 'manual', folderId: folder.id }, label: `Manual › ${name}` }
}

// A destination value that no longer exists (its folder was deleted) falls
// back to Manual's top level rather than leaving the select blank.
function validDestination(cats: Categories, value: string): string {
  if (value === NEW_FOLDER) return value
  return destinationOptions(cats, MANAGER_ORDER).some((o) => o.value === value)
    ? value
    : destinationValue({ cat: 'manual', folderId: null })
}

export default function GeneSetManagerModal() {
  const source = useStore((s) => s.geneSetManager)
  const setSource = useStore((s) => s.setGeneSetManager)
  const cats = useStore((s) => s.geneSetCategories)
  const replace = useStore((s) => s.replaceGeneSetCategories)
  const setClearGeneSetsOpen = useStore((s) => s.setClearGeneSetsOpen)
  const hasData = useStore((s) => s.schema !== null)
  const activeSlot = useStore((s) => s.activeSlot)
  const currentVarIndex = useStore((s) => s.currentVarIndex)

  const [tab, setTab] = useState<GeneSetManagerTab>('new')
  const [query, setQuery] = useState('')
  const deferredQuery = useDeferredValue(query)
  const [selection, setSelection] = useState<Selection>(EMPTY_SELECTION)
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [collapsedCats, setCollapsedCats] = useState<Set<string>>(new Set())
  const [focusKey, setFocusKey] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)

  // Tree editing: a new folder, renaming in place, drag and drop
  const [creatingFolder, setCreatingFolder] = useState(false)
  const [folderDraft, setFolderDraft] = useState('')
  // 'f:<folderKey>' or 's:<setKey>' — the row whose name is being edited
  const [renaming, setRenaming] = useState<string | null>(null)
  const [renameDraft, setRenameDraft] = useState('')
  // Enter commits and the input's blur would commit again; this makes it once.
  const renameOpenRef = useRef(false)
  const [hoverRow, setHoverRow] = useState<string | null>(null)
  const [dragKeys, setDragKeys] = useState<string[] | null>(null)
  // 'c:<cat>' | 'f:<folderKey>' | 's:<setKey>'
  const [dropTarget, setDropTarget] = useState<string | null>(null)
  const expandTimerRef = useRef<{ key: string; id: number } | null>(null)
  // The last move or merge, undoable while nothing else has changed the tree.
  const [undo, setUndo] = useState<{
    label: string; before: Categories; after: Categories; selection: Selection; focusKey: string | null
  } | null>(null)

  // New tab
  const [newName, setNewName] = useState('')
  const [newDest, setNewDest] = useState(destinationValue({ cat: 'manual', folderId: null }))
  const [newFolderName, setNewFolderName] = useState('')
  const [newUp, setNewUp] = useState('')
  const [newDown, setNewDown] = useState('')
  const [newShowDown, setNewShowDown] = useState(false)
  const [newKeepMissing, setNewKeepMissing] = useState(false)
  const [newError, setNewError] = useState<string | null>(null)

  // Edit tab — loaded from the focused set whenever the focus changes
  const [editName, setEditName] = useState('')
  const [editUp, setEditUp] = useState('')
  const [editDown, setEditDown] = useState('')
  const [editShowDown, setEditShowDown] = useState(false)
  // Existing members were put there on purpose, and a set is not tied to
  // one dataset — so an edit keeps genes this dataset lacks unless told not to.
  const [editKeepMissing, setEditKeepMissing] = useState(true)
  const [editLoadedFor, setEditLoadedFor] = useState<GeneSet | null>(null)
  // Whether the user has typed since the form was loaded. Distinct from
  // "differs from the set": once the dataset's genes arrive, recasing alone
  // can make the form differ, and that must not read as unsaved work.
  const [editTouched, setEditTouched] = useState(false)

  // Merge tab
  const [ruleKind, setRuleKind] = useState<MergeRule['kind']>('union')
  const [ruleK, setRuleK] = useState(2)
  const [mergeName, setMergeName] = useState('')
  const [mergeNameEdited, setMergeNameEdited] = useState(false)
  const [mergeDest, setMergeDest] = useState(destinationValue({ cat: 'manual', folderId: null }))
  const [mergeFolderName, setMergeFolderName] = useState('')
  const [mergeDeleteSources, setMergeDeleteSources] = useState(false)
  const [mergeError, setMergeError] = useState<string | null>(null)

  // Dataset genes for checking pasted symbols, fetched once per open.
  const [universe, setUniverse] = useState<string[] | null>(null)

  const loadEdit = (gs: GeneSet | null) => {
    setEditLoadedFor(gs)
    setEditName(gs?.name ?? '')
    setEditUp((gs?.genes ?? []).join('\n'))
    setEditDown((gs?.genesDown ?? []).join('\n'))
    setEditShowDown(!!gs?.genesDown?.length)
    setEditKeepMissing(true)
    setEditTouched(false)
  }

  // Per-open reset. This modal is mounted for the life of the app (it returns
  // null when closed, after its hooks), so nothing from the last visit may
  // leak into this one — a stale selection would make Delete take sets the
  // user can no longer see ticked.
  const [prevSource, setPrevSource] = useState<GeneSetManagerSource | null>(source)
  if (source !== prevSource) {
    setPrevSource(source)
    if (source) {
      setTab(source.tab)
      setQuery('')
      setSelection(EMPTY_SELECTION)
      setExpanded(new Set())
      setCollapsedCats(new Set())
      setStatus(null)
      setConfirmDelete(false)
      setCreatingFolder(false); setFolderDraft('')
      setRenaming(null); renameOpenRef.current = false
      setDragKeys(null); setDropTarget(null); setUndo(null)
      setNewName(''); setNewUp(''); setNewDown(''); setNewShowDown(false); setNewKeepMissing(false); setNewError(null)
      setNewFolderName('')
      setNewDest(validDestination(cats, source.dest ? destinationValue(source.dest) : destinationValue({ cat: 'manual', folderId: null })))
      setRuleKind('union'); setRuleK(2); setMergeName(''); setMergeNameEdited(false); setMergeDeleteSources(false); setMergeError(null)
      setMergeFolderName('')
      setMergeDest(destinationValue({ cat: 'manual', folderId: null }))
      setFocusKey(source.focus ?? null)
      loadEdit(source.focus ? findGeneSet(cats, source.focus) : null)
      setUniverse(null)
      if (source.focus && source.dest === undefined) {
        // Opened on a set: show where it lives.
        const parts = source.focus.split('|')
        if (parts[1]) setExpanded(new Set([`${parts[0]}|${parts[1]}`]))
      }
    }
  }

  const open = source !== null
  useEffect(() => {
    if (!open || !hasData) return
    let cancelled = false
    fetch(appendDataset('/api/genes'))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => { if (!cancelled && Array.isArray(d?.genes)) setUniverse(d.genes as string[]) })
      .catch(() => { /* unchecked: genes are kept as typed */ })
    return () => { cancelled = true }
    // currentVarIndex: swapping gene IDs changes what the dataset calls its genes.
  }, [open, hasData, activeSlot, currentVarIndex])

  const index: GeneIndex | null = useMemo(() => (universe ? makeGeneIndex(universe) : null), [universe])

  // Escape goes through the same guard as the backdrop and ×; a ref, so the
  // listener sees the current edit state without re-subscribing per keystroke.
  const requestCloseRef = useRef<() => void>(() => {})
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') requestCloseRef.current() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  useEffect(() => {
    if (!status) return
    const t = window.setTimeout(() => setStatus(null), 6000)
    return () => window.clearTimeout(t)
  }, [status])

  const tree = useMemo(() => filterTree(cats, MANAGER_ORDER, deferredQuery), [cats, deferredQuery])
  const totals = useMemo(() => countGeneSets(cats), [cats])
  const summary = useMemo(() => selectionSummary(cats, MANAGER_ORDER, selection), [cats, selection])
  const selectedSets = useMemo(() => gatherSelectedSets(cats, MANAGER_ORDER, selection), [cats, selection])
  const nTicked = summary.folders + summary.setsInFolders + summary.looseSets

  // New tab: parse + resolve as the user types
  const newParsed = useMemo(() => parseGeneText(newUp), [newUp])
  const newRes = useMemo(() => resolveGenes(newParsed.genes, index, newKeepMissing), [newParsed, index, newKeepMissing])
  const newDownParsed = useMemo(() => parseGeneText(newShowDown ? newDown : ''), [newDown, newShowDown])
  const newDownRes = useMemo(() => resolveGenes(newDownParsed.genes, index, newKeepMissing), [newDownParsed, index, newKeepMissing])

  // Edit tab
  const focused = focusKey ? findGeneSet(cats, focusKey) : null
  const editParsed = useMemo(() => parseGeneText(editUp), [editUp])
  const editRes = useMemo(() => resolveGenes(editParsed.genes, index, editKeepMissing), [editParsed, index, editKeepMissing])
  const editDownParsed = useMemo(() => parseGeneText(editShowDown ? editDown : ''), [editDown, editShowDown])
  const editDownRes = useMemo(() => resolveGenes(editDownParsed.genes, index, editKeepMissing), [editDownParsed, index, editKeepMissing])
  const editDownFinal = useMemo(() => {
    const up = new Set(editRes.genes)
    return editDownRes.genes.filter((g) => !up.has(g))
  }, [editRes, editDownRes])
  const editDirty = !!focused && (
    (editName.trim() || focused.name) !== focused.name
    || editRes.genes.join('\n') !== focused.genes.join('\n')
    || editDownFinal.join('\n') !== (focused.genesDown ?? []).join('\n')
  )
  // The set changed under the form (saved, or renamed in the panel) while
  // nothing here was typed: follow it rather than offering to overwrite it.
  if (focused && focused !== editLoadedFor && !editTouched) loadEdit(focused)

  // Merge tab
  const rule: MergeRule = ruleKind === 'atLeast' ? { kind: 'atLeast', k: ruleK } : { kind: ruleKind }
  const merged = useMemo(() => mergeGeneSets(selectedSets.map((s) => s.set), rule),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [selectedSets, ruleKind, ruleK])
  const suggested = selectedSets.length >= 2 ? suggestMergeName(selectedSets.map((s) => s.set.name), rule) : ''
  const shownMergeName = mergeNameEdited ? mergeName : suggested

  // Leaving with typed, unsaved edits asks first — the same question as
  // switching to another set. Closing is the likelier way to lose them.
  const requestClose = () => {
    if (editTouched && editDirty && !window.confirm(`Discard your unsaved edits to "${focused?.name}"?`)) return
    setSource(null)
  }
  requestCloseRef.current = requestClose

  if (!source) return null

  const close = requestClose

  // A remembered destination whose folder has since been deleted falls back
  // to Manual rather than failing at Create.
  const newDestValue = validDestination(cats, newDest)
  const mergeDestValue = validDestination(cats, mergeDest)

  const toggleExpanded = (key: string) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key); else next.add(key)
      return next
    })

  const focusSet = (cat: GeneSetCategoryType, folderId: string | null, gs: GeneSet) => {
    const key = setKey({ cat, folderId, setId: gs.id })
    if (editTouched && editDirty && key !== focusKey && !window.confirm(`Discard your unsaved edits to "${focused?.name}"?`)) return
    setFocusKey(key)
    loadEdit(gs)
    setTab('edit')
  }

  const createSet = () => {
    setNewError(null)
    const name = newName.trim()
    if (!name) { setNewError('Name the set.'); return }
    try {
      const prepared = prepareDestination(cats, newDestValue, newFolderName)
      const up = new Set(newRes.genes)
      const genesDown = newDownRes.genes.filter((g) => !up.has(g))
      const set: GeneSet = {
        id: generateGeneSetId(), name, genes: newRes.genes,
        ...(genesDown.length > 0 ? { genesDown } : {}),
      }
      replace(insertGeneSet(prepared.cats, prepared.dest, set))
      setStatus(`Created “${name}” (${plural(set.genes.length, 'gene')}) in ${prepared.label}`)
      setNewName(''); setNewUp(''); setNewDown('')
      if (newDestValue === NEW_FOLDER) { setNewDest(destinationValue(prepared.dest)); setNewFolderName('') }
    } catch (e) {
      setNewError(e instanceof Error ? e.message : String(e))
    }
  }

  const saveEdit = () => {
    if (!focused || !focusKey) return
    const name = editName.trim() || focused.name
    replace(updateGeneSet(cats, focusKey, { name, genes: editRes.genes, genesDown: editDownFinal }))
    setEditTouched(false)
    setStatus(`Saved “${name}”`)
  }

  const duplicateFocused = () => {
    if (!focused || !focusKey) return
    const [cat, folderId] = focusKey.split('|')
    const copy: GeneSet = {
      id: generateGeneSetId(), name: `${focused.name} (copy)`, genes: [...focused.genes],
      ...(focused.genesDown?.length ? { genesDown: [...focused.genesDown] } : {}),
    }
    const dest = { cat: cat as GeneSetCategoryType, folderId: folderId || null }
    replace(insertGeneSet(cats, dest, copy))
    const key = setKey({ ...dest, setId: copy.id })
    setFocusKey(key)
    loadEdit(copy)
    setStatus(`Duplicated as “${copy.name}”`)
  }

  const deleteTicked = () => {
    const parts: string[] = []
    if (summary.folders > 0) parts.push(plural(summary.folders, 'folder'))
    if (summary.setsInFolders + summary.looseSets > 0) parts.push(plural(summary.setsInFolders + summary.looseSets, 'gene set'))
    replace(deleteSelection(cats, selection))
    setSelection(EMPTY_SELECTION)
    setConfirmDelete(false)
    setStatus(`Deleted ${parts.join(' and ')}`)
  }

  // --- tree editing: new folder, rename, move, merge ---------------------------

  // A move or merge rewrites keys (a key names the set's folder): ticks and
  // the Edit focus follow the set, and the change can be undone.
  const applyTreeChange = (next: Categories, keyMap: ReadonlyMap<string, string>, label: string) => {
    setUndo({ label, before: cats, after: next, selection, focusKey })
    replace(next)
    setSelection(remapSelection(selection, keyMap, next))
    if (focusKey && keyMap.has(focusKey)) setFocusKey(keyMap.get(focusKey)!)
    setStatus(null)
  }

  const undoLast = () => {
    if (!undo || undo.after !== cats) return
    replace(undo.before)
    setSelection(undo.selection)
    setFocusKey(undo.focusKey)
    setUndo(null)
    setStatus('Undone')
  }

  const createFolder = () => {
    const name = folderDraft.trim()
    setCreatingFolder(false)
    setFolderDraft('')
    if (!name) return
    const folder: GeneSetFolder = { id: generateFolderId(), name, expanded: true, createdAt: new Date().toISOString(), geneSets: [] }
    replace(addFolder(cats, 'manual', folder))
    setExpanded((prev) => new Set(prev).add(folderKey({ cat: 'manual', folderId: folder.id })))
    setCollapsedCats((prev) => { const n = new Set(prev); n.delete('manual'); return n })
    setStatus(`Created folder “${name}” in Manual — drag sets onto it`)
  }

  const startRename = (rowId: string, current: string) => {
    renameOpenRef.current = true
    setRenaming(rowId)
    setRenameDraft(current)
  }

  const commitRename = () => {
    if (!renameOpenRef.current || !renaming) return
    renameOpenRef.current = false
    const target = renaming
    const name = renameDraft.trim()
    setRenaming(null)
    if (!name) return
    if (target.startsWith('f:')) {
      const [cat, folderId] = target.slice(2).split('|') as [GeneSetCategoryType, string]
      const f = cats[cat]?.folders.find((x) => x.id === folderId)
      if (!f || f.name === name) return
      replace(renameFolder(cats, cat, folderId, name))
      setStatus(`Renamed folder “${f.name}” to “${name}”`)
    } else {
      const key = target.slice(2)
      const gs = findGeneSet(cats, key)
      if (!gs || gs.name === name) return
      replace(updateGeneSet(cats, key, { name }))
      setStatus(`Renamed “${gs.name}” to “${name}”`)
    }
  }

  const cancelRename = () => { renameOpenRef.current = false; setRenaming(null) }

  const renameInput = () => (
    <input
      autoFocus
      value={renameDraft}
      onChange={(e) => setRenameDraft(e.target.value)}
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commitRename()
        // Stop here, or the manager's own Escape handler closes the window too.
        if (e.key === 'Escape') { e.stopPropagation(); cancelRename() }
      }}
      onBlur={commitRename}
      style={{ ...styles.input, padding: '1px 6px', fontSize: '12px', flex: 1, minWidth: 0 }}
    />
  )

  const clearExpandTimer = () => {
    if (expandTimerRef.current) { window.clearTimeout(expandTimerRef.current.id); expandTimerRef.current = null }
  }

  const endDrag = () => { setDragKeys(null); setDropTarget(null); clearExpandTimer() }

  const startDrag = (e: DragEvent, key: string, name: string) => {
    // A ticked set carries every ticked set with it; an unticked one goes alone.
    const keys = selection.sets.has(key) ? selectedSets.map((s) => s.key) : [key]
    e.dataTransfer.setData(MANAGER_DRAG_TYPE, JSON.stringify(keys))
    e.dataTransfer.setData('text/plain', keys.length > 1 ? `${keys.length} gene sets` : name)
    e.dataTransfer.effectAllowed = 'move'
    if (keys.length > 1) setDragBadge(e, `${keys.length} sets`)
    setDragKeys(keys)
  }

  // Drop-target handlers for one row. `accept` false leaves the drop refused.
  const dropProps = (rowId: string, accept: boolean, onDropKeys: (keys: string[]) => void, openFolderKey?: string) => ({
    onDragOver: (e: DragEvent) => {
      if (!dragKeys || !accept) return
      e.preventDefault()
      e.dataTransfer.dropEffect = 'move'
      if (dropTarget !== rowId) setDropTarget(rowId)
      // Hovering a closed folder opens it, so a set inside can be a merge target.
      if (openFolderKey && !expanded.has(openFolderKey) && expandTimerRef.current?.key !== openFolderKey) {
        clearExpandTimer()
        const id = window.setTimeout(() => {
          setExpanded((prev) => new Set(prev).add(openFolderKey))
          expandTimerRef.current = null
        }, 600)
        expandTimerRef.current = { key: openFolderKey, id }
      }
    },
    onDragLeave: (e: DragEvent) => {
      if (e.currentTarget.contains(e.relatedTarget as Node | null)) return
      if (dropTarget === rowId) setDropTarget(null)
      if (openFolderKey && expandTimerRef.current?.key === openFolderKey) clearExpandTimer()
    },
    onDrop: (e: DragEvent) => {
      if (!dragKeys || !accept) return
      e.preventDefault()
      const keys = dragKeys
      endDrag()
      onDropKeys(keys)
    },
  })

  const moveTo = (keys: string[], dest: Destination, label: string) => {
    try {
      const { cats: next, keyMap } = moveSets(cats, keys, dest)
      if (keyMap.size === 0) { setStatus(`Already in ${label}`); return }
      applyTreeChange(next, keyMap, `Moved ${keyMap.size === 1 ? `“${findGeneSet(next, [...keyMap.values()][0])?.name}”` : plural(keyMap.size, 'set')} to ${label}`)
      if (dest.folderId) setExpanded((prev) => new Set(prev).add(folderKey({ cat: dest.cat, folderId: dest.folderId! })))
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err))
    }
  }

  // Where a set lives, for messages: sets often share a name (every pathway
  // folder has its "ligands"), so the name alone does not say which.
  const whereIs = (key: string) => {
    const [cat, folderId] = key.split('|') as [GeneSetCategoryType, string]
    return (folderId && cats[cat]?.folders.find((f) => f.id === folderId)?.name) || cats[cat]?.name || ''
  }

  const mergeInto = (keys: string[], targetKey: string) => {
    const sourceKeys = keys.filter((k) => k !== targetKey && findGeneSet(cats, k))
    if (sourceKeys.length === 0) return
    try {
      const { cats: next, merged } = mergeSetsInto(cats, keys, targetKey)
      const first = findGeneSet(cats, sourceKeys[0])!
      const what = sourceKeys.length === 1 ? `“${first.name}” (${whereIs(sourceKeys[0])})` : plural(sourceKeys.length, 'set')
      applyTreeChange(next, new Map(),
        `Merged ${what} into “${merged.name}” (${whereIs(targetKey)}) — ${plural(merged.genes.length, 'gene')}`)
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err))
    }
  }

  const dropHint = (kind: 'move' | 'merge') => (
    <span style={{ ...styles.count, color: kind === 'move' ? '#4ecdc4' : '#e9a23b', fontWeight: 600 }}>
      {kind === 'move' ? '⤵ move here' : '⊕ merge into'}
    </span>
  )

  const targetStyle = (rowId: string, kind: 'move' | 'merge'): CSSProperties => (dropTarget === rowId
    ? kind === 'move'
      ? { outline: '1px dashed #4ecdc4', backgroundColor: 'rgba(78,205,196,0.12)' }
      : { outline: '1px dashed #e9a23b', backgroundColor: 'rgba(233,162,59,0.16)' }
    : {})

  const editButton = (rowId: string, current: string) => (hoverRow === rowId && renaming !== rowId && !dragKeys ? (
    <button
      onClick={(e) => { e.stopPropagation(); startRename(rowId, current) }}
      style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: '11px', padding: '0 2px', flex: '0 0 auto' }}
      title="Rename (or double-click the name)"
    >
      ✎
    </button>
  ) : null)

  const mergeDestKey = mergeDestValue === NEW_FOLDER ? null : mergeDestValue
  const mergeDestDoomed = mergeDeleteSources && !!mergeDestKey && (() => {
    const d = destinationFromValue(mergeDestKey)
    return !!d?.folderId && selection.folders.has(folderKey({ cat: d.cat, folderId: d.folderId }))
  })()

  const doMerge = () => {
    setMergeError(null)
    const name = shownMergeName.trim()
    if (!name) { setMergeError('Name the merged set.'); return }
    if (mergeDestDoomed) { setMergeError('The destination folder is among those being deleted — pick another.'); return }
    try {
      const base = mergeDeleteSources ? deleteSelection(cats, selection) : cats
      const prepared = prepareDestination(base, mergeDestValue, mergeFolderName)
      const set: GeneSet = { id: generateGeneSetId(), name, genes: merged.genes, ...(merged.genesDown ? { genesDown: merged.genesDown } : {}) }
      replace(insertGeneSet(prepared.cats, prepared.dest, set))
      const n = selectedSets.length
      setStatus(`Merged ${plural(n, 'set')} into “${name}” (${plural(set.genes.length, 'gene')}) in ${prepared.label}`
        + (mergeDeleteSources ? ' — sources deleted' : ''))
      if (mergeDeleteSources) setSelection(EMPTY_SELECTION)
      setMergeNameEdited(false)
      if (mergeDestValue === NEW_FOLDER) { setMergeDest(destinationValue(prepared.dest)); setMergeFolderName('') }
    } catch (e) {
      setMergeError(e instanceof Error ? e.message : String(e))
    }
  }

  // --- tree -------------------------------------------------------------------

  const renderSetRow = (cat: GeneSetCategoryType, folder: GeneSetFolder | null, gs: GeneSet, indent: number) => {
    const key = setKey({ cat, folderId: folder?.id ?? null, setId: gs.id })
    const rowId = `s:${key}`
    const isFocused = key === focusKey && tab === 'edit'
    const isDragged = !!dragKeys?.includes(key)
    return (
      <div
        key={key}
        draggable={renaming !== rowId}
        onDragStart={(e) => startDrag(e, key, gs.name)}
        onDragEnd={endDrag}
        onMouseEnter={() => setHoverRow(rowId)}
        onMouseLeave={() => setHoverRow((h) => (h === rowId ? null : h))}
        {...dropProps(rowId, !isDragged, (keys) => mergeInto(keys, key))}
        style={{
          ...styles.row, paddingLeft: `${indent}px`, backgroundColor: isFocused ? '#0f3460' : undefined,
          opacity: isDragged ? 0.45 : 1, cursor: 'grab', ...targetStyle(rowId, 'merge'),
        }}
        title="Drag onto a folder to move it, onto another set to merge (union) into that set"
      >
        <TriCheckbox state={selection.sets.has(key) ? 'all' : 'none'} onChange={() => setSelection((s) => toggleSet(s, cat, folder, gs.id))} />
        {renaming === rowId ? renameInput() : (
          <>
            <span
              onClick={() => focusSet(cat, folder?.id ?? null, gs)}
              onDoubleClick={(e) => { e.stopPropagation(); startRename(rowId, gs.name) }}
              style={{ cursor: 'pointer', color: isFocused ? '#4ecdc4' : '#ddd', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
              title="Click to edit its genes, double-click to rename"
            >
              {gs.name}
            </span>
            {editButton(rowId, gs.name)}
          </>
        )}
        {dropTarget === rowId ? dropHint('merge') : <span style={styles.count}>{geneCountLabel(gs)}</span>}
      </div>
    )
  }

  const searching = deferredQuery.trim().length > 0

  const renderTree = () => {
    if (tree.length === 0) {
      return <div style={styles.hint}>{searching ? `Nothing matches “${deferredQuery}”.` : 'No gene sets yet — create one on the right.'}</div>
    }
    return tree.map((c) => {
      const collapsed = collapsedCats.has(c.cat) && !searching
      const nSets = c.category.geneSets.length + c.category.folders.reduce((s, f) => s + f.geneSets.length, 0)
      return (
        <div key={c.cat} style={{ marginBottom: '6px' }}>
          <div
            {...dropProps(`c:${c.cat}`, true, (keys) => moveTo(keys, { cat: c.cat, folderId: null }, `${c.category.name} (top level)`))}
            style={{ ...styles.row, backgroundColor: '#0f1625', cursor: 'pointer', fontWeight: 600, ...targetStyle(`c:${c.cat}`, 'move') }}
            onClick={() => setCollapsedCats((prev) => {
              const next = new Set(prev)
              if (next.has(c.cat)) next.delete(c.cat); else next.add(c.cat)
              return next
            })}
          >
            <TriCheckbox
              state={shownState(selection, [c])}
              onChange={() => setSelection((s) => toggleShown(s, [c]))}
              title={searching ? `Tick every match in ${c.category.name}` : `Tick everything in ${c.category.name}`}
            />
            <span style={{ color: '#888', fontSize: '9px', width: '10px' }}>{collapsed ? '▶' : '▼'}</span>
            <span>{c.category.name}</span>
            {dropTarget === `c:${c.cat}` ? dropHint('move') : (
              <span style={styles.count}>
                {c.category.folders.length > 0 && `${plural(c.category.folders.length, 'folder')} · `}{plural(nSets, 'set')}
              </span>
            )}
          </div>
          {!collapsed && (
            <>
              {c.folders.map((f) => {
                const isOpen = searching || expanded.has(f.key)
                const view: FilteredCategory[] = [{ ...c, folders: [f], sets: [] }]
                const rowId = `f:${f.key}`
                return (
                  <div key={f.key}>
                    <div
                      {...dropProps(rowId, true, (keys) => moveTo(keys, { cat: c.cat, folderId: f.folder.id }, `${c.category.name} › ${f.folder.name}`), f.key)}
                      onMouseEnter={() => setHoverRow(rowId)}
                      onMouseLeave={() => setHoverRow((h) => (h === rowId ? null : h))}
                      style={{ ...styles.row, paddingLeft: '20px', cursor: 'pointer', ...targetStyle(rowId, 'move') }}
                      onClick={() => { if (renaming !== rowId) toggleExpanded(f.key) }}
                    >
                      <TriCheckbox
                        state={shownState(selection, view)}
                        onChange={() => setSelection((s) => toggleShown(s, view))}
                        title={f.whole ? 'Tick this folder and everything in it' : `Tick the ${f.sets.length} matching sets`}
                      />
                      <span style={{ color: '#888', fontSize: '9px', width: '10px' }}>{isOpen ? '▼' : '▶'}</span>
                      {renaming === rowId ? (<><span>📁</span>{renameInput()}</>) : (
                        <>
                          <span
                            onDoubleClick={(e) => { e.stopPropagation(); startRename(rowId, f.folder.name) }}
                            style={{ color: '#ccc', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                            title="Double-click to rename"
                          >
                            📁 {f.folder.name}
                          </span>
                          {editButton(rowId, f.folder.name)}
                        </>
                      )}
                      {dropTarget === rowId ? dropHint('move') : (
                        <span style={styles.count}>
                          {searching && !f.whole ? `${f.sets.length} of ${f.folder.geneSets.length}` : plural(f.folder.geneSets.length, 'set')}
                        </span>
                      )}
                    </div>
                    {isOpen && f.sets.map((gs) => renderSetRow(c.cat, f.folder, gs, 44))}
                    {isOpen && f.folder.geneSets.length === 0 && (
                      <div style={{ ...styles.row, paddingLeft: '44px', color: '#666', fontStyle: 'italic' }}>empty</div>
                    )}
                  </div>
                )
              })}
              {c.sets.map((gs) => renderSetRow(c.cat, null, gs, 20))}
            </>
          )}
        </div>
      )
    })
  }

  // --- right-hand tabs -------------------------------------------------------

  const renderNew = () => {
    const nGenes = newRes.genes.length
    const canCreate = newName.trim().length > 0 && (newDestValue !== NEW_FOLDER || newFolderName.trim().length > 0)
    return (
      <>
        <label style={{ ...styles.label, marginTop: 0 }}>Name</label>
        <input
          autoFocus
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && canCreate) createSet() }}
          placeholder="e.g. Chondrocyte markers"
          style={styles.input}
        />
        <label style={styles.label}>Put it in</label>
        <DestinationSelect cats={cats} value={newDestValue} onChange={setNewDest} newFolderName={newFolderName} onNewFolderName={setNewFolderName} />
        <label style={styles.label}>Genes {newShowDown && <span style={{ color: '#666' }}>(up)</span>}</label>
        <textarea
          value={newUp}
          onChange={(e) => setNewUp(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && canCreate) createSet() }}
          placeholder={'Paste or type gene symbols — separated by spaces, commas, tabs or new lines.\n\nSox9 Col2a1 Acan\nMeis1, Meis2'}
          rows={newShowDown ? 6 : 10}
          style={styles.textarea}
        />
        <ResolutionSummary
          res={newRes} duplicates={newParsed.duplicates}
          keepMissing={newKeepMissing} onKeepMissing={setNewKeepMissing}
          keepLabel="Keep them anyway"
        />
        {newShowDown ? (
          <>
            <label style={styles.label}>Down genes <span style={{ color: '#666' }}>(scored against — UCell)</span></label>
            <textarea value={newDown} onChange={(e) => setNewDown(e.target.value)} rows={4} style={styles.textarea} />
            {newDownParsed.genes.length > 0 && (
              <ResolutionSummary
                res={newDownRes} duplicates={newDownParsed.duplicates}
                keepMissing={newKeepMissing} onKeepMissing={setNewKeepMissing}
                keepLabel="Keep them anyway"
              />
            )}
          </>
        ) : (
          <button onClick={() => setNewShowDown(true)} style={{ ...styles.smallButton, marginTop: '8px', background: 'none', color: '#888' }}>
            + Down genes (directional set)
          </button>
        )}
        {newError && <div style={{ color: '#e94560', fontSize: '12px', marginTop: '8px' }}>{newError}</div>}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '14px' }}>
          <button onClick={createSet} disabled={!canCreate} style={canCreate ? styles.primaryButton : styles.disabledButton}>
            Create set
          </button>
          <span style={{ fontSize: '11px', color: '#888' }}>
            {nGenes === 0
              ? 'Empty for now — genes can be dragged onto it in the Genes pane'
              : plural(nGenes, 'gene')}
            {canCreate && <span style={{ color: '#555' }}> · ⌘↵</span>}
          </span>
        </div>
      </>
    )
  }

  const renderEdit = () => {
    if (!focused || !focusKey) {
      return <div style={styles.hint}>Click a set’s name on the left to edit it.</div>
    }
    const [cat, folderId] = focusKey.split('|')
    const c = cats[cat as GeneSetCategoryType]
    const f = folderId ? c.folders.find((x) => x.id === folderId) : null
    return (
      <>
        <div style={{ fontSize: '11px', color: '#888' }}>{c.name}{f ? ` › ${f.name}` : ''}</div>
        <label style={styles.label}>Name</label>
        <input value={editName} onChange={(e) => { setEditName(e.target.value); setEditTouched(true) }} style={styles.input} />
        <label style={styles.label}>Genes {editShowDown && <span style={{ color: '#666' }}>(up)</span>} — one per line, or paste more</label>
        <textarea
          value={editUp}
          onChange={(e) => { setEditUp(e.target.value); setEditTouched(true) }}
          onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && editDirty) saveEdit() }}
          rows={editShowDown ? 7 : 12}
          style={styles.textarea}
        />
        <ResolutionSummary
          res={editRes} duplicates={editParsed.duplicates}
          keepMissing={editKeepMissing} onKeepMissing={setEditKeepMissing}
          keepLabel="Keep genes this dataset lacks (sets are not tied to one dataset)"
        />
        {editShowDown ? (
          <>
            <label style={styles.label}>Down genes</label>
            <textarea value={editDown} onChange={(e) => { setEditDown(e.target.value); setEditTouched(true) }} rows={4} style={styles.textarea} />
          </>
        ) : (
          <button onClick={() => setEditShowDown(true)} style={{ ...styles.smallButton, marginTop: '8px', background: 'none', color: '#888' }}>
            + Down genes (directional set)
          </button>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '14px' }}>
          <button onClick={saveEdit} disabled={!editDirty} style={editDirty ? styles.primaryButton : styles.disabledButton}>Save</button>
          <button onClick={() => loadEdit(focused)} disabled={!editDirty} style={{ ...styles.smallButton, opacity: editDirty ? 1 : 0.5 }}>Revert</button>
          <button onClick={duplicateFocused} style={styles.smallButton}>Duplicate</button>
          <span style={{ fontSize: '11px', color: '#888', marginLeft: 'auto' }}>
            {plural(editRes.genes.length, 'gene')}{editDirty ? ' · unsaved' : ''}
          </span>
        </div>
      </>
    )
  }

  const renderMerge = () => {
    if (selectedSets.length < 2) {
      return (
        <div style={styles.hint}>
          Tick two or more sets on the left — or a folder, to merge everything in it — and they are combined here into one new set.
          {selectedSets.length === 1 && <div style={{ marginTop: '8px', color: '#aaa' }}>1 set ticked so far.</div>}
        </div>
      )
    }
    const n = selectedSets.length
    const canMerge = shownMergeName.trim().length > 0 && !mergeDestDoomed && (mergeDestValue !== NEW_FOLDER || mergeFolderName.trim().length > 0)
    return (
      <>
        <div style={{ fontSize: '11px', color: '#888', marginBottom: '4px' }}>{plural(n, 'set')} ticked</div>
        <div style={{ maxHeight: '110px', overflowY: 'auto', backgroundColor: '#0f1625', borderRadius: '4px', padding: '4px 0' }}>
          {selectedSets.map((s) => (
            <div key={s.key} style={{ ...styles.row, minHeight: '18px', padding: '1px 8px' }}>
              <span style={{ color: '#ddd', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{s.set.name}</span>
              <span style={{ color: '#666', fontSize: '10px', whiteSpace: 'nowrap' }}>{s.path}</span>
              <span style={styles.count}>{geneCountLabel(s.set)}</span>
            </div>
          ))}
        </div>
        <label style={styles.label}>Keep genes that are</label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px' }}>
          {([
            ['union', 'in any of them (union)'],
            ['intersection', 'in every one (intersection)'],
            ['atLeast', null],
          ] as const).map(([kind, text]) => (
            <label key={kind} style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
              <input type="radio" name="mergeRule" checked={ruleKind === kind} onChange={() => setRuleKind(kind)} style={{ margin: 0 }} />
              {text ?? (
                <>
                  in at least
                  <input
                    type="number" min={1} max={n} value={Math.min(ruleK, n)}
                    onChange={(e) => { setRuleK(Math.max(1, Number(e.target.value) || 1)); setRuleKind('atLeast') }}
                    // Clicking into the count means this rule, even before the number changes.
                    onFocus={() => setRuleKind('atLeast')}
                    style={{ ...styles.input, width: '52px', padding: '2px 4px' }}
                  />
                  of {n}
                </>
              )}
            </label>
          ))}
        </div>
        <div style={{ marginTop: '10px', fontSize: '12px' }}>
          Result: <span style={{ color: merged.genes.length ? '#4ecdc4' : '#e94560' }}>{plural(merged.genes.length, 'gene')}</span>
          {merged.genesDown && <span style={{ color: '#888' }}> · ↓{merged.genesDown.length} down</span>}
        </div>
        {merged.genes.length > 0 && (
          <div style={{ fontSize: '11px', color: '#888', marginTop: '2px', lineHeight: '15px', maxHeight: '45px', overflow: 'hidden' }}>
            {merged.genes.slice(0, 30).join(', ')}{merged.genes.length > 30 ? ', …' : ''}
          </div>
        )}
        <label style={styles.label}>Name</label>
        <input
          value={shownMergeName}
          onChange={(e) => { setMergeName(e.target.value); setMergeNameEdited(true) }}
          style={styles.input}
        />
        <label style={styles.label}>Put it in</label>
        <DestinationSelect cats={cats} value={mergeDestValue} onChange={setMergeDest} newFolderName={mergeFolderName} onNewFolderName={setMergeFolderName} />
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', marginTop: '10px', cursor: 'pointer' }}>
          <input type="checkbox" checked={mergeDeleteSources} onChange={(e) => setMergeDeleteSources(e.target.checked)} style={{ margin: 0 }} />
          Then delete what is ticked
          <span style={{ color: '#888' }}>
            ({[summary.folders > 0 ? plural(summary.folders, 'folder') : null, plural(summary.setsInFolders + summary.looseSets, 'set')].filter(Boolean).join(', ')})
          </span>
        </label>
        {mergeDestDoomed && <div style={{ color: '#e9a23b', fontSize: '11px', marginTop: '4px' }}>The destination folder would be deleted too — pick another.</div>}
        {mergeError && <div style={{ color: '#e94560', fontSize: '12px', marginTop: '8px' }}>{mergeError}</div>}
        <div style={{ marginTop: '14px' }}>
          <button onClick={doMerge} disabled={!canMerge} style={canMerge ? styles.primaryButton : styles.disabledButton}>
            Merge {n} sets
          </button>
        </div>
      </>
    )
  }

  const tabButton = (key: GeneSetManagerTab, label: string) => (
    <button
      key={key}
      onClick={() => setTab(key)}
      style={{
        flex: 1, padding: '8px', fontSize: '12px', cursor: 'pointer', background: 'none',
        color: tab === key ? '#4ecdc4' : '#888', border: 'none',
        borderBottom: `2px solid ${tab === key ? '#4ecdc4' : 'transparent'}`,
      }}
    >
      {label}
    </button>
  )

  const deleteNames = [
    ...MANAGER_ORDER.flatMap((cat) => cats[cat].folders
      .filter((f) => selection.folders.has(folderKey({ cat, folderId: f.id })))
      .map((f) => `📁 ${f.name}`)),
    ...selectedSets
      .filter((s) => !(s.folderId && selection.folders.has(folderKey({ cat: s.cat, folderId: s.folderId }))))
      .map((s) => s.set.name),
  ]

  return (
    <>
    <div style={styles.overlay} onClick={close}>
      <div style={styles.card} onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Gene set manager">
        <div style={styles.header}>
          <div style={{ fontSize: '15px', fontWeight: 600 }}>Gene sets</div>
          <div style={{ fontSize: '11px', color: '#888' }}>
            {plural(totals.sets, 'set')}{totals.folders > 0 ? ` in ${plural(totals.folders, 'folder')}` : ''}
          </div>
          <button
            onClick={close}
            style={{ marginLeft: 'auto', background: 'none', border: 'none', color: '#888', fontSize: '18px', cursor: 'pointer', lineHeight: 1 }}
            title="Close (Esc)"
            aria-label="Close gene set manager"
          >
            ×
          </button>
        </div>

        <div style={styles.body}>
          <div style={styles.left}>
            <div style={{ padding: '10px 12px 6px' }}>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter by set, folder or gene…"
                style={styles.input}
              />
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '8px', flexWrap: 'wrap' }}>
                <button
                  style={{ ...styles.smallButton, color: '#4ecdc4' }}
                  onClick={() => { setCreatingFolder(true); setFolderDraft('') }}
                  title="Make a folder in Manual, then drag sets onto it"
                >
                  + New folder
                </button>
                <button style={styles.smallButton} onClick={() => setSelection((s) => selectAllShown(s, tree))}>
                  {searching ? 'Tick all matches' : 'Tick all'}
                </button>
                <button style={{ ...styles.smallButton, opacity: nTicked ? 1 : 0.5 }} disabled={!nTicked} onClick={() => setSelection(EMPTY_SELECTION)}>
                  Untick
                </button>
                <span style={{ fontSize: '11px', color: '#888' }}>
                  {nTicked === 0 ? 'nothing ticked'
                    : [summary.folders > 0 ? plural(summary.folders, 'folder') : null,
                      summary.setsInFolders + summary.looseSets > 0 ? plural(summary.setsInFolders + summary.looseSets, 'set') : null,
                    ].filter(Boolean).join(', ') + ' ticked'}
                </span>
                <span style={{ marginLeft: 'auto', display: 'flex', gap: '6px' }}>
                  <button
                    style={{ ...styles.smallButton, color: nTicked ? '#e94560' : '#666', borderColor: nTicked ? '#5a2a3a' : '#1a1a2e', cursor: nTicked ? 'pointer' : 'default' }}
                    disabled={!nTicked}
                    onClick={() => setConfirmDelete(true)}
                    title="Delete everything ticked, in one step"
                  >
                    Delete ticked
                  </button>
                  <button
                    style={{ ...styles.smallButton, color: '#888' }}
                    onClick={() => setClearGeneSetsOpen(true)}
                    title="Delete every gene set and folder"
                  >
                    Clear all…
                  </button>
                </span>
              </div>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '4px 8px 8px' }}>
              {creatingFolder && (
                <div style={{ ...styles.row, marginBottom: '4px', backgroundColor: '#0f1625' }}>
                  <span>📁</span>
                  <input
                    autoFocus
                    value={folderDraft}
                    onChange={(e) => setFolderDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') createFolder()
                      if (e.key === 'Escape') { e.stopPropagation(); setCreatingFolder(false); setFolderDraft('') }
                    }}
                    onBlur={createFolder}
                    placeholder="New folder name (in Manual) — Enter to create"
                    style={{ ...styles.input, padding: '2px 6px', flex: 1 }}
                  />
                </div>
              )}
              {renderTree()}
            </div>
            <div style={{ minHeight: '26px', padding: '4px 12px', borderTop: '1px solid #0f3460', fontSize: '11px', color: '#4ecdc4', display: 'flex', alignItems: 'center', gap: '6px' }}>
              {undo && undo.after === cats ? (
                <>
                  <span>{undo.label}</span>
                  <button onClick={undoLast} style={{ ...styles.smallButton, padding: '1px 8px', color: '#e9a23b' }}>Undo</button>
                </>
              ) : status ?? (
                <span style={{ color: '#666' }}>Drag a set onto a folder to move it, or onto another set to merge them.</span>
              )}
            </div>
          </div>

          <div style={styles.right}>
            <div style={{ display: 'flex', borderBottom: '1px solid #0f3460' }}>
              {tabButton('new', 'New set')}
              {tabButton('edit', focused ? `Edit “${focused.name.length > 18 ? focused.name.slice(0, 17) + '…' : focused.name}”` : 'Edit')}
              {tabButton('merge', selectedSets.length >= 2 ? `Merge (${selectedSets.length})` : 'Merge')}
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px' }}>
              {tab === 'new' && renderNew()}
              {tab === 'edit' && renderEdit()}
              {tab === 'merge' && renderMerge()}
            </div>
          </div>
        </div>
      </div>
    </div>

      {/* A sibling, not a child: React bubbles clicks through the component
          tree, so a click on this dialog's backdrop would otherwise also reach
          the manager's overlay and close it. */}
      <ConfirmDialog
        open={confirmDelete}
        title={`Delete ${[summary.folders > 0 ? plural(summary.folders, 'folder') : null,
          summary.setsInFolders + summary.looseSets > 0 ? plural(summary.setsInFolders + summary.looseSets, 'gene set') : null,
        ].filter(Boolean).join(' and ')}?`}
        confirmLabel="Delete"
        onConfirm={deleteTicked}
        onCancel={() => setConfirmDelete(false)}
      >
        <div style={{ maxHeight: '150px', overflowY: 'auto', color: '#aaa' }}>
          {deleteNames.slice(0, 12).map((n, i) => <div key={i}>{n}</div>)}
          {deleteNames.length > 12 && <div>… and {deleteNames.length - 12} more</div>}
        </div>
        <div style={{ marginTop: '8px' }}>This cannot be undone.</div>
      </ConfirmDialog>
    </>
  )
}
