# Official gene symbols for datasets that only have Ensembl IDs

## The problem

Some AnnData files carry nothing but Ensembl gene IDs in `.var` —
`ENSMUSG00000002603`, not `Tgfb1`. Every part of xcell that names a gene then
becomes unusable at once: gene search returns nothing for `Shh`, imported gene
sets match zero genes, marker panels are unreadable, and the Localize gene basis
cannot be curated. The data is fine; the labels are unusable by a human.

The fix a user reaches for today is to leave xcell, map the IDs in a script, and
re-save the file. That is a round trip through another tool for a lookup xcell
could do instantly.

## What already exists, and is therefore not being rebuilt

`.var` already supports alternative identifier columns end to end:

- `get_var_identifier_columns()` (`adaptor.py:1530`) lists `.var` columns that
  look like gene identifiers — string dtype, >90% unique.
- `swap_var_index(column)` (`adaptor.py:1567`) promotes one to the gene index,
  moving the old index to a column, calling `var_names_make_unique()`, clearing
  the expression and UCell caches and regenerating the gene mask.
- The **Gene IDs:** dropdown in the Genes panel (`GenePanel.tsx:1897`) drives it.

So this feature is not "a way to rename genes". It is **one missing piece: a
good `.var['gene_symbol']` column**, after which the existing machinery already
does the rest. Building a parallel renaming path would duplicate
`swap_var_index`'s cache invalidation, which is the part that is easy to get
wrong.

## Approach

### 1. Shipped lookup tables, not a network call

Two gzipped TSVs in `xcell/data/`, `ensembl_symbols_human.tsv.gz` and
`ensembl_symbols_mouse.tsv.gz`, each `ensembl_id<TAB>symbol` sorted by ID.

Shipped rather than fetched, for three reasons: it works on a laptop with no
network, it is instant rather than a multi-second API round trip per session,
and it is *deterministic* — the same dataset maps the same way today and in a
year, which a live API cannot promise. `xcell/data/build_gene_symbol_maps.py`
regenerates them and is committed beside the data, matching the existing
`build_lr_db.py` / `lr_pairs.csv` pair.

**Two GENCODE releases per species, newest winning.** A single current release
misses retired IDs, and real datasets are usually older than the current
annotation — 10x's `refdata-gex-GRCh38-2020-A`, still one of the most widely
used references anywhere, is GENCODE v32/M23. The tables merge v32 with v50
(human) and M23 with M39 (mouse), later releases overwriting earlier ones so a
renamed gene gets its current symbol.

### 2. Species is detected, not asked

Ensembl IDs carry their species in the prefix: `ENSG` is human, `ENSMUSG` is
mouse. The user is not asked a question the data already answers. Detection is
by majority over the IDs present, and the result is reported rather than
assumed silently, because a mixed-species dataset is a real thing and the user
should see which way it went.

A dataset whose IDs are neither — `ENSDARG` (zebrafish), `ENSRNOG` (rat) — is
**refused with the species named**, not mapped badly. Guessing there would
produce a column of mostly-unmapped IDs that looks like it worked.

### 3. Version suffixes are stripped

`ENSG00000141510.15` and `ENSG00000141510` are the same gene. The tables are
keyed without the version and lookups strip it, because whether a dataset keeps
the suffix depends on which tool wrote it and is not a property of the biology.

### 4. The cost is shown before the action, not after

`POST /api/genes/symbol_mapping_preview` reports, without mutating anything:
detected species, how many IDs map, how many do not, how many symbols would
collide, and a handful of worked examples. This is the same contract as the
gene-overlap panel in Localize and the confidence distribution in its results —
a user should see what an operation will cost while the decision is still
theirs.

The preview is also how the feature says **"you do not need this"**: if `.var`
already carries a symbol column, or the index is already symbols rather than
Ensembl IDs, the preview says so and names the existing column, because the
Gene IDs picker already solves that case.

### 5. Unmapped IDs keep their ID

A gene that is not in the tables keeps its Ensembl ID as its "symbol". Never
blank, never dropped, never `NaN`. A blank would break the identifier-column
detection (`>90% unique`) and silently make some genes unaddressable; dropping
genes to make labels tidy would be a data loss for a cosmetic gain.

### 6. Duplicate symbols are expected and tolerated

Several Ensembl IDs legitimately map to one symbol. In a `.var` **column** that
is harmless, and if the user promotes it to the index `swap_var_index` already
calls `var_names_make_unique()`. The preview reports the count so a user who
cares can see it; the code does not refuse over it. This is deliberately unlike
`rename_genes`, which refuses on collision — there the collision is a *symptom
of a bad pattern the user typed*, while here it is a fact about the annotation
and there is no pattern to fix.

### 7. Optionally promote to the gene index

`map_gene_symbols(set_as_index=False)` writes only the column. The route accepts
`set_as_index`, and when true simply calls the existing `swap_var_index` after
writing, so there is exactly one implementation of that operation. The UI
checkbox defaults to **on**, because a user who asks for gene symbols wants
genes named by them; the API default is **off**, because a library call should
not reshape the dataset unless asked.

## Where it lives

Genes panel → the existing `⋯` OverflowMenu → **"Add gene symbols…"**, beside
the gene-mask item. A small modal: what was detected, what will map, the
checkbox, and the result summary afterwards.

## Testing

`backend/tests/test_gene_symbols.py`, self-contained per the house pattern.

1. `ENSG` detects human, `ENSMUSG` detects mouse, mixed reports the majority.
2. A versioned ID maps identically to an unversioned one.
3. An unmapped ID keeps its own ID rather than becoming blank or `NaN`.
4. Duplicate symbols are reported and do not raise.
5. A non-human/mouse prefix is refused with the species named.
6. An `.obs`-style already-symbols dataset is reported as not needing this.
7. `map_gene_symbols` writes `.var['gene_symbol']` and leaves the index alone;
   with `set_as_index` the index becomes symbols and stays unique.
8. The preview mutates nothing — `.var` is byte-identical afterwards.
9. Preview output is JSON-serializable: no `NaN`, no `inf`.

The lookup module is pure and takes lists of strings, so its tests never touch
AnnData or the shipped tables.

## Out of scope

Reverse mapping (symbol → Ensembl). Cross-species ortholog mapping, which is a
different problem already partly served by the case-convention work. Species
beyond human and mouse — the table format is per-species so adding one later is
a data change, not a code change. Live queries to mygene/BioMart. Entrez or
RefSeq identifiers.
