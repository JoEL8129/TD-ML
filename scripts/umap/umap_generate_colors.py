# Callback args: me (this DAT), par (changed Par), val (current value), prev (previous value).
# Parameter Execute DAT must have the matching toggle enabled.

def onValueChange(par, prev):
	# par.eval() for current value
	return

# changes: list of (Par, previous value) named tuples
def onValuesChanged(changes):
	for c in changes:
		# par.eval() for current value
		par = c.par
		prev = c.prev
	return

UMAP_COLOR_PALETTE = [
	(1.0, 0.2, 0.2), (0.2, 0.6, 1.0), (0.2, 0.8, 0.4), (1.0, 0.6, 0.0),
	(0.6, 0.2, 0.8), (0.0, 0.8, 0.8), (0.9, 0.5, 0.7), (0.5, 0.5, 0.5),
	(0.95, 0.7, 0.3), (0.4, 0.9, 0.6), (0.7, 0.4, 0.9), (0.9, 0.9, 0.3),
	(0.3, 0.5, 0.9), (0.9, 0.4, 0.4), (0.4, 0.8, 0.9), (0.8, 0.6, 0.2),
]

def onPulse(par):
	# Labels from DAT (path: labelsDAT)
	label_dat = op('labelsDAT')
	rows = label_dat.rows()
	labels = [row[0].val for row in rows]

	# Unique labels → fixed color by class index (palette cycles if more than 16)
	unique_labels = sorted(set(labels))
	label_color_map = {}
	for i, lbl in enumerate(unique_labels):
		r, g, b = UMAP_COLOR_PALETTE[i % len(UMAP_COLOR_PALETTE)]
		label_color_map[lbl] = [r, g, b]

	# Write label→color mapping to legend table
	map_dat = op('labelColorDAT')
	map_dat.clear()
	map_dat.appendRow(['label','r','g','b'])
	for lbl in unique_labels:
		r, g, b = label_color_map[lbl]
		map_dat.appendRow([lbl, r, g, b])

	# Merge colors into UMAP 2D output: [x,y] → [r,g,b] per row
	umap_dat = op('umap2D_DAT')
	out_dat  = op('colored2D_DAT')
	out_dat.clear()
	out_dat.appendRow(['r','g','b'])

	for i, row in enumerate(umap_dat.rows()[1:]):  # skip header
		lbl = labels[i]
		r, g, b = label_color_map[lbl]
		out_dat.appendRow([r, g, b])
	return

def onExpressionChange(par, val, prev):
	return

def onExportChange(par, val, prev):
	return

def onEnableChange(par, val, prev):
	return

def onModeChange(par, val, prev):
	return
	