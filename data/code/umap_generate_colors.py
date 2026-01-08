# me - this DAT
# par - the Par object that has changed
# val - the current value
# prev - the previous value
# 
# Make sure the corresponding toggle is enabled in the Parameter Execute DAT.

def onValueChange(par, prev):
	# use par.eval() to get current value
	return

# The changes are a list of named tuples, where each tuple is (Par, previous value)
def onValuesChanged(changes):
	for c in changes:
		# use par.eval() to get current value
		par = c.par
		prev = c.prev
	return

def onPulse(par):
	import random

	# 1) Read your labels table
	label_dat = op('labelsDAT')        # replace with your labels DAT path
	rows = label_dat.rows()       # skip header row if present
	labels = [row[0].val for row in rows]

	# 2) Compute unique labels and assign random colors
	unique_labels = sorted(set(labels))
	label_color_map = {}
	for lbl in unique_labels:
		# generate a random RGB color as floats [0.0–1.0]
		label_color_map[lbl] = [random.random() for _ in range(3)]

	# 3) Write out the label→color mapping
	map_dat = op('labelColorDAT')      # a Table DAT to hold the legend
	map_dat.clear()
	map_dat.appendRow(['label','r','g','b'])
	for lbl in unique_labels:
		r, g, b = label_color_map[lbl]
		map_dat.appendRow([lbl, r, g, b])

	# 4) Merge colors into your UMAP 2D output
	umap_dat = op('umap2D_DAT')        # your DAT with columns [x,y]
	out_dat  = op('colored2D_DAT')     # a new Table DAT for [x,y,r,g,b]
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
	