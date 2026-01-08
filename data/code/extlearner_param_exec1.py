# me - this DAT
# par - the Par object that has changed
# val - the current value
# prev - the previous value
# 
# Make sure the corresponding toggle is enabled in the Parameter Execute DAT.

def onValueChange(par, prev):
	# use par.eval() to get current value
	if par.name == 'Menu' and par.eval()!='None':
		color = parent.ext.StoredColors[parent.ext.par.Menu.eval()]
		parent.ext.par.Colorr = color[0]
		parent.ext.par.Colorg = color[1]
		parent.ext.par.Colorb = color[2]


	return

# Called at end of frame with complete list of individual parameter changes.
# The changes are a list of named tuples, where each tuple is (Par, previous value)
def onValuesChanged(changes):
	for c in changes:
		# use par.eval() to get current value
		par = c.par
		prev = c.prev
	return

def onPulse(par):
	parentOp = parent.ext
	if par.name == 'Save':
		parentOp.Save(name=parentOp.par.Name.eval(), color=[parentOp.par.Colorr.eval(),parentOp.par.Colorg.eval(),parentOp.par.Colorb.eval()])

	if par.name == 'Clear':
		parentOp.Clear()
	

	return

def onExpressionChange(par, val, prev):
	return

def onExportChange(par, val, prev):
	return

def onEnableChange(par, val, prev):
	return

def onModeChange(par, val, prev):
	return
	