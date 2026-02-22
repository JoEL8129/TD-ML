# me - this DAT
# scriptOp - the OP which is cooking

# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
	return

# called whenever custom pulse parameter is pushed
def onPulse(par):
	return

def onCook(scriptOp):
	scriptOp.clear()
	scriptOp.isTimeSlice = False
	if not hasattr(parent.ps, 'Predict'):
		scriptOp.numSamples = 1
		for ch in ('x', 'y'):
			scriptOp.appendChan(ch)[0] = 0
		return  # UMAP extension failed to load (e.g. Numba import error)
	arr = scriptOp.inputs[0].numpyArray()
	# assume one sample per frame, so pick the first (and only) column:
	emb_pt = parent.ps.Predict(arr)

	# 4) write out a 2-channel CHOP with channels named 'x' and 'y'
	scriptOp.clear()
	scriptOp.numSamples = 1
	# scriptOp.rate = 60
	# scriptOp.numChannels = 2
	x = scriptOp.appendChan('x')
	y = scriptOp.appendChan('y')
	# set the single sample:
	x[0] = emb_pt[0]
	y[0] = emb_pt[1]
	if parent.ps.par.Ncomponents == 3:
		z = scriptOp.appendChan('z')
		z[0] = emb_pt[2]
	return
