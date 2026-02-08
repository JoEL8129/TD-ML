# me - this DAT
# scriptOp - the OP which is cooking

# press 'Setup Parameters' in the OP to call this function to re-create the parameters.
def onSetupParameters(scriptOp):
	return

# called whenever custom pulse parameter is pushed
def onPulse(par):
	return

def onCook(scriptOP):
	scriptOP.clear()
	scriptOP.isTimeSlice = False
	if not hasattr(parent.ps, 'Predict'):
		scriptOP.numSamples = 1
		for ch in ('x', 'y'):
			scriptOP.appendChan(ch)[0] = 0
		return  # UMAP extension failed to load (e.g. Numba import error)
	arr = scriptOP.inputs[0].numpyArray()
	# assume one sample per frame, so pick the first (and only) column:
	emb_pt = parent.ps.Predict(arr)


    # 4) write out a 2-channel CHOP with channels named 'x' and 'y'
	scriptOP.clear()
	scriptOP.numSamples  = 1
	#scriptOP.rate = 60
    #scriptOp.numChannels = 2
	x = scriptOP.appendChan('x')
	y = scriptOP.appendChan('y')
	
    # set the single sample:
	x[0] = emb_pt[0]
	y[0] = emb_pt[1]
	if parent.ps.par.Ncomponents == 3:
		z = scriptOP.appendChan('z')
		z[0] = emb_pt[2]
	return
