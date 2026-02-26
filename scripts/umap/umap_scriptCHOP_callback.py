# Script CHOP callbacks: me (this DAT), scriptOp (the cooking OP).
# onSetupParameters runs when "Setup Parameters" is pressed on the OP.

def onSetupParameters(scriptOp):
	return

def onPulse(par):
	"""Custom pulse parameter callback (no-op)."""
	return

def onCook(scriptOp):
	"""Run UMAP transform on first input CHOP; output CHOP has x,y(,z) embedding channels."""
	scriptOp.clear()
	scriptOp.isTimeSlice = False

	# No extension or Predict missing 
	if not hasattr(parent.ps, 'Predict'):
		scriptOp.numSamples = 1
		for ch in ('x', 'y'):
			scriptOp.appendChan(ch)[0] = 0
		return
	# No input CHOP
	if not scriptOp.inputs or len(scriptOp.inputs) == 0:
		scriptOp.numSamples = 1
		for ch in ('x', 'y'):
			scriptOp.appendChan(ch)[0] = 0
		return
	arr = scriptOp.inputs[0].numpyArray()
	# Empty or missing array
	if arr is None or arr.size == 0:
		scriptOp.numSamples = 1
		for ch in ('x', 'y'):
			scriptOp.appendChan(ch)[0] = 0
		return
	# Model not fitted yet
	if parent.ps.Reducer is None or parent.ps.Scaler is None:
		scriptOp.numSamples = 1
		for ch in ('x', 'y'):
			scriptOp.appendChan(ch)[0] = 0
		return

	try:
		emb_pt = parent.ps.Predict(arr)
	except Exception as e:
		print(f"[UMAP Predict] skipped: {e}")
		scriptOp.numSamples = 1
		for ch in ('x', 'y'):
			scriptOp.appendChan(ch)[0] = 0
		return

	# Write single-sample CHOP: x, y and z if n_components == 3
	scriptOp.clear()
	scriptOp.numSamples = 1
	x = scriptOp.appendChan('x')
	y = scriptOp.appendChan('y')
	x[0] = emb_pt[0]
	y[0] = emb_pt[1]
	if parent.ps.par.Ncomponents.eval() == 3:
		z = scriptOp.appendChan('z')
		z[0] = emb_pt[2]
	return
