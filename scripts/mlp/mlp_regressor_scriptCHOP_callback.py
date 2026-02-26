"""
ScriptCHOP callback for MLP Regressor. Cook: input[0] -> parent.ps.Predict -> one channel 'predictions'
with float32 values (one per regression output). No model: output zeros; channel count from Meta.n_outputs (default 1).
"""

import numpy as np

def onSetupParameters(scriptOp):
	"""No custom parameters."""
	return

def onCook(scriptOp):
	scriptOp.clear()
	scriptOp.isTimeSlice = False

	# No pipeline/Nfeat: output zeros; channel count from Meta so downstream layout is stable
	if parent.ps.Pipeline is None or parent.ps.Nfeat is None:
		try:
			n_outputs = parent.ps.Meta.get('n_outputs', 1) or 1
		except Exception:
			n_outputs = 1
		preds = np.zeros((n_outputs,), dtype=np.float32)
	else:
		arr = scriptOp.inputs[0].numpyArray()
		if arr.dtype != np.float32:
			arr = arr.astype(np.float32)
		preds = parent.ps.Predict(arr)
		if preds is None:
			try:
				n_outputs = parent.ps.Meta.get('n_outputs', 1) or 1
			except Exception:
				n_outputs = 1
			preds = np.zeros((n_outputs,), dtype=np.float32)

	scriptOp.numSamples = preds.size
	scriptOp.rate = 60
	chan = scriptOp.appendChan('predictions')
	chan.copyNumpyArray(preds)

