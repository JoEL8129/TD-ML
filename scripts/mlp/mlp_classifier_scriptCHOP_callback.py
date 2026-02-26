"""
ScriptCHOP callback for MLP Classifier. Cook: input[0] -> parent.ps.Predict -> one channel per class,
values from Predictmethod (one-hot, proba, or multilabel 0/1/P). No model: one channel per Meta.classes with 0, or 'empty'.
"""

import numpy as np

def onSetupParameters(scriptOp):
	"""No custom parameters."""
	return

def onCook(scriptOp):
	scriptOp.clear()
	scriptOp.isTimeSlice = False

	# No pipeline/Nfeat: one channel per Meta.classes (value 0) so layout matches trained model; else single 'empty' channel
	if parent.ps.Pipeline is None or parent.ps.Nfeat is None:
		try:
			classes = parent.ps.Meta.get('classes', None)
		except Exception:
			classes = None
		if classes:
			scriptOp.numSamples = 1
			for name in classes:
				chan = scriptOp.appendChan(str(name))
				chan[0] = 0
		else:
			scriptOp.numSamples = 1
			chan = scriptOp.appendChan('empty')
			chan[0] = 0
		return

	arr = scriptOp.inputs[0].numpyArray()
	if arr.dtype != np.float32:
		arr = arr.astype(np.float32)
	result = parent.ps.Predict(arr)

	if result is None:
		scriptOp.numSamples = 1
		chan = scriptOp.appendChan('empty')
		chan[0] = 0
		return

	# One channel per class; values from Predict (one-hot, proba, or multilabel 0/1/P per class)
	class_names, values = result
	scriptOp.numSamples = 1
	for name, val in zip(class_names, values):
		chan = scriptOp.appendChan(str(name))
		chan[0] = float(val)
