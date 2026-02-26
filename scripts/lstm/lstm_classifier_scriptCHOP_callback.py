"""
ScriptCHOP callback for LSTM Classifier. Cook: input[0] -> rolling window (seq_len, n_feat) -> parent.ps.Predict(buf) -> one channel per class.
Uses scriptOp.storage for rolling buffer; outputs zeros per class until buffer is full or when no model.
"""

import numpy as np

def onSetupParameters(scriptOp):
	"""No custom parameters."""
	return

def _get_buffer(scriptOp, seq_len, n_feat):
	"""Returns or creates rolling buffer (seq_len, n_feat) and filled count in scriptOp.storage."""
	buf = scriptOp.storage.get('buf', None)
	filled = scriptOp.storage.get('filled', 0)
	if buf is None or buf.shape != (seq_len, n_feat):
		buf = np.zeros((seq_len, n_feat), dtype=np.float32)
		filled = 0
		scriptOp.storage['buf'] = buf
		scriptOp.storage['filled'] = filled
	return buf, filled

def _push(scriptOp, x_vec):
	"""Appends x_vec to rolling buffer (shift left, new row at end); updates filled count."""
	buf = scriptOp.storage['buf']
	filled = scriptOp.storage['filled']
	buf[:-1] = buf[1:]
	buf[-1] = x_vec
	filled = min(filled + 1, buf.shape[0])
	scriptOp.storage['filled'] = filled
	return buf, filled

def onCook(scriptOp):
	scriptOp.clear()
	scriptOp.isTimeSlice = False

	pipeline = parent.ps.Pipeline
	n_feat   = parent.ps.Feat
	seq_len  = getattr(parent.ps, 'SeqLen', None)
	classes  = getattr(parent.ps, 'Classes', None)

	# No pipeline/Feat/SeqLen/Classes: one channel per class (value 0) or single 'empty'
	if pipeline is None or n_feat is None or seq_len is None or classes is None:
		scriptOp.numSamples = 1
		if classes:
			for name in classes:
				chan = scriptOp.appendChan(str(name))
				chan[0] = 0
		else:
			chan = scriptOp.appendChan('empty')
			chan[0] = 0
		return

	# Input: first sample (or 1D); float32; pad or trim to n_feat
	arr = scriptOp.inputs[0].numpyArray()
	if arr.dtype != np.float32:
		arr = arr.astype(np.float32)
	if arr.ndim == 2 and arr.shape[1] >= 1:
		vals = arr[:, 0]
	elif arr.ndim == 1:
		vals = arr
	else:
		raise ValueError("Unexpected input shape: %s" % (arr.shape,))

	L = vals.shape[0]
	if L < n_feat:
		padded = np.zeros((n_feat,), dtype=np.float32)
		padded[:L] = vals
		vals = padded
	elif L > n_feat:
		vals = vals[:n_feat]

	# Rolling window; output zeros per class until seq_len frames received
	buf, filled = _get_buffer(scriptOp, seq_len, n_feat)
	buf, filled = _push(scriptOp, vals)

	if filled < seq_len:
		scriptOp.numSamples = 1
		for name in classes:
			chan = scriptOp.appendChan(str(name))
			chan[0] = 0
		return

	# Predict(buf) -> (class_names, values); one channel per class
	result = parent.ps.Predict(buf)

	if result is None:
		scriptOp.numSamples = 1
		chan = scriptOp.appendChan('empty')
		chan[0] = 0
		return

	class_names, values = result

	# One channel per class, one sample
	scriptOp.numSamples = 1
	for name, val in zip(class_names, values):
		chan = scriptOp.appendChan(str(name))
		chan[0] = float(val)
