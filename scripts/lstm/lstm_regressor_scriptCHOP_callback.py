"""
ScriptCHOP callback for LSTM Regressor. Cook: input[0] -> rolling window (seq_len, n_feat) -> parent.ps.predict_window -> channel 'predictions' (one per output).
Uses scriptOp.storage for rolling buffer; outputs zeros until buffer is full or when no model.
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
	scriptOp.isTimeSlice = False

	pipeline = parent.ps.Pipeline
	n_feat   = parent.ps.Feat
	seq_len  = getattr(parent.ps, 'SeqLen', None)

	# No pipeline/Feat/SeqLen: set labelOut message, single channel 0, return
	if pipeline is None or n_feat is None or seq_len is None:
		msg = "No LSTM model found. Click TRAIN to create one."
		try:
			td = op("labelOut")
			if td.text != msg:
				td.text = msg
		except Exception:
			pass
		scriptOp.clear()
		scriptOp.numSamples = 1
		scriptOp.rate = 60
		scriptOp.appendChan("predictions")[0] = 0.0
		return

	# Input: first sample (or 1D); float32
	arr = scriptOp.inputs[0].numpyArray()
	if arr.dtype != np.float32:
		arr = arr.astype(np.float32)
	if arr.ndim == 2 and arr.shape[1] >= 1:
		vals = arr[:, 0]  # first sample per channel
	elif arr.ndim == 1:
		vals = arr
	else:
		raise ValueError("Unexpected input shape: %s" % (arr.shape,))

	# Pad or trim to n_feat for pipeline
	L = vals.shape[0]
	if L < n_feat:
		padded = np.zeros((n_feat,), dtype=np.float32)
		padded[:L] = vals
		vals = padded
	elif L > n_feat:
		vals = vals[:n_feat]

	# Update rolling window; output zeros until seq_len frames received
	buf, filled = _get_buffer(scriptOp, seq_len, n_feat)
	buf, filled = _push(scriptOp, vals)

	if filled < seq_len:
		scriptOp.clear()
		scriptOp.numSamples = 1
		scriptOp.rate = 60
		chan = scriptOp.appendChan('predictions')
		chan[0] = 0.0
		return

	# predict_window(buf) -> (n_outputs,) float32
	try:
		preds = pipeline.predict_window(buf).astype(np.float32)
	except Exception as e:
		debug = f"LSTM predict failed: {repr(e)}"
		try:
			td = op("labelOut")
			td.text = debug
		except Exception:
			pass
		preds = np.zeros((1,), dtype=np.float32)

	# Write CHOP: one channel 'predictions', numSamples = preds.size
	scriptOp.clear()
	scriptOp.numSamples = preds.size
	scriptOp.rate = 60
	chan = scriptOp.appendChan('predictions')
	chan.copyNumpyArray(preds)
