import numpy as np

def onSetupParameters(scriptOp):
	return
def _get_buffer(scriptOp, seq_len, n_feat):
	buf = scriptOp.storage.get('buf', None)
	filled = scriptOp.storage.get('filled', 0)
	if buf is None or buf.shape != (seq_len, n_feat):
		buf = np.zeros((seq_len, n_feat), dtype=np.float32)
		filled = 0
		scriptOp.storage['buf'] = buf
		scriptOp.storage['filled'] = filled
	return buf, filled

def _push(scriptOp, x_vec):
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

	# No model yet -> friendly message + placeholder out
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

	# 1) Read input as a single feature vector
	arr = scriptOp.inputs[0].numpyArray().astype(np.float32)
	if arr.ndim == 2 and arr.shape[1] >= 1:
		vals = arr[:, 0]  # first sample per channel
	elif arr.ndim == 1:
		vals = arr
	else:
		raise ValueError("Unexpected input shape: %s" % (arr.shape,))

	# 2) Pad/truncate to n_feat
	L = vals.shape[0]
	if L < n_feat:
		padded = np.zeros((n_feat,), dtype=np.float32)
		padded[:L] = vals
		vals = padded
	elif L > n_feat:
		vals = vals[:n_feat]

	# 3) Update rolling window buffer
	buf, filled = _get_buffer(scriptOp, seq_len, n_feat)
	buf, filled = _push(scriptOp, vals)

	# Not enough history yet? Output zeros (or keep last prediction if you prefer)
	if filled < seq_len:
		scriptOp.clear()
		scriptOp.numSamples = 1
		scriptOp.rate = 60
		chan = scriptOp.appendChan('predictions')
		chan[0] = 0.0
		return

	# 4) Predict using the single window (seq_len, n_feat)
	try:
		preds = pipeline.predict_window(buf).astype(np.float32)  # shape (O,)
	except Exception as e:
		debug = f"LSTM predict failed: {repr(e)}"
		try:
			td = op("labelOut")
			td.text = debug
		except Exception:
			pass
		preds = np.zeros((1,), dtype=np.float32)

	# 5) Output CHOP
	scriptOp.clear()
	scriptOp.numSamples = preds.size
	scriptOp.rate = 60
	chan = scriptOp.appendChan('predictions')
	chan.copyNumpyArray(preds)
