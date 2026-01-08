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
	classes  = getattr(parent.ps, 'Classes', None)

	# No model yet -> friendly message + placeholder out
	if pipeline is None or n_feat is None or seq_len is None or classes is None:
		msg = "No LSTM classifier model found. Click TRAIN to create one."
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

	# Get prediction method from parent parameter (if available)
	method = "predict"  # default to class prediction
	try:
		if hasattr(parent.par, 'Predictmethod'):
			method = parent.par.Predictmethod.eval()
	except:
		pass

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
		if method == "predict":
			# Return class label (as integer index)
			pred_label = pipeline.predict_window(buf)
			# Convert label to class index for output
			try:
				class_idx = np.where(classes == pred_label)[0][0]
			except:
				class_idx = 0
			preds = np.array([float(class_idx)], dtype=np.float32)
			
			# Update text output
			try:
				td = op("labelOut")
				td.text = str(pred_label)
			except Exception:
				pass
				
		elif method == "predict_proba":
			# Return probabilities for all classes
			proba = pipeline.predict_proba_window(buf)  # shape (n_classes,)
			preds = proba.astype(np.float32)
			
			# Find best class
			best_i = int(np.argmax(proba))
			pred_label = classes[best_i]
			
			# Update text output with probabilities
			try:
				td = op("labelOut")
				out_text = "\n".join(f"{cls}: {p*100:.1f}%" for cls, p in zip(classes, proba))
				if td.text != out_text:
					td.text = out_text
			except Exception:
				pass
				
		else:
			# Default to predict
			pred_label = pipeline.predict_window(buf)
			try:
				class_idx = np.where(classes == pred_label)[0][0]
			except:
				class_idx = 0
			preds = np.array([float(class_idx)], dtype=np.float32)
			
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
	
	# Create channels based on prediction method
	if method == "predict_proba":
		# One channel per class
		for i, cls in enumerate(classes):
			chan = scriptOp.appendChan(f'prob_{cls}')
			chan[0] = float(preds[i])
	else:
		# Single channel with class index
		chan = scriptOp.appendChan('predictions')
		chan.copyNumpyArray(preds)

