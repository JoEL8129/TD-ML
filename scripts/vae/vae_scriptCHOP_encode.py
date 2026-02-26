"""
Script CHOP callback for VAE encode.
Input: feature CHOP (one sample per channel = one feature vector).
Output: latent CHOP with latent_dim channels (z0, z1, ...).
Requires parent COMP with extvae and a trained model.
"""
import numpy as np


def onSetupParameters(scriptOp):
	# No parameters.
	return


def onCook(scriptOp):
	scriptOp.isTimeSlice = False
	# Resolve parent extvae (parent.ps); Pipeline, Feat, Latentdim must be available.
	try:
		pipeline = parent.ps.Pipeline
		n_feat = parent.ps.Feat
		latent_dim = getattr(parent.ps, 'Latentdim', None)
	except Exception:
		debug('VAE encode: extension not ready yet')
		pipeline = None
		n_feat = None
		latent_dim = None

	if pipeline is None or n_feat is None or latent_dim is None:
		msg = "No VAE model found. Click TRAIN to create one."
		try:
			td = op("labelOut")
			if td.text != msg:
				td.text = msg
		except Exception:
			pass
		scriptOp.clear()
		scriptOp.numSamples = 1
		scriptOp.rate = 60
		scriptOp.appendChan("z0")[0] = 0.0
		return

	# Input CHOP: one sample per channel => one feature vector (length n_feat).
	arr = scriptOp.inputs[0].numpyArray().astype(np.float32)
	if arr.ndim == 2 and arr.shape[1] >= 1:
		vals = arr[:, 0]
	elif arr.ndim == 1:
		vals = arr
	else:
		vals = np.zeros((n_feat,), dtype=np.float32)

	# Pad or truncate to n_feat.
	L = vals.shape[0]
	if L < n_feat:
		padded = np.zeros((n_feat,), dtype=np.float32)
		padded[:L] = vals
		vals = padded
	elif L > n_feat:
		vals = vals[:n_feat].astype(np.float32)

	try:
		# Encode to latent; on error, label shows message and output is zero-filled.
		z = parent.ps.Encode(vals)
		z = np.asarray(z, dtype=np.float32).ravel()
		if z.size != latent_dim:
			z = np.zeros((latent_dim,), dtype=np.float32)
	except Exception as e:
		try:
			td = op("labelOut")
			td.text = "VAE encode failed: " + str(e)
		except Exception:
			pass
		z = np.zeros((latent_dim,), dtype=np.float32)

	# Output: single-sample CHOP, one channel per latent dimension.
	scriptOp.clear()
	scriptOp.numSamples = 1
	scriptOp.rate = 60
	for i in range(latent_dim):
		chan = scriptOp.appendChan("z%d" % i)
		chan[0] = float(z[i])
