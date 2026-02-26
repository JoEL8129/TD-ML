"""
Script CHOP callback for VAE decode.
Input: latent CHOP (latent_dim channels; one sample per channel = one latent vector).
Output: reconstruction CHOP with n_features channels (recon_0, recon_1, ...).
Requires parent COMP with extvae and a trained model.
"""
import numpy as np


def onSetupParameters(scriptOp):
	# No parameters.
	return


def onCook(scriptOp):
	scriptOp.isTimeSlice = False
	# Resolve parent extvae; Pipeline, Feat, Latentdim must be available.
	try:
		parentcomp = parent().par.Comp.eval().ext.extvae
		pipeline = parentcomp.Pipeline
		n_feat = parentcomp.Feat
		latent_dim = getattr(parentcomp, 'Latentdim', None)
	except Exception:
		debug('VAE decode: extension not ready yet')
		pipeline = None
		n_feat = None
		latent_dim = None

	if pipeline is None or n_feat is None or latent_dim is None:
		msg = "No VAE model found. Load or train a model first."
		try:
			td = op("labelOut")
			if td.text != msg:
				td.text = msg
		except Exception:
			pass
		scriptOp.clear()
		scriptOp.numSamples = 1
		scriptOp.rate = 60
		scriptOp.appendChan("recon_0")[0] = 0.0
		return

	# Input CHOP: one sample per channel => one latent vector. numpyArray is (numChans, numSamples); [:, 0] = first sample.
	arr = scriptOp.inputs[0].numpyArray().astype(np.float32)
	if arr.ndim == 2:
		# Accept (latent_dim x 1) or (1 x latent_dim).
		if arr.shape[0] >= latent_dim and arr.shape[1] >= 1:
			z = arr[:latent_dim, 0]
		elif arr.shape[1] >= latent_dim and arr.shape[0] >= 1:
			z = arr[0, :latent_dim]
		else:
			z = np.zeros((latent_dim,), dtype=np.float32)
	elif arr.ndim == 1:
		z = arr[:latent_dim] if arr.size >= latent_dim else np.zeros((latent_dim,), dtype=np.float32)
	else:
		z = np.zeros((latent_dim,), dtype=np.float32)
	z = np.asarray(z, dtype=np.float32)
	if z.size != latent_dim:
		# Pad to latent_dim.
		pad = np.zeros((latent_dim,), dtype=np.float32)
		pad[:z.size] = z
		z = pad

	try:
		# Decode to original scale; on error, label shows message and output is zero-filled.
		recon = parentcomp.Decode(z)
		recon = np.asarray(recon, dtype=np.float32).ravel()
		if recon.size != n_feat:
			recon = np.zeros((n_feat,), dtype=np.float32)
	except Exception as e:
		try:
			td = op("labelOut")
			td.text = "VAE decode failed: " + str(e)
		except Exception:
			pass
		recon = np.zeros((n_feat,), dtype=np.float32)

	# Output: single-sample CHOP, one channel per feature.
	scriptOp.clear()
	scriptOp.numSamples = 1
	scriptOp.rate = 60
	for i in range(n_feat):
		chan = scriptOp.appendChan("recon_%d" % i)
		chan[0] = float(recon[i])
