"""
VAE (Variational Autoencoder) extension for TouchDesigner.
Unsupervised: single table of feature vectors (rows=samples, cols=features).
Learns a latent space; Encode() and Decode() support latent interpolation / in-between use.

COMP requirements:
- DATs: x (feature table), x_chan_names (optional), training_params_no_header (optional)
- Parameters: Modelfolder, Modelmenu, Loadfile, etc. (same pattern as LSTM)
"""

import copy

import torch

import helper_modules as hf
#from ParTemplate import ParTemplate
from vae_nn import VAEModule, VAEPipeline
from nn_base_ext import nnbaseext


class extvae(nnbaseext):
	"""
	VAE extension: trains on table x (unsupervised), save/load,
	Encode/Decode for latent interpolation.
	"""

	def __init__(self, ownerComp):
		super().__init__(ownerComp)

# --- nnbaseext overrides ---

	def _extra_stored_items(self):
		# Custom stored parameters beyond base (Latentdim).
		return [
			{'name': 'Latentdim', 'default': None, 'readOnly': False, 'property': True, 'dependable': True},
		]

	def _prepare_train_data(self):
		# Build training dict from x table; x_table must exist on COMP.
		if self.x_table is None:
			raise ValueError('No x table found on COMP.')
		self.x = hf._table_to_numpy(self.x_table)
		return {'x': self.x}

	def _read_train_params(self):
		# Training hyperparameters from COMP pars (with defaults).
		latent_dim = self._get_param_value('Latentdim', 'latent_dim', 8, 'int')
		enc_hidden = self._get_param_value('Enchidden', 'enc_hidden', 400, 'int')
		dec_hidden = self._get_param_value('Dechidden', 'dec_hidden', 400, 'int')
		max_epochs = self._get_param_value('Maxepochs', 'max_epochs', 50, 'int')
		lr = self._get_param_value('Lr', 'lr', 1e-3, 'float')
		batch_size = self._get_param_value('Batchsize', 'batch_size', 64, 'int')
		device_str = self._get_param_value('Device', 'device', 'cpu', 'str').lower()
		device = 'cuda' if (device_str in ('cuda', 'gpu') and torch.cuda.is_available()) else 'cpu'  # fallback to cpu if no CUDA
		beta = self._get_param_value('Beta', 'beta', 1.0, 'float')
		return {
			'latent_dim': latent_dim,
			'enc_hidden': enc_hidden,
			'dec_hidden': dec_hidden,
			'max_epochs': max_epochs,
			'lr': lr,
			'batch_size': batch_size,
			'device': device,
			'beta': beta,
		}

	def _build_and_train(self, data, params):
		# Instantiate VAEPipeline and fit on x.
		x = data['x']
		n_features = int(x.shape[1])
		pipeline = VAEPipeline(
			n_features=n_features,
			latent_dim=params['latent_dim'],
			enc_hidden=params['enc_hidden'],
			dec_hidden=params['dec_hidden'],
			lr=params['lr'],
			max_epochs=params['max_epochs'],
			batch_size=params['batch_size'],
			device=params['device'],
			beta=params['beta'],
		)
		pipeline.fit(x)
		return pipeline

	def _build_save_payload(self):
		# Serializable dict for save: config, scalers, VAE state_dict.
		return {
			"pipeline_config": {
				"n_features": self.Pipeline.n_features,
				"latent_dim": self.Pipeline.latent_dim,
				"enc_hidden": self.Pipeline.enc_hidden,
				"dec_hidden": self.Pipeline.dec_hidden,
				"lr": self.Pipeline.lr,
				"max_epochs": self.Pipeline.max_epochs,
				"batch_size": self.Pipeline.batch_size,
				"device": self.Pipeline.device,
				"beta": self.Pipeline.beta,
			},
			"scalers": {"x_scaler": copy.deepcopy(self.Pipeline.x_scaler)},
			"vae_state_dict": copy.deepcopy(self.Pipeline.vae.state_dict()),
		}

	def _load_from_payload(self, loaded):
		# Supports legacy (pipeline object) and current (config + scalers + state_dict) format.
		if "pipeline" in loaded:
			self.Pipeline = loaded.get("pipeline", None)
			return loaded.get("meta", {}) or {}

		config = loaded.get("pipeline_config", {})
		scalers = loaded.get("scalers", {})
		state_dict = loaded.get("vae_state_dict")
		if not config or not scalers:
			raise ValueError("Missing required components in saved model")

		# Reconstruct pipeline from config + scalers + state_dict (no fit).
		pipeline = VAEPipeline.__new__(VAEPipeline)
		pipeline.n_features = config["n_features"]
		pipeline.latent_dim = config["latent_dim"]
		pipeline.enc_hidden = config["enc_hidden"]
		pipeline.dec_hidden = config["dec_hidden"]
		pipeline.lr = config.get("lr", 1e-3)
		pipeline.max_epochs = config.get("max_epochs", 50)
		pipeline.batch_size = config.get("batch_size", 64)
		pipeline.device = config.get("device", "cpu")
		pipeline.beta = config.get("beta", 1.0)
		pipeline.x_scaler = scalers["x_scaler"]
		pipeline.vae = VAEModule(
			input_dim=pipeline.n_features,
			latent_dim=pipeline.latent_dim,
			enc_hidden=pipeline.enc_hidden,
			dec_hidden=pipeline.dec_hidden
		)
		if state_dict is not None:
			pipeline.vae.load_state_dict(state_dict)
		pipeline.vae.eval()
		self.Pipeline = pipeline
		return loaded.get("meta", {}) or {}

	def SetupNnPars(self):
		# Register VAE training parameters on the COMP (requires ParTemplate).
		page = self.GetPage('VAE Training')
		pars = [
			ParTemplate('Latentdim', par_type='int', label='Latent Dimension', default=8, min=1, max=512, callback=False),
			ParTemplate('Enchidden', par_type='int', label='Encoder Hidden Size', default=400, min=1, max=10000, callback=False),
			ParTemplate('Dechidden', par_type='int', label='Decoder Hidden Size', default=400, min=1, max=10000, callback=False),
			ParTemplate('Maxepochs', par_type='int', label='Max Epochs', default=50, min=1, max=10000, callback=False),
			ParTemplate('Lr', par_type='float', label='Learning Rate', default=0.001, min=1e-6, max=1.0, norm_min=0.0, norm_max=1.0, callback=False),
			ParTemplate('Batchsize', par_type='int', label='Batch Size', default=64, min=1, max=10000, callback=False),
			ParTemplate('Beta', par_type='float', label='Beta (KL weight)', default=1.0, min=0.0, max=10.0, norm_min=0.0, norm_max=10.0, callback=False),
			ParTemplate('Device', par_type='menu', label='Device', menu_names=['cpu', 'cuda'], menu_labels=['CPU', 'CUDA'], default='cpu', callback=False),
		]
		for p in pars:
			p.createPar(page)

	def _build_extra_meta(self):
		# Metadata written into save (kind, latent_dim).
		latent_dim = int(getattr(self.Pipeline, 'latent_dim', 8)) if self.Pipeline is not None else 8
		return {
			"kind": "train_vae",
			"latent_dim": latent_dim,
		}

	def _restore_state_from_pipeline(self):
		# Sync Latentdim par from loaded pipeline.
		super()._restore_state_from_pipeline()
		try:
			if self.Pipeline is not None:
				self.Latentdim = int(self.Pipeline.latent_dim)
		except Exception:
			pass

	def _after_train(self):
		# Refresh channel names table after training.
		self.WriteChanNames()

	def _on_load_complete(self):
		self.WriteChanNames()

	def _clear_subclass(self):
		# Clear VAE-specific stored state.
		self.Latentdim = None

	def _on_pulse_extra(self, par):
		# CreateDecoder pulse handler.
		if par.name == "Createdecoder":
			self.CreateDecoder()

# --- VAE-specific methods ---

	def CreateDecoder(self):
		# Copy vae_decoder template into parent network and wire to this COMP.
		source_decoder = self.ownerComp.op('vae_decoder')
		parent_net = self.ownerComp.parent()
		new_decoder = parent_net.copy(source_decoder)
		offset_x = 300
		new_decoder.nodeX = self.ownerComp.nodeX + offset_x
		new_decoder.nodeY = self.ownerComp.nodeY
		new_decoder.par.Comp.val = self.ownerComp.name
		self.ownerComp.outputConnectors[0].connect(new_decoder.inputConnectors[0])
		new_decoder.allowCooking = False
		new_decoder.allowCooking = True

	def WriteChanNames(self):
		# Write x_channel_names from Meta to table1 for display.
		meta = getattr(self, 'Meta', None) or {}
		try:
			tbl = self.ownerComp.op('table1')
			if tbl:
				tbl.clear()
				for item in meta.get('x_channel_names', []):
					tbl.appendCol(item)
		except Exception:
			pass

	# --- Promoted API (latent interpolation) ---

	def Encode(self, X, deterministic=None):
		"""Encode feature vector(s) to latent. X: (n_features,) or (batch, n_features); returns (1D or 2D).
		If deterministic is None, uses COMP par Encodedeterministic. True => mu (reproducible); False => sampled z."""
		if self.Pipeline is None:
			raise RuntimeError("No trained VAE. Train or load a model first.")
		if deterministic is None:
			deterministic = bool(self.ownerComp.par.Encodedeterministic.eval())
		return self.Pipeline.Encode(X, deterministic=deterministic)

	def Decode(self, z):
		"""Decode latent z to reconstruction. z: (latent_dim,) or (batch, latent_dim); return shape matches."""
		if self.Pipeline is None:
			raise RuntimeError("No trained VAE. Train or load a model first.")
		return self.Pipeline.Decode(z)

	def EncodeDecode(self, X, deterministic=None):
		"""Full forward pass; returns (recon, mu, logvar, latent). deterministic=None uses COMP par Encodedeterministic."""
		if self.Pipeline is None:
			raise RuntimeError("No trained VAE. Train or load a model first.")
		if deterministic is None:
			deterministic = bool(self.ownerComp.par.Encodedeterministic.eval())
		return self.Pipeline.EncodeDecode(X, deterministic=deterministic)
