"""
Base extension for LSTM-based TouchDesigner components.
Shared logic for storage, file I/O, metadata, menus, and par callbacks;
subclasses implement model-specific training, save/load, and prediction.
Serialization uses dill + skorch save_params (not joblib pipeline pickle).
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF
from pathlib import Path
from datetime import datetime, timezone
import os
import tempfile

import numpy as np
import dill
import sklearn
import torch
from skorch.callbacks import Callback

import helper_modules as hf


class _TrainingCancelled(Exception):
	pass


class _CancelCheckCallback(Callback):
	"""Skorch callback: raises _TrainingCancelled when flag_holder._cancel_requested is set (checked at epoch end)."""
	def __init__(self, flag_holder):
		self._flag_holder = flag_holder

	def on_epoch_end(self, net, **kwargs):
		if self._flag_holder._cancel_requested:
			raise _TrainingCancelled("Training cancelled by user")


class lstmbaseext:
	"""
	Abstract base for LSTM extensions. Train() can run in a background thread via ThreadManager (cancel via CancelTrain).
	Required overrides: _read_y_data(), _build_pipeline(), _save_pipeline_config(), _save_scalers(), _reconstruct_pipeline(), _build_extra_meta().
	Optional hooks: _extra_stored_items(), _on_train_complete(), _after_train(), _on_load_complete(), _clear_subclass().
	"""

	_train_status_label = 'Stored In COMP / not saved'

	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.ThreadManager = op.TDResources.ThreadManager
		self.Working = False
		self._train_result = None
		self._cancel_requested = False
		self.Meta = {}
		self.last_params = None

		# Stored on component: Pipeline, Feat, SeqLen, plus subclass items from _extra_stored_items()
		storedItems = [
			{'name': 'Pipeline', 'default': None, 'readOnly': False, 'property': True, 'dependable': True},
			{'name': 'Feat',     'default': None, 'readOnly': False, 'property': True, 'dependable': True},
			{'name': 'SeqLen',   'default': None, 'readOnly': False, 'property': True, 'dependable': True},
		]
		storedItems.extend(self._extra_stored_items())
		self.stored = StorageManager(self, ownerComp, storedItems)

		# Table and menu ops from component layout
		self.x_table = self.ownerComp.op('x')
		self.y_table = self.ownerComp.op('y')
		self.w_table = self.ownerComp.op('weight')
		self.x_chan_names = self.ownerComp.op('x_chan_names')
		self.y_chan_names = self.ownerComp.op('y_chan_names')
		self.menuSourceOp = self.ownerComp.op('model_menu')
		self.param_table = self.ownerComp.op('training_params_no_header')

		self.x = None
		self.y = None
		self.w = None

		self.updateMeta()
		self.UpdateMenu()

# Subclass hooks ---------------------------------------------------------

	def _extra_stored_items(self):
		"""Returns a list of additional StorageManager item dicts. Default: []."""
		return []

	def _read_y_data(self):
		"""Returns y data from self.y_table. Override for model type (labels vs numeric)."""
		raise NotImplementedError("Subclasses must implement _read_y_data()")

	def _build_pipeline(self, n_features, params):
		"""Builds and returns an unfitted SkorchLSTMPipeline. params: seq_len, hidden_size, num_layers, dropout, bidirectional, max_epochs, lr, batch_size, device. Subclass sets output dim from self.y."""
		raise NotImplementedError("Subclasses must implement _build_pipeline()")

	def _save_pipeline_config(self):
		"""Returns a dict of pipeline config for serialization."""
		raise NotImplementedError("Subclasses must implement _save_pipeline_config()")

	def _save_scalers(self):
		"""Returns a dict of scalers (deep-copied) for serialization."""
		raise NotImplementedError("Subclasses must implement _save_scalers()")

	def _reconstruct_pipeline(self, config, scalers, net_params, net_state_dict):
		"""Rebuilds pipeline from loaded config, scalers, net_params, net_state_dict. Returns reconstructed pipeline."""
		raise NotImplementedError("Subclasses must implement _reconstruct_pipeline()")

	def _build_extra_meta(self):
		"""Returns a dict of extra keys merged into meta in updateMeta(). Default: {}."""
		return {}

	def _on_train_complete(self, pipeline):
		"""Invoked after Pipeline is set, before updateMeta(). Override to set subclass state."""
		pass

	def _after_train(self):
		"""Invoked after training and metadata update, before UpdateInfo. Default: no-op."""
		pass

	def _on_load_complete(self):
		"""Invoked after Load() succeeds. Override for post-load setup."""
		pass

	def _clear_subclass(self):
		"""Invoked at end of Clear(). Override to reset subclass-specific state."""
		pass

# PAR Callbacks -----------------------------------------------------------
	# Pulse/ValueChange handlers for Train, Save, Load, Clear, Cancelthreadedtrain, menu, etc.

	def OnPulse(self, par):
		match par.name:
			case "Train" | "Train2":
				self.Train()
			case "Retrain" | "Retrain2":
				self.Partial_Fit()

			case "Save":
				self.SaveOverride()

			case "Saveinc":
				self.SaveIncremental()

			case "Reload":
				self.LoadFromMenu()

			case "Savecopyfilepulse":
				self.Save(self.ownerComp.par.Savecopyfile.eval())

			case "Reloadfromfile":
				self.Load(self.ownerComp.par.Loadfromfile.eval())

			case "Clear":
				self.Clear()

			case "Cancelthreadedtrain":
				self.CancelTrain()

			case _:
				print('none')

	def OnValueChange(self, par, prev):
		match par.name:
			case "Modelmenu":
				if prev is not None and prev != 'None':
					self.LoadFromMenu()
			case "Modelfolder":
				self.UpdateMenu()
			case "Savecopyfile":
				self.Save(par.eval())
			case "Loadfromfile":
				self.Load(par.eval())

# Helpers -----------------------------------------------------------------
	# Custom pages, LSTM par setup, param reading from component or param table.

	def GetPage(self, pageName, create_if_missing=True):
		"""Returns custom Page by name or label (case-insensitive). Creates page if missing and create_if_missing=True; else None."""
		if not pageName:
			raise ValueError("GetPage: pageName must be a non-empty string.")

		lname = pageName.lower()
		for p in self.ownerComp.customPages:
			label = getattr(p, 'label', p.name)
			if p.name.lower() == lname or str(label).lower() == lname:
				return p

		if create_if_missing:
			return self.ownerComp.appendCustomPage(pageName)

		return None

	def SetupNnPars(self):
		"""Adds LSTM training parameters (Seqlen, Hiddensize, Numlayers, Dropout, etc.) to custom page 'LSTM Training'."""
		page = self.GetPage('LSTM Training')
		pars = [
			ParTemplate(
				'Seqlen', par_type='int', label='Sequence Length',
				default=16, min=1, max=1000, callback=False
			),
			ParTemplate(
				'Hiddensize', par_type='int', label='Hidden Size',
				default=64, min=1, max=10000, callback=False
			),
			ParTemplate(
				'Numlayers', par_type='int', label='Number of Layers',
				default=1, min=1, max=10, callback=False
			),
			ParTemplate(
				'Dropout', par_type='float', label='Dropout',
				default=0.0, min=0.0, max=1.0,
				norm_min=0.0, norm_max=1.0, callback=False
			),
			ParTemplate(
				'Bidirectional', par_type='toggle', label='Bidirectional',
				default=False, callback=False
			),
			ParTemplate(
				'Maxepochs', par_type='int', label='Max Epochs',
				default=20, min=1, max=10000, callback=False
			),
			ParTemplate(
				'Lr', par_type='float', label='Learning Rate',
				default=0.001, min=1e-6, max=1.0,
				norm_min=0.0, norm_max=1.0, callback=False
			),
			ParTemplate(
				'Batchsize', par_type='int', label='Batch Size',
				default=64, min=1, max=10000, callback=False
			),
			ParTemplate(
				'Device', par_type='menu', label='Device',
				menu_names=['cpu', 'cuda'], menu_labels=['CPU', 'CUDA'],
				default='cpu', callback=False
			),
		]

		for p in pars:
			p.createPar(page)

	def _get_param_value(self, par_name, param_table_key, default_value, param_type='int'):
		"""Reads value from ownerComp.par[par_name], with fallback to param_table; returns value cast to param_type."""
		try:
			value = self.ownerComp.par[par_name].eval()
			if param_type == 'int':
				return int(value)
			elif param_type == 'float':
				return float(value)
			elif param_type == 'bool':
				return bool(value)
			elif param_type == 'str':
				return str(value)
			return value
		except Exception:
			try:
				if self.param_table:
					params = hf._sk_param_table_to_dict(self.param_table)
					value = params.get(param_table_key, default_value)
					if param_type == 'int':
						return int(value)
					elif param_type == 'float':
						return float(value)
					elif param_type == 'bool':
						return bool(value)
					elif param_type == 'str':
						return str(value)
					return value
			except Exception:
				pass
			return default_value

	def _read_train_params(self):
		"""Reads LSTM training params from component pars (fallback to param table); normalizes device to cpu/cuda."""
		seq_len       = self._get_param_value('Seqlen', 'seq_len', 16, 'int')
		hidden_size   = self._get_param_value('Hiddensize', 'hidden_size', 64, 'int')
		num_layers    = self._get_param_value('Numlayers', 'num_layers', 1, 'int')
		dropout       = self._get_param_value('Dropout', 'dropout', 0.0, 'float')
		bidirectional = self._get_param_value('Bidirectional', 'bidirectional', False, 'bool')
		max_epochs    = self._get_param_value('Maxepochs', 'max_epochs', 20, 'int')
		lr            = self._get_param_value('Lr', 'lr', 1e-3, 'float')
		batch_size    = self._get_param_value('Batchsize', 'batch_size', 64, 'int')
		device        = self._get_param_value('Device', 'device', 'cpu', 'str').lower()

		device = 'cuda' if (device in ('cuda', 'gpu') and torch.cuda.is_available()) else 'cpu'

		return {
			'seq_len': seq_len,
			'hidden_size': hidden_size,
			'num_layers': num_layers,
			'dropout': dropout,
			'bidirectional': bidirectional,
			'max_epochs': max_epochs,
			'lr': lr,
			'batch_size': batch_size,
			'device': device,
		}

# Core --------------------------------------------------------------------

	def Train(self):
		"""Reads x/y/w and params, builds pipeline. If Traininthread: enqueues worker (loading_bar, cancel callback); else fits on main thread."""
		if self.Working:
			print('Training already in progress')
			return

		self._cancel_requested = False
		self.x = hf._table_to_numpy(self.x_table)
		self.y = self._read_y_data()

		try:
			use_w = self.ownerComp.par.Useweights.eval()
		except AttributeError:
			use_w = (self.w_table is not None)
		self.w = None
		if use_w and self.w_table is not None:
			try:
				self.w = hf._weights_from_table(self.w_table)
			except Exception:
				pass

		params = self._read_train_params()
		# First column typically time/index; feature dim = cols - 1
		n_features = int(self.x.shape[1]) - 1

		self.last_params = dict(params)

		try:
			threaded = self.ownerComp.par.Traininthread.eval()
		except Exception:
			threaded = False

		if threaded:
			self.ownerComp.par.opviewer = './loading_bar'
			self.ownerComp.op('loading_bar').par.Active = True
			task = self.ThreadManager.TDTask(
				target=self._train_worker,
				SuccessHook=self._train_success,
				ExceptHook=self._train_except,
				args=(self.x, self.y, n_features, params)
			)
			self.ThreadManager.EnqueueTask(task)
			self.Working = True
			print('Training enqueued (background thread)')
		else:
			pipeline = self._build_pipeline(n_features, params)
			try:
				pipeline.fit(self.x, self.y)
			except Exception as e:
				print('Training failed:', repr(e))
				import traceback
				traceback.print_exc()
				return

			self.Pipeline = pipeline
			self._on_train_complete(pipeline)
			self.updateMeta()
			self._after_train()
			self.UpdateInfo(self._train_status_label)
			print('Training completed (main thread)')

	def _train_worker(self, x, y, n_features, params):
		"""Runs in worker thread; must not access TouchDesigner ops or parameters."""
		pipeline = self._build_pipeline(n_features, params)
		cancel_cb = _CancelCheckCallback(self)
		existing = pipeline.net.callbacks or []
		pipeline.net.callbacks = list(existing) + [('cancel_check', cancel_cb)]
		pipeline.fit(x, y)
		self._train_result = pipeline

	def _train_success(self):
		"""Main-thread callback when background training completes successfully (or was cancelled; result discarded)."""
		pipeline = self._train_result
		self._train_result = None
		if self._cancel_requested:
			self._cancel_requested = False
			self.Working = False
			print('Training cancelled — result discarded')
			return
		self.Pipeline = pipeline
		self._on_train_complete(pipeline)
		self.updateMeta()
		self._after_train()
		self.UpdateInfo(self._train_status_label)
		self.Working = False
		self.ownerComp.op('loading_bar').par.Active = False
		self.ownerComp.par.opviewer = './out1'
		print('Training completed successfully')

	def _train_except(self, *args):
		"""Main-thread callback when background training raises an exception."""
		self.Working = False
		if self._cancel_requested:
			self._cancel_requested = False
			print('Training cancelled by user')
		else:
			print('Training failed:', args)

	def CancelTrain(self):
		"""Sets _cancel_requested so worker stops after current epoch; no-op if not training."""
		if not self.Working:
			print('No training in progress to cancel')
			return
		self._cancel_requested = True
		print('Cancellation requested — training will stop after current epoch')

	def Partial_Fit(self):
		"""Incremental fit. Default: no pipeline -> message; else delegates to Train()."""
		if self.Pipeline is None:
			print("No existing model to retrain. Use Train instead.")
			return
		self.Train()

	def Save(self, name='default'):
		"""Serializes pipeline via skorch save_params (temp .pt) then dill; payload: meta, pipeline_config, scalers, net_params, net_state_dict."""
		self.updateMeta()
		path = Path(str(name))
		if path.suffix.lower() != '.joblib':
			path = path.with_suffix('.joblib')
		path.parent.mkdir(parents=True, exist_ok=True)
		if self.Pipeline is None:
			print('Save aborted: no trained Pipeline.')
			return

		net = self.Pipeline.net
		with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
			tmp_path = tmp_file.name

		try:
			net.save_params(f_params=tmp_path)

			saved_state = torch.load(tmp_path, map_location='cpu')
			net_state_dict = saved_state.get('module_state_dict', saved_state)

			# Serialize net params excluding module, callbacks, history
			net_params = net.get_params()
			clean_net_params = {}
			for k, v in net_params.items():
				if k == 'module' or k.startswith('callback') or k == 'history':
					continue
				clean_net_params[k] = v
			if 'module' in clean_net_params:
				del clean_net_params['module']

			payload = {
				"meta": self.Meta or {},
				"pipeline_config": self._save_pipeline_config(),
				"scalers": self._save_scalers(),
				"net_params": clean_net_params,
				"net_state_dict": net_state_dict,
			}

			with open(path, 'wb') as f:
				dill.dump(payload, f)
			print('saved', path.name, 'to', str(path))
		finally:
			try:
				os.unlink(tmp_path)
			except Exception:
				pass

		self.UpdateMenu()

	def Load(self, name='default'):
		"""Loads .joblib via dill. Supports legacy 'pipeline' key or reconstruct from pipeline_config, scalers, net_params, net_state_dict."""
		if name in ('None', None):
			print('No Model Found to Load, Train? Or adjust Loadfolder')
			return
		path = Path(str(name)).with_suffix('.joblib')
		if not path.exists():
			print('Load failed: file not found ->', str(path))
			return
		try:
			with open(path, 'rb') as f:
				loaded = dill.load(f)
		except Exception as e:
			print('Load failed:', repr(e))
			import traceback
			traceback.print_exc()
			return

		# Legacy format: single 'pipeline' key; else reconstruct from config + scalers + net_*
		if "pipeline" in loaded:
			self.Pipeline = loaded.get("pipeline", None)
			loaded_meta = loaded.get("meta", {}) or {}
		else:
			try:
				config = loaded.get("pipeline_config", {})
				scalers = loaded.get("scalers", {})
				net_params = loaded.get("net_params", {})
				net_state_dict = loaded.get("net_state_dict")

				if not config or not scalers:
					raise ValueError("Missing required components in saved model")

				self.Pipeline = self._reconstruct_pipeline(
					config, scalers, net_params, net_state_dict
				)
				loaded_meta = loaded.get("meta", {}) or {}
			except Exception as e:
				print('Load failed: Could not reconstruct pipeline:', repr(e))
				import traceback
				traceback.print_exc()
				return

		self.Meta.clear()
		self.Meta.update(loaded_meta)

		try:
			if self.Pipeline is not None:
				self.Feat = int(self.Pipeline.n_features)
				self.SeqLen = int(self.Pipeline.seq_len)
		except Exception:
			pass

		self._on_load_complete()
		print('loaded', path.name, 'from', path)
		self.UpdateInfo(path.name)

	def Clear(self):
		"""Resets Pipeline, Meta, Feat, SeqLen to defaults; calls _clear_subclass()."""
		self.Pipeline = None
		self.Meta.clear()
		self.Feat = None
		self.SeqLen = None
		self._clear_subclass()
		self.UpdateInfo('None')
		print('Model cleared: Pipeline and Meta reset to defaults')

# Updaters ----------------------------------------------------------------
	# Sync UI: Modelactive label and model dropdown from Modelfolder.

	def UpdateInfo(self, modelactive):
		"""Sets the component's Modelactive par (display label for current model)."""
		self.ownerComp.par.Modelactive = modelactive

	def UpdateMenu(self):
		"""Refreshes model_load_folder and repopulates models_table from the folder listing."""
		self.ownerComp.op('model_load_folder').par.refreshpulse.pulse()
		self.ownerComp.op('models_table').clear()
		if self.ownerComp.op('model_load_folder').numRows == 1:
			rown = ['None']
			self.ownerComp.op('models_table').appendRow(rown)
		else:
			for row in self.ownerComp.op('model_menu2').rows():
				self.ownerComp.op('models_table').appendRow(row)

# Metadata ----------------------------------------------------------------
	# Meta: created_utc, n_samples, n_features, seq_len, params, versions, channel names. Subclasses can add keys via _build_extra_meta().

	def updateMeta(self):
		"""Builds meta from pipeline (or x), channel names, and _build_extra_meta(); writes to self.Meta and sets self.Feat, self.SeqLen."""
		try:
			n_feat = int(getattr(self.Pipeline, 'n_features', None))
		except Exception:
			n_feat = int(self.x.shape[1]) - 1 if getattr(self, 'x', None) is not None else None

		seq_len = int(getattr(self.Pipeline, 'seq_len', 16)) if getattr(self, 'Pipeline', None) is not None else 16

		x_names = [self.x_chan_names[0, c].val for c in range(self.x_chan_names.numCols)]
		y_names = [self.y_chan_names[0, c].val for c in range(self.y_chan_names.numCols)]

		new_meta = {
			"created_utc": datetime.now(timezone.utc).isoformat(),
			"n_samples": int(self.x.shape[0]) if getattr(self, 'x', None) is not None else None,
			"n_features": n_feat,
			"seq_len": seq_len,
			"params": self.last_params,
			"sklearn": sklearn.__version__,
			"numpy": np.__version__,
			"torch": torch.__version__,
			"op_path": self.ownerComp.path,
			"x_channel_names": list(x_names),
			"y_channel_names": list(y_names),
		}

		extra = self._build_extra_meta()
		if extra:
			new_meta.update(extra)

		self.Meta.clear()
		self.Meta.update(new_meta)
		self.Feat = n_feat
		self.SeqLen = seq_len

# FileIO High Level -------------------------------------------------------
	# Save/Load using Modelfolder and Modelmenu; handle "internal stored" and incremental names.

	def SaveOverride(self):
		"""Save to Modelfolder; name from Modelmenu or Modelactive. Updates menu and Modelactive when saving over current or as new."""
		menu_val = self.ownerComp.par.Modelmenu.eval()
		activemodel = self.ownerComp.par.Modelactive.eval()
		menu_item_has_to_change = False
		if Path(menu_val).with_suffix('.joblib') != Path(activemodel).with_suffix('.joblib') or menu_val is None or menu_val == 'None':
			name = self.ownerComp.par.Modelactive.eval()
			menu_item_has_to_change = True
		else:
			name = Path(menu_val)
			menu_item_has_to_change = False

		if str(name) == 'internal stored':
			if menu_val == 'None':
				name = Path('default')
			else:
				name = Path(menu_val)
				menu_item_has_to_change = False

		folder = Path(self.ownerComp.par.Modelfolder.eval())
		path = folder / name
		self.Save(path)
		display_name = name
		if str(display_name).lower().endswith('.joblib'):
			display_name = str(display_name)[:-len('.joblib')]
		if menu_item_has_to_change:
			run(lambda: setattr(self.ownerComp.par, 'Modelmenu', display_name), delayFrames=5)

		infoname = Path(str(name)).with_suffix('.joblib')
		self.UpdateInfo(infoname)

	def SaveIncremental(self):
		"""Save to Modelfolder with an incremented filename (e.g. default_001.joblib) and set menu to the new name."""
		folder = Path(self.ownerComp.par.Modelfolder.eval())

		menu_val = self.ownerComp.par.Modelmenu.eval()
		base_name = 'default' if (menu_val is None or menu_val == 'None') else str(menu_val)

		base_path = folder / (base_name + '.joblib')
		path = hf.next_incremented_path(base_path, width=3)

		self.Save(path)

		display_name = path.name
		if display_name.lower().endswith('.joblib'):
			display_name = display_name[:-len('.joblib')]
		run(lambda: setattr(self.ownerComp.par, 'Modelmenu', display_name), delayFrames=5)
		self.UpdateInfo(display_name)

	def SaveAs(self, name='default'):
		"""Save to Modelfolder under the given name and set Modelmenu to it."""
		new_name = Path(name)
		folder = Path(self.ownerComp.par.Modelfolder.eval())
		path = folder / new_name
		self.Save(path)
		run(lambda: setattr(self.ownerComp.par, 'Modelmenu', name), delayFrames=5)
		self.UpdateInfo(name)

	def LoadFromMenu(self):
		"""Load model from path Modelfolder / Modelmenu."""
		name = Path(self.ownerComp.par.Modelmenu.eval())
		folder = Path(self.ownerComp.par.Modelfolder.eval())
		path = folder / name
		self.Load(path)

	def GetLoadPath(self):
		"""Returns Modelfolder / Loadfile as a Path, or None if Loadfile is empty."""
		folder = Path(self.ownerComp.par.Modelfolder.eval())
		filename = Path(self.ownerComp.par.Loadfile.eval()).with_suffix('.joblib')
		if filename == 'None' or filename == None:
			return None
		p = folder / filename
		return p

# Lifecycle ---------------------------------------------------------------
	# Cleanup on component destroy.

	def onDestroyTD(self):
		"""Clears thread reference, working flag, and cancel flag when the component is destroyed."""
		self.ThreadManager = None
		self.Working = False
		self._cancel_requested = False
