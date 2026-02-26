"""
Base extension class for neural-network-based TouchDesigner components.
Consolidates shared logic (storage, file I/O, metadata, menus, par callbacks,
threading, cancel support) so that subclasses only implement model-specific
training, saving, loading, and inference behaviour.

Designed for PyTorch-based models (VAE, GAN, Diffusion, Transformer, etc.).
LSTM and MLP retain their own dedicated base classes for .. reasons.

Subclasses MUST implement:
	_prepare_train_data()      - read tables into arrays, return data dict
	_read_train_params()       - read model params from component pars
	_build_and_train(data, params) - build & fit pipeline, return it
	_build_save_payload()      - return dict for dill serialization
	_load_from_payload(loaded) - reconstruct pipeline from loaded dict
	SetupNnPars()              - create custom par page

Optional hooks:
	_extra_stored_items()            - additional StorageManager items
	_build_extra_meta()              - extra metadata keys
	_on_train_complete(pipeline)     - after Pipeline set, before updateMeta
	_after_train()                   - after updateMeta, before UpdateInfo
	_on_load_complete()              - after Load succeeds
	_restore_state_from_pipeline()   - set stored properties from Pipeline
	_clear_subclass()                - reset subclass-specific state
	_on_pulse_extra(par)             - handle extra pulse parameters
	_on_value_change_extra(par, prev)- handle extra value-change parameters
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import dill
import sklearn
import torch

import helper_modules as hf


class nnbaseext:
	"""
	Abstract base for neural-network TouchDesigner extensions.
	Train() optionally runs in a background thread via ThreadManager.
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

		storedItems = [
			{'name': 'Pipeline', 'default': None, 'readOnly': False, 'property': True, 'dependable': True},
			{'name': 'Feat',     'default': None, 'readOnly': False, 'property': True, 'dependable': True},
		]
		storedItems.extend(self._extra_stored_items())
		self.stored = StorageManager(self, ownerComp, storedItems)

		self.x_table = self.ownerComp.op('x') if self.ownerComp.op('x') else None
		self.x_chan_names = self.ownerComp.op('x_chan_names') if self.ownerComp.op('x_chan_names') else None
		self.param_table = self.ownerComp.op('training_params_no_header') if self.ownerComp.op('training_params_no_header') else None
		self.menuSourceOp = self.ownerComp.op('model_menu') if self.ownerComp.op('model_menu') else None

		self.x = None

		self.updateMeta()
		self.UpdateMenu()

# Subclass hooks ---------------------------------------------------------

	def _extra_stored_items(self):
		"""Return a list of additional StorageManager item dicts."""
		return []

	def _prepare_train_data(self):
		"""
		Read training data from tables into numpy arrays.
		Store them on self (e.g. self.x) and return a dict that will be
		passed to _build_and_train(). Runs on the main thread.
		Raise ValueError to abort training with a message.
		"""
		raise NotImplementedError("Subclasses must implement _prepare_train_data()")

	def _read_train_params(self):
		"""
		Read model-specific training parameters from component pars or
		param table. Return a dict of parameter values.
		"""
		raise NotImplementedError("Subclasses must implement _read_train_params()")

	def _build_and_train(self, data, params):
		"""
		Build the pipeline/model and fit it on the provided data.
		May run in a WORKER THREAD -- must NOT access any TD objects.
		Returns the fitted pipeline object.
		"""
		raise NotImplementedError("Subclasses must implement _build_and_train()")

	def _build_save_payload(self):
		"""
		Return a dict to be serialized with dill for Save().
		The "meta" key is added automatically by the base Save().
		"""
		raise NotImplementedError("Subclasses must implement _build_save_payload()")

	def _load_from_payload(self, loaded):
		"""
		Reconstruct the pipeline from a loaded dill dict.
		Must set self.Pipeline and return the meta dict from the payload.
		"""
		raise NotImplementedError("Subclasses must implement _load_from_payload()")

	def SetupNnPars(self):
		"""Create model-specific training parameters on a custom page."""
		raise NotImplementedError("Subclasses must implement SetupNnPars()")

	def _build_extra_meta(self):
		"""Return a dict of extra keys to merge into meta during updateMeta()."""
		return {}

	def _on_train_complete(self, pipeline):
		"""Called after Pipeline is set but before updateMeta()."""
		pass

	def _after_train(self):
		"""Called after training and metadata update, before UpdateInfo."""
		pass

	def _on_load_complete(self):
		"""Called after Load() succeeds."""
		pass

	def _restore_state_from_pipeline(self):
		"""
		Set stored properties from the current Pipeline.
		Called by updateMeta() and Load(). Base sets Feat; override
		(with super call) to add subclass properties.
		"""
		try:
			if self.Pipeline is not None:
				self.Feat = int(self.Pipeline.n_features)
		except Exception:
			pass

	def _clear_subclass(self):
		"""Called at the end of Clear(). Override to reset subclass-specific state."""
		pass

	def _on_pulse_extra(self, par):
		"""Handle subclass-specific pulse parameters."""
		pass

	def _on_value_change_extra(self, par, prev):
		"""Handle subclass-specific value change parameters."""
		pass

# PAR Callbacks -----------------------------------------------------------

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
				self._on_pulse_extra(par)

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
			case _:
				self._on_value_change_extra(par, prev)

# Helpers -----------------------------------------------------------------

	def GetPage(self, pageName, create_if_missing=True):
		"""
		Return a custom Page on self.ownerComp with the given name.
		If it doesn't exist and create_if_missing=True, create it.
		Matching is case-insensitive against page.name and page.label.
		"""
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

	def _get_param_value(self, par_name, param_table_key, default_value, param_type='int'):
		"""Get parameter value from component parameter, with fallback to param table."""
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

# Core --------------------------------------------------------------------

	def Train(self):
		if self.Working:
			print('Training already in progress')
			return

		self._cancel_requested = False

		try:
			data = self._prepare_train_data()
		except Exception as e:
			print('Training aborted:', repr(e))
			return

		params = self._read_train_params()
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
				args=(data, params)
			)
			self.ThreadManager.EnqueueTask(task)
			self.Working = True
			print('Training enqueued (background thread)')
		else:
			try:
				pipeline = self._build_and_train(data, params)
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

	def _train_worker(self, data, params):
		"""Runs in worker thread. No TD access allowed."""
		self._train_result = self._build_and_train(data, params)

	def _train_success(self):
		"""Main thread callback after training completes."""
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
		"""Main thread callback if training raises."""
		self.Working = False
		if self._cancel_requested:
			self._cancel_requested = False
			print('Training cancelled by user')
		else:
			print('Training failed:', args)

	def CancelTrain(self):
		"""Request cancellation of a running threaded training."""
		if not self.Working:
			print('No training in progress to cancel')
			return
		self._cancel_requested = True
		print('Cancellation requested — training will stop after current epoch')

	def Partial_Fit(self):
		"""Partial fit with new data. Default: just retrain from scratch."""
		if self.Pipeline is None:
			print("No existing model to partial_fit / incremental train. Use Train instead.")
			return
		self.Train()

# File I/O ----------------------------------------------------------------

	def Save(self, name='default'):
		"""Save pipeline using dill serialization."""
		self.updateMeta()
		path = Path(str(name))
		if path.suffix.lower() != '.joblib':
			path = path.with_suffix('.joblib')
		path.parent.mkdir(parents=True, exist_ok=True)
		if self.Pipeline is None:
			print('Save aborted: no trained Pipeline.')
			return

		payload = self._build_save_payload()
		payload["meta"] = self.Meta or {}

		with open(path, 'wb') as f:
			dill.dump(payload, f)
		print('saved', path.name, 'to', str(path))
		self.UpdateMenu()

	def Load(self, name='default'):
		"""Load pipeline using dill deserialization."""
		if name in ('None', None):
			print('No model to load. Train or set Loadfromfile.')
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

		try:
			loaded_meta = self._load_from_payload(loaded)
		except Exception as e:
			print('Load failed: Could not reconstruct pipeline:', repr(e))
			import traceback
			traceback.print_exc()
			return

		self.Meta.clear()
		self.Meta.update(loaded_meta or {})

		self._restore_state_from_pipeline()
		self._on_load_complete()
		print('loaded', path.name, 'from', path)
		self.UpdateInfo(path.name)

	def SaveAs(self, name='default'):
		new_name = Path(name)
		folder = Path(self.ownerComp.par.Modelfolder.eval())
		path = folder / new_name
		self.Save(path)
		self.UpdateMenu()
		run(lambda: setattr(self.ownerComp.par, 'Modelmenu', name), delayFrames=5)
		self.UpdateInfo(name)

	def SaveOverride(self):
		menu_val = self.ownerComp.par.Modelmenu.eval()
		activemodel = self.ownerComp.par.Modelactive.eval()
		menu_item_has_to_change = False
		if Path(menu_val).with_suffix('.joblib') != Path(activemodel).with_suffix('.joblib') or menu_val is None or menu_val == 'None':
			name = self.ownerComp.par.Modelactive.eval()
			menu_item_has_to_change = True
		else:
			name = Path(menu_val)

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

	def LoadFromMenu(self):
		name = Path(self.ownerComp.par.Modelmenu.eval())
		folder = Path(self.ownerComp.par.Modelfolder.eval())
		path = folder / name
		self.Load(path)

	def GetLoadPath(self):
		folder = Path(self.ownerComp.par.Modelfolder.eval())
		filename = Path(self.ownerComp.par.Loadfile.eval()).with_suffix('.joblib')
		if filename == 'None' or filename is None:
			return None
		return folder / filename

# Updaters ----------------------------------------------------------------

	def UpdateInfo(self, modelactive):
		try:
			self.ownerComp.par.Modelactive = modelactive
		except Exception:
			pass

	def UpdateMenu(self):
		try:
			self.ownerComp.op('model_load_folder').par.refreshpulse.pulse()
			self.ownerComp.op('models_table').clear()
			if self.ownerComp.op('model_load_folder').numRows == 1:
				self.ownerComp.op('models_table').appendRow(['None'])
			else:
				for row in self.ownerComp.op('model_menu2').rows():
					self.ownerComp.op('models_table').appendRow(row)
		except Exception:
			pass

# Metadata ----------------------------------------------------------------

	def updateMeta(self):
		try:
			n_feat = int(getattr(self.Pipeline, 'n_features', None))
		except Exception:
			n_feat = int(self.x.shape[1]) if getattr(self, 'x', None) is not None else None

		x_names = []
		if self.x_chan_names is not None:
			try:
				x_names = [self.x_chan_names[0, c].val for c in range(self.x_chan_names.numCols)]
			except Exception:
				pass

		new_meta = {
			"created_utc": datetime.now(timezone.utc).isoformat(),
			"n_samples": int(self.x.shape[0]) if getattr(self, 'x', None) is not None else None,
			"n_features": n_feat,
			"params": self.last_params,
			"sklearn": sklearn.__version__,
			"numpy": np.__version__,
			"torch": torch.__version__,
			"op_path": self.ownerComp.path,
			"x_channel_names": list(x_names),
		}

		extra = self._build_extra_meta()
		if extra:
			new_meta.update(extra)

		self.Meta.clear()
		self.Meta.update(new_meta)
		self.Feat = n_feat
		self._restore_state_from_pipeline()

# Clear -------------------------------------------------------------------

	def Clear(self):
		"""Clear the stored model by resetting Pipeline, Meta, and Feat to defaults."""
		self.Pipeline = None
		self.Meta.clear()
		self.Feat = None
		self._clear_subclass()
		self.UpdateInfo('None')
		print('Model cleared: Pipeline and Meta reset to defaults')

# Lifecycle ---------------------------------------------------------------

	def onDestroyTD(self):
		self.ThreadManager = None
		self.Working = False
		self._cancel_requested = False
