"""
Base extension for MLP-based TouchDesigner components.
Shared logic for storage, file I/O, metadata, menus, and par callbacks;
subclasses implement model-specific Train, Partial_Fit, and Predict.
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import joblib
import sklearn
import helper_modules as hf


class mlpbaseext:
	"""
	Abstract base for MLP extensions. Train() can run in a background thread via ThreadManager.
	Required overrides: _read_y_data(), _build_and_fit(), Partial_Fit(), Predict().
	Optional hooks: _init_from_pipeline(), _build_extra_meta(), _clear_subclass(), _on_train_complete().
	"""

	_train_status_label = 'internal stored'

	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.ThreadManager = op.TDResources.ThreadManager
		self.Working = False
		self._train_result = None
		self.last_params = None

		# Stored on the component (DependDict); Pipeline = fitted sklearn pipeline, Meta = run metadata
		storedItems = [
			{'name': 'Pipeline', 'default': None, 'readOnly': False,
									 'property': True, 'dependable': True},
			{'name': 'Meta', 'default': {}, 'readOnly': False,
									 'property': True, 'dependable': True},
		]

		# Table and menu ops from the component layout
		self.x_table = self.ownerComp.op('x')
		self.y_table = self.ownerComp.op('y')
		self.w_table = self.ownerComp.op('weight')
		self.x_chan_names = self.ownerComp.op('x_chan_names')
		self.y_chan_names = self.ownerComp.op('y_chan_names')
		self.menuSourceOp = self.ownerComp.op('models_table')
		self.param_table = self.ownerComp.op('training_params_no_header')

		self.x = None
		self.y = None
		self.w = None

		self.stored = StorageManager(self, ownerComp, storedItems)
		self.UpdateMenu()

		# Feature count from pipeline when one is already loaded
		if self.Pipeline:
			self.Nfeat = int(self.Pipeline.named_steps["standardscaler"].scale_.shape[0])
		else:
			self.Nfeat = None

		self._init_from_pipeline()

# Subclass hooks ---------------------------------------------------------

	def _init_from_pipeline(self):
		"""Invoked at end of __init__. Override to restore subclass state from stored pipeline."""
		pass

	def _build_extra_meta(self):
		"""Returns a dict of extra keys merged into meta in setMeta(). Default: {}."""
		return {}

	def _clear_subclass(self):
		"""Invoked at end of Clear(). Override to reset subclass-specific state."""
		pass

	def _read_y_data(self):
		"""Returns y data from self.y_table. Override for model type (labels vs numeric)."""
		raise NotImplementedError("Subclasses must implement _read_y_data()")

	def _build_and_fit(self, x, y, w, params, use_w):
		"""Builds and fits the sklearn pipeline. Runs in worker thread; must not access TD ops/pars. Returns fitted pipeline."""
		raise NotImplementedError("Subclasses must implement _build_and_fit()")

	def _on_train_complete(self, pipeline):
		"""Invoked on main thread after training. Override to set subclass state from the new pipeline."""
		pass

# PAR Callbacks -----------------------------------------------------------
	# Pulse/ValueChange handlers for Train, Save, Load, Clear, menu, etc.

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

			case _:
				print('none')

	def OnValueChange(self, par, prev):
		match par.name:
			case "Modelmenu":
				if prev != None and prev != 'None':
					self.LoadFromMenu()

			case "Modelfolder":
				self.UpdateMenu()

			case "Savecopyfile":
				print('savecopyfile')
				#self.Save(par.eval())

			case "Loadfromfile":
				self.Load(par.eval())

# Core --------------------------------------------------------------------

	def Save(self, name='default'):
		"""Serialize Pipeline and Meta to a .joblib file; creates parent dirs if needed."""
		path = Path(str(name))
		if path.suffix.lower() != '.joblib':
			path = path.with_suffix('.joblib')
		path.parent.mkdir(parents=True, exist_ok=True)
		if self.Pipeline is None:
			print('Save aborted: no trained Pipeline.')
			return
		payload = {"pipeline": self.Pipeline, "meta": self.Meta}
		joblib.dump(payload, path)
		print('saved', path.name, 'to', str(path))
		self.UpdateMenu()

	def Load(self, name='default'):
		"""Load pipeline and meta from .joblib; updates Nfeat and info."""
		if name in ('None', None):
			print('No Model Found to Load, Train? Or adjust Loadfolder')
			return

		path = Path(str(name)).with_suffix('.joblib')
		if not path.exists():
			print('Load failed: file not found ->', str(path))
			return
		try:
			loaded = joblib.load(path)
		except Exception as e:
			print('Load failed:', repr(e))
			return

		self.Pipeline = loaded.get("pipeline", None)
		loaded_meta = loaded.get("meta", {}) or {}
		self.loadMeta(loaded_meta)

		try:
			if self.Pipeline is not None:
				self.Nfeat = int(self.Pipeline.named_steps["standardscaler"].scale_.shape[0])
		except Exception:
			pass

		self.loadMeta(loaded_meta)
		print('loaded', path.name, 'from', path)
		self.UpdateInfo(path.name)

	def Clear(self):
		"""Resets stored model: Pipeline and Meta to defaults, Nfeat cleared; then _clear_subclass() for subclass cleanup."""
		self.Pipeline = None
		self.Meta.clear()
		self.Nfeat = None
		self.UpdateInfo('None')
		self._clear_subclass()
		print('Model cleared: Pipeline and Meta reset to defaults')

	def Train(self):
		"""Reads x/y/w from tables, builds and fits pipeline. If Traininthread: enqueues worker and uses loading_bar; else runs on main thread."""
		if self.Working:
			print('Training already in progress')
			return

		self.x = hf._table_to_numpy(self.x_table)
		self.y = self._read_y_data()

		use_w = self.ownerComp.par.Useweights.eval()
		self.w = hf._weights_from_table(self.w_table) if (use_w and self.w_table is not None) else None

		params = hf._sk_param_table_to_dict(self.param_table)
		print(params)
		self.last_params = dict(params)

		if self.ownerComp.par.Traininthread.eval():
			self.ownerComp.par.opviewer = './loading_bar'
			self.ownerComp.op('loading_bar').par.Active = True
			task = self.ThreadManager.TDTask(
				target=self._train_worker,
				SuccessHook=self._train_success,
				ExceptHook=self._train_except,
				args=(self.x, self.y, self.w, params, use_w)
			)
			self.ThreadManager.EnqueueTask(task)
			self.Working = True
			print('Training enqueued (background thread)')
		else:
			pipeline = self._build_and_fit(self.x, self.y, self.w, params, use_w)
			self.Pipeline = pipeline
			self._on_train_complete(pipeline)
			self.setMeta()
			self.UpdateInfo(self._train_status_label)
			self.ownerComp.par.opviewer = './out1' 
			print('Training completed (main thread)')

	def _train_worker(self, x, y, w, params, use_w):
		"""Runs in worker thread; must not access TouchDesigner ops or parameters."""
		self._train_result = self._build_and_fit(x, y, w, params, use_w)




	def _train_success(self):
		"""Main-thread callback when background training completes successfully."""
		pipeline = self._train_result
		self._train_result = None
		self.Pipeline = pipeline
		self._on_train_complete(pipeline)
		
		self.ownerComp.op('loading_bar').par.Active = False
		self.ownerComp.par.opviewer = './out1'
		self.setMeta()
		self.UpdateInfo(self._train_status_label)
		self.Working = False
		print('Training completed successfully')

	def _train_except(self, *args):
		"""Main-thread callback when background training raises an exception."""
		self.Working = False
		print('Training failed:', args)

	def Partial_Fit(self):
		raise NotImplementedError("Subclasses must implement Partial_Fit()")

	def Predict(self, arr=None):
		raise NotImplementedError("Subclasses must implement Predict()")

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
	# Meta: created_utc, n_samples, n_features, n_outputs, params, versions, channel names. Subclasses can add keys via _build_extra_meta().

	def setMeta(self):
		"""Builds meta from current pipeline, x/y shapes, channel names, and _build_extra_meta(); writes to self.Meta and sets self.Nfeat."""
		try:
			n_outputs = int(self.y.shape[1]) if getattr(self.y, "ndim", None) == 2 else 1
		except (AttributeError, TypeError):
			n_outputs = None  # subclass can set via _build_extra_meta (e.g. multilabel)
		n_feat = int(self.Pipeline.named_steps["standardscaler"].scale_.shape[0])

		x_names = [self.x_chan_names[0, c].val for c in range(self.x_chan_names.numCols)]
		y_names = [self.y_chan_names[0, c].val for c in range(self.y_chan_names.numCols)]

		new_meta = {
			"created_utc": datetime.now(timezone.utc).isoformat(),
			"n_samples": int(self.x.shape[0]) if getattr(self, 'x', None) is not None else None,
			"n_features": n_feat,
			"n_outputs": n_outputs,
			"params": self.last_params,
			"sklearn": sklearn.__version__,
			"numpy": np.__version__,
			"op_path": self.ownerComp.path,
			"x_channel_names": list(x_names),
			"y_channel_names": list(y_names),
		}

		extra = self._build_extra_meta()
		if extra:
			new_meta.update(extra)
		if new_meta.get("n_outputs") is None:
			new_meta["n_outputs"] = 1

		self.Meta.clear()
		self.Meta.update(new_meta)

		self.Nfeat = n_feat

	def loadMeta(self, new_meta=None):
		"""Merges meta from disk with live pipeline/table info. Uses new_meta when provided; fills gaps from pipeline/x/y. Preserves self.Meta (DependDict)."""
		meta_in = new_meta or {}

		n_feat = meta_in.get("n_features")
		if n_feat is None:
			try:
				n_feat = int(self.Pipeline.named_steps["standardscaler"].scale_.shape[0])
			except Exception:
				n_feat = getattr(self, "Nfeat", None)
		else:
			n_feat = int(n_feat)

		n_outputs = meta_in.get("n_outputs")
		if n_outputs is None:
			try:
				n_outputs = int(self.y.shape[1]) if (self.y is not None and self.y.ndim == 2) else (1 if self.y is not None else None)
			except Exception:
				n_outputs = None
		else:
			n_outputs = int(n_outputs)

		n_samples = meta_in.get("n_samples")
		if n_samples is None:
			try:
				n_samples = int(self.x.shape[0]) if getattr(self, "x", None) is not None else None
			except Exception:
				n_samples = None
		else:
			n_samples = int(n_samples)

		x_names = meta_in.get("x_channel_names")
		y_names = meta_in.get("y_channel_names")

		params = meta_in.get("params", getattr(self, "last_params", None))

		final = {
			"created_utc": meta_in.get("created_utc") or datetime.now(timezone.utc).isoformat(),
			"n_samples": n_samples,
			"n_features": n_feat,
			"n_outputs": n_outputs,
			"params": params,
			"sklearn": meta_in.get("sklearn", getattr(sklearn, "__version__", None)),
			"numpy": meta_in.get("numpy", getattr(np, "__version__", None)),
			"op_path": meta_in.get("op_path", getattr(self.ownerComp, "path", None)),
			"x_channel_names": list(x_names) if x_names else [],
			"y_channel_names": list(y_names) if y_names else [],
		}
		# Classifier-specific keys from saved meta (classes, multilabel)
		for key in ("classes", "multilabel"):
			if key in meta_in:
				final[key] = meta_in[key]

		self.Meta.clear()
		self.Meta.update(final)

		if n_feat is not None:
			self.Nfeat = int(n_feat)

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
		"""Clears thread reference and working flag when the component is destroyed."""
		self.ThreadManager = None
		self.Working = False
