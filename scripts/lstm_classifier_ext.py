"""
Skorch LSTM Classifier extension for TouchDesigner.
Works like your MLP example: stores a single Pipeline with metadata via joblib.

Expected tables/pars on the COMP (same as your MLP setup):
- DATs: x, y, weight (optional), x_chan_names, y_chan_names, training_params_no_header
- Parameters: Modelfolder, Loadfile, Modelmenu
- Helpers: model_menu DAT or custom UI for listing files
- Note: Weights are used automatically if a 'weight' table is present
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import ast

import numpy as np
import joblib
import dill  # Handles custom classes in TouchDesigner better than pickle/joblib
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---- NEW: skorch + torch
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

import helper_modules as hf  # your helper module (table to numpy, param parsing, etc.)

# ------------------------------
# PyTorch module for LSTM classification
# ------------------------------
class LSTMClassifierModule(nn.Module):
	def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.0, n_classes=2, bidirectional=False):
		super().__init__()
		self.lstm = nn.LSTM(
			input_size=n_features,
			hidden_size=hidden_size,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
			bidirectional=bidirectional
		)
		d = 2 if bidirectional else 1
		self.head = nn.Linear(hidden_size * d, n_classes)

	def forward(self, X):
		# X: (batch, seq, feat)
		y, (hn, cn) = self.lstm(X)
		last = y[:, -1, :]              # take last time step
		out = self.head(last)           # (batch, n_classes) - logits
		return out


# ------------------------------
# Small, pickle-friendly pipeline that mimics sklearn .fit/.predict
# It holds scalers + skorch net + seq_len + label encoder.
# ------------------------------
class SkorchLSTMPipeline:
	def __init__(self, n_features, n_classes=2, seq_len=16,
	             hidden_size=64, num_layers=1, dropout=0.0, bidirectional=False,
	             lr=1e-3, max_epochs=20, batch_size=64, device='cpu',
	             train_split=None, optimizer=torch.optim.Adam, criterion=nn.CrossEntropyLoss):
		self.n_features = int(n_features)
		self.n_classes = int(n_classes)
		self.seq_len = int(seq_len)

		# scalers (only X needs scaling for classification)
		self.x_scaler = StandardScaler()
		self.label_encoder = LabelEncoder()

		# skorch classifier
		self.net = NeuralNetClassifier(
			module=LSTMClassifierModule,
			module__n_features=self.n_features,
			module__hidden_size=hidden_size,
			module__num_layers=num_layers,
			module__dropout=dropout,
			module__n_classes=self.n_classes,
			module__bidirectional=bidirectional,
			optimizer=optimizer,
			lr=lr,
			max_epochs=max_epochs,
			batch_size=batch_size,
			criterion=criterion,
			device=device,
			train_split=train_split,   # e.g. None, or skorch.helper.PredefinedSplit
			# you can add callbacks, early stopping etc. here if you want
		)

	def _windowize(self, X2d):
		"""
		X2d: shape (N, F) -> returns X3d: (N - L + 1, L, F)
		"""
		N, F = X2d.shape
		L = self.seq_len
		if N < L:
			raise ValueError(f"Not enough samples to create windows: N={N}, seq_len={L}")
		out = np.lib.stride_tricks.sliding_window_view(X2d, (L, F)).reshape(-1, L, F)
		return out

	def _align_y(self, y2d):
		"""
		Match the last index of each window -> y[L-1:]
		y2d: (N, O) -> (N - L + 1, O)
		"""
		L = self.seq_len
		return y2d[L-1:, :]

	def _windowize_segmented(self, X2d, seq_ids):
		"""
		X2d: (N, F) features (already scaled)
		seq_ids: (N,) integers indicating gesture membership
		Returns X3d: (M, L, F) built from sliding windows that do NOT cross seq boundaries.
		"""
		L = self.seq_len
		N, F = X2d.shape
		out = []
		# find boundaries where seq_id changes
		# segments are ranges [start, end) with constant id
		starts = [0]
		for i in range(1, N):
			if seq_ids[i] != seq_ids[i-1]:
				starts.append(i)
		starts.append(N)

		for a, b in zip(starts[:-1], starts[1:]):
			seg_len = b - a
			if seg_len < L:
				continue
			# build windows inside [a,b)
			seg = X2d[a:b]
			# classic sliding windows per segment
			view = np.lib.stride_tricks.sliding_window_view(seg, (L, F)).reshape(-1, L, F)
			out.append(view)
		if not out:
			raise ValueError("No valid windows found; check seq_len or segment sizes.")
		return np.concatenate(out, axis=0)

	def fit(self, X, y, sample_weight=None):
		X = np.asarray(X, dtype=np.float32)
		# y is labels (strings/objects) - encode to integers
		if isinstance(y, np.ndarray) and y.dtype == object:
			y_encoded = self.label_encoder.fit_transform(y)
		else:
			# Already numeric, but ensure it's 1D
			y_encoded = np.asarray(y, dtype=np.int64).ravel()
			if not hasattr(self.label_encoder, 'classes_'):
				# Fit encoder even if already numeric
				self.label_encoder.fit(y_encoded)

		# assume last column of X is seq_id; split it off before scaling
		seq_ids = X[:, -1].astype(np.int64)
		X_feat = X[:, :-1]

		Xs = self.x_scaler.fit_transform(X_feat)

		X3 = self._windowize_segmented(Xs, seq_ids)
		# align Y: take the last index in each window
		# we need matching target rows for each produced window
		Y_collect = []
		L = self.seq_len

		# rebuild the same segmentation to align Y
		N = len(seq_ids)
		starts = [0]
		for i in range(1, N):
			if seq_ids[i] != seq_ids[i-1]:
				starts.append(i)
		starts.append(N)
		for a, b in zip(starts[:-1], starts[1:]):
			if (b - a) >= L:
				Y_collect.append(y_encoded[a+L-1:b])  # y at the last step of each window
		Y1 = np.concatenate(Y_collect, axis=0)

		Xt = torch.from_numpy(X3.astype(np.float32))
		Yt = torch.from_numpy(Y1.astype(np.int64))

		self.net.initialize()
		self.net.fit(Xt, Yt)
		return self

	def predict(self, X):
		X = np.asarray(X, dtype=np.float32)
		if X.ndim == 3:
			X3 = X
		elif X.ndim == 2:
			# assume last column is seq_id
			seq_ids = X[:, -1].astype(np.int64)
			X_feat = X[:, :-1]
			Xs = self.x_scaler.transform(X_feat)
			X3 = self._windowize_segmented(Xs, seq_ids)
		else:
			raise ValueError(f"Unexpected X shape for predict: {X.shape}")
		Xt = torch.from_numpy(X3.astype(np.float32))
		with torch.no_grad():
			yhat = self.net.predict(Xt)  # returns class indices
		# Convert back to original labels
		return self.label_encoder.inverse_transform(yhat)

	def predict_proba(self, X):
		X = np.asarray(X, dtype=np.float32)
		if X.ndim == 3:
			X3 = X
		elif X.ndim == 2:
			# assume last column is seq_id
			seq_ids = X[:, -1].astype(np.int64)
			X_feat = X[:, :-1]
			Xs = self.x_scaler.transform(X_feat)
			X3 = self._windowize_segmented(Xs, seq_ids)
		else:
			raise ValueError(f"Unexpected X shape for predict_proba: {X.shape}")
		Xt = torch.from_numpy(X3.astype(np.float32))
		with torch.no_grad():
			proba = self.net.predict_proba(Xt)  # returns probabilities
		return proba

	def predict_window(self, window2d):
		"""
		window2d: (L, F) single window for realtime.
		Returns class label (string).
		"""
		L, F = window2d.shape
		if L != self.seq_len or F != self.n_features:
			raise ValueError(f"Bad window shape {window2d.shape}; expected ({self.seq_len},{self.n_features})")
		# scale per feature, keep shape
		ws = self.x_scaler.transform(window2d)
		yhat = self.predict(ws[np.newaxis, ...])  # (1,) -> single label
		return yhat[0]

	def predict_proba_window(self, window2d):
		"""
		window2d: (L, F) single window for realtime.
		Returns (n_classes,) probabilities.
		"""
		L, F = window2d.shape
		if L != self.seq_len or F != self.n_features:
			raise ValueError(f"Bad window shape {window2d.shape}; expected ({self.seq_len},{self.n_features})")
		# scale per feature, keep shape
		ws = self.x_scaler.transform(window2d)
		proba = self.predict_proba(ws[np.newaxis, ...])  # (1, n_classes)
		return proba[0].astype(np.float32)

	@property
	def classes_(self):
		"""Return the class labels (original strings/values)."""
		return self.label_encoder.classes_


class extskorchlstmclassifier:
	"""
	TouchDesigner extension that trains/saves/loads a Skorch LSTM classifier pipeline.
	"""
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp

		self.Meta = {}
		self.last_params = None

		storedItems = [
			{'name': 'Pipeline', 'default': None, 'readOnly': False, 'property': True, 'dependable': True},
			{'name': 'Feat',     'default': None, 'readOnly': False, 'property': True, 'dependable': True},
			{'name': 'SeqLen',   'default': None, 'readOnly': False, 'property': True, 'dependable': True},
			{'name': 'Classes',  'default': None, 'readOnly': False, 'property': True, 'dependable': True},
		]
		self.stored = StorageManager(self, ownerComp, storedItems)

		# TouchDesigner nodes / params (same names as your MLP comp)
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

	# -------------- Callbacks --------------

	def OnPulse(self, par):
		match par.name:
			case "Train" | "Train2":
				self.Train()
			case "Retrain" | "Retrain2":
				self.Retrain()				

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
				if prev is not None or prev != 'None':
					self.LoadFromMenu()
			case "Modelfolder":
				self.UpdateMenu()
			case "Savecopyfile":
				self.Save(par.eval())
			case "Loadfromfile":
				self.Load(par.eval())

	# -------------- Helpers --------------
	def GetPage(self, pageName, create_if_missing=True):
		"""
		Return a custom Page on self.ownerComp with the given name.
		If it doesn't exist and create_if_missing=True, create it.

		Matching is case-insensitive against page.name and page.label.
		Returns the Page, or None if not found and create_if_missing=False.
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

	def SetupNnPars(self):
		"""
		Create neural network training parameters on a custom page.
		"""
		page = self.GetPage('LSTM Training')
		pars = [
			ParTemplate(
				'Seqlen',
				par_type='int',
				label='Sequence Length',
				default=16,
				min=1,
				max=1000,
				callback=False
			),
			ParTemplate(
				'Hiddensize',
				par_type='int',
				label='Hidden Size',
				default=64,
				min=1,
				max=10000,
				callback=False
			),
			ParTemplate(
				'Numlayers',
				par_type='int',
				label='Number of Layers',
				default=1,
				min=1,
				max=10,
				callback=False
			),
			ParTemplate(
				'Dropout',
				par_type='float',
				label='Dropout',
				default=0.0,
				min=0.0,
				max=1.0,
				norm_min=0.0,
				norm_max=1.0,
				callback=False
			),
			ParTemplate(
				'Bidirectional',
				par_type='toggle',
				label='Bidirectional',
				default=False,
				callback=False
			),
			ParTemplate(
				'Maxepochs',
				par_type='int',
				label='Max Epochs',
				default=20,
				min=1,
				max=10000,
				callback=False
			),
			ParTemplate(
				'Lr',
				par_type='float',
				label='Learning Rate',
				default=0.001,
				min=1e-6,
				max=1.0,
				norm_min=0.0,
				norm_max=1.0,
				callback=False
			),
			ParTemplate(
				'Batchsize',
				par_type='int',
				label='Batch Size',
				default=64,
				min=1,
				max=10000,
				callback=False
			),
			ParTemplate(
				'Device',
				par_type='menu',
				label='Device',
				menu_names=['cpu', 'cuda'],
				menu_labels=['CPU', 'CUDA'],
				default='cpu',
				callback=False
			),
		]

		for p in pars:
			p.createPar(page)

	def UpdateInfo(self, modelactive):
		self.ownerComp.par.Modelactive = modelactive

	def UpdateMenu(self):
		self.ownerComp.op('model_load_folder').par.refreshpulse.pulse()
		self.ownerComp.op('models_table').clear()
		if self.ownerComp.op('model_load_folder').numRows ==1:
			rown = ['None']
			self.ownerComp.op('models_table').appendRow(rown)
		else:
			for row in self.ownerComp.op('model_menu2').rows():
				self.ownerComp.op('models_table').appendRow(row)

	def _get_param_value(self, par_name, param_table_key, default_value, param_type='int'):
		"""
		Get parameter value from component parameter, with fallback to param table.
		
		Args:
			par_name: Name of the component parameter
			param_table_key: Key to look up in param table
			default_value: Default value if neither source is available
			param_type: Type conversion ('int', 'float', 'bool', 'str')
		
		Returns:
			Parameter value converted to the specified type
		"""
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
		except:
			# Fallback to param table
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
			except:
				pass
			return default_value

	def Save(self, name='default'):
		"""
		Save pipeline using skorch's recommended approach:
		- Save net parameters using net.save_params() (handles state_dict properly)
		- Save scalers and pipeline config separately
		- This avoids pickling issues with TD references
		"""
		self.updateMeta()
		path = Path(str(name))
		if path.suffix.lower() != '.joblib':
			path = path.with_suffix('.joblib')
		path.parent.mkdir(parents=True, exist_ok=True)
		if self.Pipeline is None:
			print('Save aborted: no trained Pipeline.')
			return
		
		import tempfile
		import copy
		
		# Save net parameters using skorch's built-in method (to temp file)
		net = self.Pipeline.net
		with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
			tmp_path = tmp_file.name
		
		try:
			# Use skorch's save_params - saves only the model state, not the whole object
			# This creates a file with module_state_dict and potentially optimizer_state_dict
			net.save_params(f_params=tmp_path)
			
			# Read the saved state dict (skorch save_params creates a dict with 'module_state_dict')
			saved_state = torch.load(tmp_path, map_location='cpu')
			# Extract just the module state dict (what we need for loading)
			net_state_dict = saved_state.get('module_state_dict', saved_state)
			
			# Get net hyperparameters (excluding 'module' and callbacks)
			# IMPORTANT: get_params() may include 'module', so we must exclude it explicitly
			net_params = net.get_params()
			clean_net_params = {}
			for k, v in net_params.items():
				# Explicitly skip 'module' (we pass it separately) and callback-related params
				if k == 'module' or k.startswith('callback') or k == 'history':
					continue
				clean_net_params[k] = v
			
			# Double-check 'module' is not in the dict
			if 'module' in clean_net_params:
				del clean_net_params['module']
			
			# Build payload with components
			payload = {
				"meta": self.Meta or {},
				"pipeline_config": {
					"n_features": self.Pipeline.n_features,
					"n_classes": self.Pipeline.n_classes,
					"seq_len": self.Pipeline.seq_len,
				},
				"scalers": {
					"x_scaler": copy.deepcopy(self.Pipeline.x_scaler),
					"label_encoder": copy.deepcopy(self.Pipeline.label_encoder),
				},
				"net_params": clean_net_params,
				"net_state_dict": net_state_dict,
			}
			
			# Save using dill
			with open(path, 'wb') as f:
				dill.dump(payload, f)
			print('saved', path.name, 'to', str(path))
		finally:
			# Clean up temp file
			try:
				os.unlink(tmp_path)
			except:
				pass
		
		self.UpdateMenu()


	def SaveOverride(self):
		menu_val = self.ownerComp.par.Modelmenu.eval()
		activemodel = self.ownerComp.par.Modelactive.eval()
		menu_item_has_to_change = False
		if Path(menu_val).with_suffix('.joblib') != Path(activemodel).with_suffix('.joblib') or menu_val is None or menu_val == 'None':
			name = self.ownerComp.par.Modelactive.eval() 
			menu_item_has_to_change = True
		else:
			name = Path(menu_val)
			menu_item_has_to_change = False

		if str(name)=='internal stored':
			if menu_val=='None':
				name=Path('default')
			else:
				name = Path(menu_val)
				menu_item_has_to_change = False
		

		folder = Path(self.ownerComp.par.Modelfolder.eval())
		path = folder / name
		self.Save(path)
		display_name = name
		if str(display_name).lower().endswith('.joblib'):
			display_name = str(display_name)[:-len('.joblib')]  # keep dots in stem intact
		if menu_item_has_to_change:
			run(lambda: setattr(self.ownerComp.par,'Modelmenu',display_name), delayFrames=5)

		infoname = Path(str(name)).with_suffix('.joblib')
		self.UpdateInfo(infoname)

	def SaveIncremental(self):
		folder = Path(self.ownerComp.par.Modelfolder.eval())

		# menu shows name WITHOUT suffix; default if empty/None
		menu_val = self.ownerComp.par.Modelmenu.eval()
		base_name = 'default' if (menu_val is None or menu_val == 'None') else str(menu_val)

		# Build path IN THE TARGET FOLDER and increment the trailing number
		base_path = folder / (base_name + '.joblib')
		path = hf.next_incremented_path(base_path, width=3)

		# Save
		self.Save(path)

		display_name = path.name
		if display_name.lower().endswith('.joblib'):
			display_name = display_name[:-len('.joblib')]  # keep dots in stem intact
		run(lambda: setattr(self.ownerComp.par,'Modelmenu',display_name), delayFrames=5)
		self.UpdateInfo(display_name)

	def SaveAs(self, name='default'):
		new_name = Path(name)
		folder = Path(self.ownerComp.par.Modelfolder.eval())
		path = folder / new_name
		self.Save(path)
		run(lambda: setattr(self.ownerComp.par,'Modelmenu',name), delayFrames=5)
		self.UpdateInfo(name)

	def LoadFromMenu(self):
		name = Path(self.ownerComp.par.Modelmenu.eval())
		folder = Path(self.ownerComp.par.Modelfolder.eval())
		path = folder / name
		self.Load(path)
		
	def GetLoadPath(self):

		folder = Path(self.ownerComp.par.Modelfolder.eval())
		filename = Path(self.ownerComp.par.Loadfile.eval()).with_suffix('.joblib')  # ensures .joblib
		if filename == 'None' or filename==None:
			return None
		p = folder / filename
		return p




	def Load(self, name='default'):
		"""
		Load pipeline using skorch's recommended approach:
		- Reconstruct net from saved params and state_dict
		- Restore scalers and pipeline config
		"""
		if name in ('None', None):
			print('No Model Found to Load, Train? Or adjust Loadfolder')
			return
		path = Path(str(name)).with_suffix('.joblib')
		if not path.exists():
			print('Load failed: file not found ->', str(path))
			return
		try:
			# Use dill to load
			with open(path, 'rb') as f:
				loaded = dill.load(f)
		except Exception as e:
			print('Load failed:', repr(e))
			import traceback
			traceback.print_exc()
			return

		# Check if this is new format (components) or old format (full pipeline)
		if "pipeline" in loaded:
			# Old format - direct pipeline (may have issues, but try)
			self.Pipeline = loaded.get("pipeline", None)
			loaded_meta = loaded.get("meta", {}) or {}
		else:
			# New format - reconstruct from components
			try:
				config = loaded.get("pipeline_config", {})
				scalers = loaded.get("scalers", {})
				net_params = loaded.get("net_params", {})
				net_state_dict = loaded.get("net_state_dict")
				
				if not config or not scalers:
					raise ValueError("Missing required components in saved model")
				
				# Reconstruct pipeline
				pipeline = SkorchLSTMPipeline.__new__(SkorchLSTMPipeline)
				pipeline.n_features = config["n_features"]
				pipeline.n_classes = config["n_classes"]
				pipeline.seq_len = config["seq_len"]
				pipeline.x_scaler = scalers["x_scaler"]
				pipeline.label_encoder = scalers["label_encoder"]
				
				# Reconstruct net - ensure 'module' is not in net_params
				# (it might be in old saved files)
				load_net_params = {k: v for k, v in net_params.items() if k != 'module'}
				
				# Reconstruct net
				pipeline.net = NeuralNetClassifier(
					module=LSTMClassifierModule,
					**load_net_params
				)
				
				# Initialize and load state
				pipeline.net.initialize()
				if net_state_dict is not None:
					# Load the state dict manually
					if isinstance(net_state_dict, dict) and 'module_state_dict' in net_state_dict:
						pipeline.net.module_.load_state_dict(net_state_dict['module_state_dict'])
					else:
						# If it's the raw state dict from save_params
						pipeline.net.module_.load_state_dict(net_state_dict)
				
				self.Pipeline = pipeline
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
				self.Classes = self.Pipeline.classes_
		except Exception:
			pass

		print('loaded', path.name, 'from', path)
		self.UpdateInfo(path.name)

	def Clear(self):
		"""
		Clear the stored model by resetting Pipeline, Meta, Feat, SeqLen, and Classes to their defaults.
		"""
		self.Pipeline = None
		self.Meta.clear()
		self.Feat = None
		self.SeqLen = None
		self.Classes = None
		self.UpdateInfo('None')
		print('Model cleared: Pipeline, Meta, Feat, SeqLen, and Classes reset to defaults')

	# -------------- Training --------------
	def Train(self):
		self.x = hf._table_to_numpy(self.x_table)   # shape (N, F)
		self.y = hf._labels_from_table(self.y_table)   # shape (N,) - labels as strings/objects
		# Use weights if weight table exists and has data
		if self.w_table is not None and self.w_table.numRows > 0:
			try:
				self.w = hf._weights_from_table(self.w_table)
			except Exception:
				self.w = None
		else:
			self.w = None

		# Get parameters from component parameters (with fallback to param table for backwards compatibility)
		seq_len      = self._get_param_value('Seqlen', 'seq_len', 16, 'int')
		hidden_size  = self._get_param_value('Hiddensize', 'hidden_size', 64, 'int')
		num_layers   = self._get_param_value('Numlayers', 'num_layers', 1, 'int')
		dropout      = self._get_param_value('Dropout', 'dropout', 0.0, 'float')
		bidirectional= self._get_param_value('Bidirectional', 'bidirectional', False, 'bool')
		max_epochs   = self._get_param_value('Maxepochs', 'max_epochs', 20, 'int')
		lr           = self._get_param_value('Lr', 'lr', 1e-3, 'float')
		batch_size   = self._get_param_value('Batchsize', 'batch_size', 64, 'int')
		device       = self._get_param_value('Device', 'device', 'cpu', 'str').lower()
		
		device = 'cuda' if (device in ('cuda', 'gpu') and torch.cuda.is_available()) else 'cpu'
		
		# Determine number of classes from unique labels
		unique_labels = np.unique(self.y)
		n_classes = len(unique_labels)
		
		# Store params for metadata
		self.last_params = {
			'seq_len': seq_len,
			'hidden_size': hidden_size,
			'num_layers': num_layers,
			'dropout': dropout,
			'bidirectional': bidirectional,
			'max_epochs': max_epochs,
			'lr': lr,
			'batch_size': batch_size,
			'device': device,
			'n_classes': n_classes
		}

		# Last column of X is seq_id, so subtract 1 from feature count
		n_features = int(self.x.shape[1]) - 1

		clf = SkorchLSTMPipeline(
			n_features=n_features,
			n_classes=n_classes,
			seq_len=seq_len,
			hidden_size=hidden_size,
			num_layers=num_layers,
			dropout=dropout,
			bidirectional=bidirectional,
			lr=lr,
			max_epochs=max_epochs,
			batch_size=batch_size,
			device=device,
		)

		try:
			# (Optional) sample weights not wired in by default; see note in class.
			clf.fit(self.x, self.y)
		except Exception as e:
			print('Training failed:', repr(e))
			import traceback
			traceback.print_exc()
			return

		self.Pipeline = clf
		self.Classes = clf.classes_
		self.updateMeta()
		self.UpdateInfo('Stored In COMP / not saved')

	def Retrain(self):
		"""Retrain with new data - not fully implemented for LSTM yet."""
		if self.Pipeline is None:
			print("No existing model to retrain. Use Train instead.")
			return
		# For now, just call Train
		self.Train()

	def updateMeta(self):
		try:
			n_feat = int(getattr(self.Pipeline, 'n_features', None))
		except Exception:
			# Last column of X is seq_id, so subtract 1 from feature count
			n_feat = int(self.x.shape[1]) - 1 if getattr(self, 'x', None) is not None else None

		n_classes = len(self.Classes) if getattr(self, 'Classes', None) is not None else None
		seq_len = int(getattr(self.Pipeline, 'seq_len', 16)) if getattr(self, 'Pipeline', None) is not None else 16

		x_names = [self.x_chan_names[0, c].val for c in range(self.x_chan_names.numCols)]
		y_names = [self.y_chan_names[0, c].val for c in range(self.y_chan_names.numCols)]

		new_meta = {
			"created_utc": datetime.now(timezone.utc).isoformat(),
			"kind": "train_lstm_skorch_classifier",
			"n_samples": int(self.x.shape[0]) if getattr(self, 'x', None) is not None else None,
			"n_features": n_feat,
			"n_classes": n_classes,
			"seq_len": seq_len,
			"params": self.last_params,
			"sklearn": sklearn.__version__,
			"numpy": np.__version__,
			"torch": torch.__version__,
			"op_path": self.ownerComp.path,
			"x_channel_names": list(x_names),
			"y_channel_names": list(y_names),
			"classes": list(self.Classes) if getattr(self, 'Classes', None) is not None else [],
		}
		self.Meta.clear()
		self.Meta.update(new_meta)
		self.Feat = n_feat
		self.SeqLen = seq_len
