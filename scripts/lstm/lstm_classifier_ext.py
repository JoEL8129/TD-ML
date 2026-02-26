"""
LSTM Classifier extension. Uses lstmbaseext for storage, file I/O, metadata, and menu logic.
Pipeline: SkorchLSTMPipeline (x_scaler, label_encoder, skorch NeuralNetClassifier); segmented sliding windows, last column = seq_id.
"""

from lstm_base_ext import lstmbaseext
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from skorch import NeuralNetClassifier

import helper_modules as hf


# PyTorch LSTM + linear head for classification (logits).
class LSTMClassifierModule(nn.Module):
	"""LSTM(batch_first=True) + Linear -> n_classes logits. forward(X): (batch, seq, feat) -> (batch, n_classes)."""
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
		# X: (batch, seq, feat) -> last time step -> logits (batch, n_classes)
		y, (hn, cn) = self.lstm(X)
		last = y[:, -1, :]
		out = self.head(last)
		return out


# Pipeline: x_scaler, label_encoder, skorch net, seq_len. fit/predict over segmented sequences (last col = seq_id); pickle-friendly.
class SkorchLSTMPipeline:
	def __init__(self, n_features, n_classes=2, seq_len=16,
	             hidden_size=64, num_layers=1, dropout=0.0, bidirectional=False,
	             lr=1e-3, max_epochs=20, batch_size=64, device='cpu',
	             train_split=None, optimizer=torch.optim.Adam, criterion=nn.CrossEntropyLoss):
		self.n_features = int(n_features)
		self.n_classes = int(n_classes)
		self.seq_len = int(seq_len)

		self.x_scaler = StandardScaler()
		self.label_encoder = LabelEncoder()

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
			train_split=train_split,
		)

	def _windowize(self, X2d):
		"""(N, F) -> (N - seq_len + 1, seq_len, F) sliding windows."""
		N, F = X2d.shape
		L = self.seq_len
		if N < L:
			raise ValueError(f"Not enough samples to create windows: N={N}, seq_len={L}")
		out = np.lib.stride_tricks.sliding_window_view(X2d, (L, F)).reshape(-1, L, F)
		return out

	def _align_y(self, y2d):
		"""Labels aligned to last index of each window: y[seq_len-1:]."""
		L = self.seq_len
		return y2d[L-1:, :]

	def _windowize_segmented(self, X2d, seq_ids):
		"""(N, F) + (N,) seq_ids -> (M, seq_len, F) sliding windows that do not cross segment boundaries."""
		L = self.seq_len
		N, F = X2d.shape
		out = []
		starts = [0]
		for i in range(1, N):
			if seq_ids[i] != seq_ids[i-1]:
				starts.append(i)
		starts.append(N)

		for a, b in zip(starts[:-1], starts[1:]):
			seg_len = b - a
			if seg_len < L:
				continue
			seg = X2d[a:b]
			view = np.lib.stride_tricks.sliding_window_view(seg, (L, F)).reshape(-1, L, F)
			out.append(view)
		if not out:
			raise ValueError("No valid windows found; check seq_len or segment sizes.")
		return np.concatenate(out, axis=0)

	def fit(self, X, y, sample_weight=None):
		"""Last column of X = seq_id; features = X[:, :-1]. Segmented windowing then skorch fit."""
		X = np.asarray(X, dtype=np.float32)
		if isinstance(y, np.ndarray) and y.dtype == object:
			y_encoded = self.label_encoder.fit_transform(y)
		else:
			y_encoded = np.asarray(y, dtype=np.int64).ravel()
			if not hasattr(self.label_encoder, 'classes_'):
				self.label_encoder.fit(y_encoded)

		seq_ids = X[:, -1].astype(np.int64)
		X_feat = X[:, :-1]

		Xs = self.x_scaler.fit_transform(X_feat)

		X3 = self._windowize_segmented(Xs, seq_ids)
		# Align y to last index of each window per segment
		Y_collect = []
		L = self.seq_len

		N = len(seq_ids)
		starts = [0]
		for i in range(1, N):
			if seq_ids[i] != seq_ids[i-1]:
				starts.append(i)
		starts.append(N)
		for a, b in zip(starts[:-1], starts[1:]):
			if (b - a) >= L:
				Y_collect.append(y_encoded[a+L-1:b])
		Y1 = np.concatenate(Y_collect, axis=0)

		Xt = torch.from_numpy(X3.astype(np.float32))
		Yt = torch.from_numpy(Y1.astype(np.int64))

		self.net.initialize()
		self.net.fit(Xt, Yt)
		return self

	def predict(self, X):
		"""X: (N, F+1) with last col seq_id, or (M, L, F) already windowed. Returns class labels (strings)."""
		X = np.asarray(X, dtype=np.float32)
		if X.ndim == 3:
			X3 = X
		elif X.ndim == 2:
			seq_ids = X[:, -1].astype(np.int64)
			X_feat = X[:, :-1]
			Xs = self.x_scaler.transform(X_feat)
			X3 = self._windowize_segmented(Xs, seq_ids)
		else:
			raise ValueError(f"Unexpected X shape for predict: {X.shape}")
		Xt = torch.from_numpy(X3.astype(np.float32))
		self.net.module_.eval()
		device = next(self.net.module_.parameters()).device
		Xt = Xt.to(device)
		with torch.no_grad():
			logits = self.net.module_(Xt)
			yhat = torch.argmax(logits, dim=1).cpu().numpy()
		return self.label_encoder.inverse_transform(yhat)

	def predict_proba(self, X):
		"""Same input convention as predict. Returns (n_windows, n_classes) float32."""
		X = np.asarray(X, dtype=np.float32)
		if X.ndim == 3:
			X3 = X
		elif X.ndim == 2:
			seq_ids = X[:, -1].astype(np.int64)
			X_feat = X[:, :-1]
			Xs = self.x_scaler.transform(X_feat)
			X3 = self._windowize_segmented(Xs, seq_ids)
		else:
			raise ValueError(f"Unexpected X shape for predict_proba: {X.shape}")
		Xt = torch.from_numpy(X3.astype(np.float32))
		self.net.module_.eval()
		device = next(self.net.module_.parameters()).device
		Xt = Xt.to(device)
		with torch.no_grad():
			logits = self.net.module_(Xt)
			proba = torch.softmax(logits, dim=1).cpu().numpy()
		return proba

	def predict_window(self, window2d):
		"""Single window (seq_len, n_features). Returns class label (string)."""
		L, F = window2d.shape
		if L != self.seq_len or F != self.n_features:
			raise ValueError(f"Bad window shape {window2d.shape}; expected ({self.seq_len},{self.n_features})")
		ws = self.x_scaler.transform(window2d)
		yhat = self.predict(ws[np.newaxis, ...])
		return yhat[0]

	def predict_proba_window(self, window2d):
		"""Single window (seq_len, n_features). Returns (n_classes,) float32 probabilities."""
		L, F = window2d.shape
		if L != self.seq_len or F != self.n_features:
			raise ValueError(f"Bad window shape {window2d.shape}; expected ({self.seq_len},{self.n_features})")
		ws = self.x_scaler.transform(window2d)
		proba = self.predict_proba(ws[np.newaxis, ...])
		return proba[0].astype(np.float32)

	@property
	def classes_(self):
		"""Returns class labels (original strings/values) from label_encoder."""
		return self.label_encoder.classes_


class extskorchlstmclassifier(lstmbaseext):
	"""LSTM classifier extension; implements base hooks for SkorchLSTMPipeline (segmented windows, label encoder, Classes stored)."""
	def __init__(self, ownerComp):
		super().__init__(ownerComp)

# Subclass hooks ----------------------------------------------------------

	def _extra_stored_items(self):
		"""Stored Classes (label list) for UI and Predict."""
		return [
			{'name': 'Classes', 'default': None, 'readOnly': False, 'property': True, 'dependable': True},
		]

	def _read_y_data(self):
		"""Y table as 1D labels (strings) for classification."""
		return hf._labels_from_table(self.y_table)

	def _build_pipeline(self, n_features, params):
		"""SkorchLSTMPipeline with n_classes from unique(self.y)."""
		unique_labels = np.unique(self.y)
		n_classes = len(unique_labels)
		self.last_params['n_classes'] = n_classes

		return SkorchLSTMPipeline(
			n_features=n_features,
			n_classes=n_classes,
			**params,
		)

	def _save_pipeline_config(self):
		"""Config dict for reconstruction: n_features, n_classes, seq_len."""
		return {
			"n_features": self.Pipeline.n_features,
			"n_classes": self.Pipeline.n_classes,
			"seq_len": self.Pipeline.seq_len,
		}

	def _save_scalers(self):
		"""Deep-copied x_scaler and label_encoder for serialization."""
		return {
			"x_scaler": copy.deepcopy(self.Pipeline.x_scaler),
			"label_encoder": copy.deepcopy(self.Pipeline.label_encoder),
		}

	def _reconstruct_pipeline(self, config, scalers, net_params, net_state_dict):
		"""Reconstructs pipeline without fit: set attributes, rebuild net, load_state_dict."""
		pipeline = SkorchLSTMPipeline.__new__(SkorchLSTMPipeline)
		pipeline.n_features = config["n_features"]
		pipeline.n_classes = config["n_classes"]
		pipeline.seq_len = config["seq_len"]
		pipeline.x_scaler = scalers["x_scaler"]
		pipeline.label_encoder = scalers["label_encoder"]

		# Rebuild net from saved params (exclude 'module'); then load state_dict
		load_net_params = {k: v for k, v in net_params.items() if k != 'module'}

		pipeline.net = NeuralNetClassifier(
			module=LSTMClassifierModule,
			**load_net_params
		)

		pipeline.net.initialize()
		if net_state_dict is not None:
			if isinstance(net_state_dict, dict) and 'module_state_dict' in net_state_dict:
				pipeline.net.module_.load_state_dict(net_state_dict['module_state_dict'])
			else:
				pipeline.net.module_.load_state_dict(net_state_dict)

		return pipeline

	def _build_extra_meta(self):
		"""Adds kind, n_classes, classes to meta."""
		n_classes = len(self.Classes) if getattr(self, 'Classes', None) is not None else None
		return {
			"kind": "train_lstm_skorch_classifier",
			"n_classes": n_classes,
			"classes": list(self.Classes) if getattr(self, 'Classes', None) is not None else [],
		}

	def _on_train_complete(self, pipeline):
		"""Stores Classes from pipeline.classes_."""
		self.Classes = pipeline.classes_

	def _on_load_complete(self):
		"""Restores Classes from loaded pipeline."""
		try:
			self.Classes = self.Pipeline.classes_
		except Exception:
			pass

	def _clear_subclass(self):
		"""Resets Classes to None."""
		self.Classes = None

# Model-specific ----------------------------------------------------------
	# Predict: single window buf -> (class_names, values). Predictmethod: one-hot, proba, or log_proba.

	def Predict(self, buf=None):
		"""Returns (class_names, values). values: float32, one per class. Predictmethod: predict -> one-hot; predict_proba -> probability; predict_log_proba -> log-probability."""
		if self.Pipeline is None or self.Feat is None or self.Classes is None:
			return None
		if buf is None:
			return None

		method = self.ownerComp.par.Predictmethod.eval()
		class_names = [str(c) for c in self.Classes]

		if method == "predict":
			pred_label = self.Pipeline.predict_window(buf)
			out = np.zeros(len(self.Classes), dtype=np.float32)
			out[list(self.Classes).index(pred_label)] = 1.0
		elif method == "predict_proba":
			out = self.Pipeline.predict_proba_window(buf)
		elif method == "predict_log_proba":
			proba = self.Pipeline.predict_proba_window(buf)
			out = np.log(proba + 1e-12).astype(np.float32)
		else:
			pred_label = self.Pipeline.predict_window(buf)
			out = np.zeros(len(self.Classes), dtype=np.float32)
			out[list(self.Classes).index(pred_label)] = 1.0

		return class_names, out
