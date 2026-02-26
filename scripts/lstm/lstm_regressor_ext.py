"""
LSTM Regressor extension. Uses lstmbaseext for storage, file I/O, metadata, and menu logic.
Pipeline: SkorchLSTMPipeline (x_scaler, y_scaler, skorch NeuralNetRegressor); segmented sliding windows, last column = seq_id.
"""

from lstm_base_ext import lstmbaseext
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor

import helper_modules as hf


# PyTorch LSTM + linear head for regression.
class LSTMRegressorModule(nn.Module):
	"""LSTM(batch_first=True) + Linear -> n_outputs. forward(X): (batch, seq, feat) -> (batch, n_outputs)."""
	def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.0, n_outputs=1, bidirectional=False):
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
		self.head = nn.Linear(hidden_size * d, n_outputs)

	def forward(self, X):
		# X: (batch, seq, feat) -> last time step -> (batch, n_outputs)
		y, (hn, cn) = self.lstm(X)
		last = y[:, -1, :]
		out = self.head(last)
		return out


# Pipeline: x_scaler, y_scaler, skorch net, seq_len. fit/predict over segmented sequences (last col = seq_id); pickle-friendly.
class SkorchLSTMPipeline:
	def __init__(self, n_features, n_outputs=1, seq_len=16,
	             hidden_size=64, num_layers=1, dropout=0.0, bidirectional=False,
	             lr=1e-3, max_epochs=20, batch_size=64, device='cpu',
	             train_split=None, optimizer=torch.optim.Adam, criterion=nn.MSELoss):
		self.n_features = int(n_features)
		self.n_outputs = int(n_outputs)
		self.seq_len = int(seq_len)

		self.x_scaler = StandardScaler()
		self.y_scaler = StandardScaler()

		self.net = NeuralNetRegressor(
			module=LSTMRegressorModule,
			module__n_features=self.n_features,
			module__hidden_size=hidden_size,
			module__num_layers=num_layers,
			module__dropout=dropout,
			module__n_outputs=self.n_outputs,
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
		"""Targets aligned to last index of each window: y[seq_len-1:]."""
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
		"""Last column of X = seq_id; features = X[:, :-1]. y scaled with y_scaler; segmented windowing then skorch fit."""
		X = np.asarray(X, dtype=np.float32)
		y = np.asarray(y, dtype=np.float32)
		if y.ndim == 1:
			y = y.reshape(-1, 1)

		seq_ids = X[:, -1].astype(np.int64)
		X_feat = X[:, :-1]

		Xs = self.x_scaler.fit_transform(X_feat)
		ys = self.y_scaler.fit_transform(y)

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
				Y_collect.append(ys[a+L-1:b])
		Y2 = np.concatenate(Y_collect, axis=0)

		Xt = torch.from_numpy(X3.astype(np.float32))
		Yt = torch.from_numpy(Y2.astype(np.float32))

		self.net.initialize()
		self.net.fit(Xt, Yt)
		return self

	def predict(self, X):
		"""X: (N, F+1) with last col seq_id, or (M, L, F) already windowed. Returns inverse-transformed y (unscaled)."""
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
			yhat = self.net.module_(Xt).cpu().numpy()
		return self.y_scaler.inverse_transform(yhat)

	def predict_window(self, window2d):
		"""Single window (seq_len, n_features). Returns (n_outputs,) float32."""
		L, F = window2d.shape
		if L != self.seq_len or F != self.n_features:
			raise ValueError(f"Bad window shape {window2d.shape}; expected ({self.seq_len},{self.n_features})")
		ws = self.x_scaler.transform(window2d)
		yhat = self.predict(ws[np.newaxis, ...])
		return yhat[0].astype(np.float32)


class extskorchlstm(lstmbaseext):
	"""LSTM regressor extension; implements base hooks for SkorchLSTMPipeline (segmented windows, x/y scalers)."""
	def __init__(self, ownerComp):
		super().__init__(ownerComp)

# Subclass hooks ----------------------------------------------------------

	def _read_y_data(self):
		"""Y table as numeric array for regression."""
		return hf._table_to_numpy(self.y_table)

	def _build_pipeline(self, n_features, params):
		"""SkorchLSTMPipeline with n_outputs from self.y shape."""
		n_outputs = int(self.y.shape[1]) if self.y.ndim == 2 else 1
		return SkorchLSTMPipeline(
			n_features=n_features,
			n_outputs=n_outputs,
			**params,
		)

	def _save_pipeline_config(self):
		"""Config dict for reconstruction: n_features, n_outputs, seq_len."""
		return {
			"n_features": self.Pipeline.n_features,
			"n_outputs": self.Pipeline.n_outputs,
			"seq_len": self.Pipeline.seq_len,
		}

	def _save_scalers(self):
		"""Deep-copied x_scaler and y_scaler for serialization."""
		return {
			"x_scaler": copy.deepcopy(self.Pipeline.x_scaler),
			"y_scaler": copy.deepcopy(self.Pipeline.y_scaler),
		}

	def _reconstruct_pipeline(self, config, scalers, net_params, net_state_dict):
		"""Reconstructs pipeline without fit: set attributes, rebuild net, load_state_dict."""
		pipeline = SkorchLSTMPipeline.__new__(SkorchLSTMPipeline)
		pipeline.n_features = config["n_features"]
		pipeline.n_outputs = config["n_outputs"]
		pipeline.seq_len = config["seq_len"]
		pipeline.x_scaler = scalers["x_scaler"]
		pipeline.y_scaler = scalers["y_scaler"]

		# Rebuild net from saved params (exclude 'module'); then load state_dict
		load_net_params = {k: v for k, v in net_params.items() if k != 'module'}

		pipeline.net = NeuralNetRegressor(
			module=LSTMRegressorModule,
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
		"""Adds kind and n_outputs to meta."""
		n_outputs = int(self.y.shape[1]) if getattr(self, 'y', None) is not None and self.y.ndim == 2 else 1
		return {
			"kind": "train_lstm_skorch",
			"n_outputs": n_outputs,
		}

	def _after_train(self):
		"""Writes y_channel_names from Meta to table1 for display."""
		self.WriteChanNames()

	def _on_load_complete(self):
		"""Writes y_channel_names from Meta to table1 after load."""
		self.WriteChanNames()

# Regressor-specific ------------------------------------------------------
	# WriteChanNames: sync output channel names to table1 from Meta.

	def WriteChanNames(self):
		"""Populates table1 with Meta.y_channel_names (one column per output)."""
		meta = getattr(self, 'Meta', None) or {}
		self.ownerComp.op('table1').clear()
		for item in meta.get('y_channel_names', []):
			self.ownerComp.op('table1').appendCol(item)
