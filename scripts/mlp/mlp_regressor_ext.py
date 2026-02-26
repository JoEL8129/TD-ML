"""
MLP Regressor extension. Uses mlpbaseext for storage, file I/O, metadata, and menu logic.
Implements regression (continuous y); pipeline: StandardScaler -> TransformedTargetRegressor(MLPRegressor, StandardScaler).
"""

from mlp_base_ext import mlpbaseext
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
import helper_modules as hf


class extmlpregressor(mlpbaseext):
	"""MLP regressor extension; implements _read_y_data, _build_and_fit, Partial_Fit, Predict for continuous targets."""
	_train_status_label = 'Stored In COMP'

	def __init__(self, ownerComp):
		super().__init__(ownerComp)

# Subclass hooks ----------------------------------------------------------

	def _read_y_data(self):
		"""Y table as numeric array (regression: continuous values)."""
		return hf._table_to_numpy(self.y_table)

	def _build_and_fit(self, x, y, w, params, use_w):
		"""Pipeline: StandardScaler -> TransformedTargetRegressor(MLPRegressor, StandardScaler). Supports sample_weight when use_w."""
		reg = MLPRegressor(**params)
		ttr = TransformedTargetRegressor(
			regressor=reg,
			transformer=StandardScaler(),
			check_inverse=False,
		)
		clf = make_pipeline(StandardScaler(), ttr)

		try:
			if use_w and w is not None:
				clf.fit(x, y, transformedtargetregressor__sample_weight=w)
			else:
				clf.fit(x, y)
		except TypeError:
			clf.fit(x, y)
		return clf

# Model-specific ----------------------------------------------------------
	# Partial_Fit: incremental update; Predict: one sample, padded/trimmed to Nfeat.

	def Partial_Fit(self):
		"""Incremental fit. If no pipeline, delegates to Train(). Otherwise partial_fit on X-scaler, y-transformer, and regressor (fallback to full fit)."""
		if self.Pipeline is None:
			print('usual train')
			return self.Train()

		self.x = hf._table_to_numpy(self.x_table)
		self.y = hf._table_to_numpy(self.y_table)
		use_w = self.ownerComp.par.Useweights.eval()
		self.w = hf._weights_from_table(self.w_table) if (use_w and self.w_table is not None) else None
		scaler_X = self.Pipeline.named_steps["standardscaler"]
		ttr = self.Pipeline.named_steps["transformedtargetregressor"]
		reg = ttr.regressor if hasattr(ttr, "regressor") else getattr(ttr, "regressor_", None)

		# X: partial_fit scaler or replace with new fit on exception
		try:
			scaler_X.partial_fit(self.x)
			Xs = scaler_X.transform(self.x)
		except Exception:
			print('exception normal fit')
			new_scaler = StandardScaler().fit(self.x)
			self.Pipeline.named_steps["standardscaler"] = new_scaler
			Xs = new_scaler.transform(self.x)

		# y: partial_fit transformer if available, else fit new StandardScaler
		try:
			yt = getattr(ttr, "transformer_", ttr.transformer)
			if hasattr(yt, "partial_fit"):
				yt.partial_fit(self.y)
				ys = yt.transform(self.y)
			else:
				yt = StandardScaler().fit(self.y)
				ttr.transformer_ = yt
				ys = yt.transform(self.y)
		except Exception:
			yt = StandardScaler().fit(self.y)
			ttr.transformer_ = yt
			ys = yt.transform(self.y)

		# Regressor: partial_fit if supported, else full fit
		did_partial = False
		if hasattr(reg, "partial_fit"):
			try:
				if self.w is not None:
					reg.partial_fit(Xs, ys, sample_weight=self.w)
				else:
					reg.partial_fit(Xs, ys)
				did_partial = True
			except TypeError:
				pass

		if not did_partial:
			try:
				if self.w is not None:
					reg.fit(Xs, ys, sample_weight=self.w)
				else:
					reg.fit(Xs, ys)
			except TypeError:
				reg.fit(Xs, ys)

		try:
			self.last_params = dict(reg.get_params(deep=False)) if hasattr(reg, "get_params") else self.last_params
		except Exception:
			pass

		self.setMeta()
		self.UpdateInfo('Stored In COMP')

	def Predict(self, arr=None):
		"""Single sample: arr 1D or 2D (first column used). Padded with zeros or trimmed to Nfeat; returns float32 scalar or 1D array."""
		if self.Pipeline is None or self.Nfeat is None:
			print("No model found. Click TRAIN to create one.")
			return

		if arr.ndim == 2 and arr.shape[1] == 1:
			vals = arr[:, 0]
		elif arr.ndim == 2:
			vals = arr[:, 0]
		elif arr.ndim == 1:
			vals = arr
		else:
			raise ValueError("Unexpected input shape: %s" % (arr.shape,))

		# Pad or trim to Nfeat for pipeline input
		L = vals.shape[0]
		if L < self.Nfeat:
			padded = np.zeros((self.Nfeat,), dtype=np.float32)
			padded[:L] = vals
			vals = padded
		elif L > self.Nfeat:
			vals = vals[:self.Nfeat]

		return self.Pipeline.predict(vals.reshape(1, -1))[0].astype(np.float32)
