"""
MLP Classifier extension. Uses mlpbaseext for storage, file I/O, metadata, and menu logic.
Supports single-label (multiclass) and multi-label; pipeline: StandardScaler -> MLPClassifier.
Multi-label uses MultiLabelBinarizer stored on pipeline as _mlb.
"""

from mlp_base_ext import mlpbaseext
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import helper_modules as hf


def _prob_positive_class(arr):
	"""Extracts single probability from array (last element, P(positive)); handles shapes (1,), (2,), (1,2)."""
	return float(np.asarray(arr).flat[-1])


class extmlpclassifier(mlpbaseext):
	"""MLP classifier extension; single-label and multi-label. Implements _read_y_data, _build_and_fit, Partial_Fit, Predict; uses self.Classes and optional _mlb on pipeline."""
	def __init__(self, ownerComp):
		super().__init__(ownerComp)

# Subclass hooks ----------------------------------------------------------

	def _init_from_pipeline(self):
		"""Restore self.Classes from pipeline (_mlb.classes_ for multilabel, else mlpclassifier.classes_)."""
		if self.Pipeline:
			mlb = getattr(self.Pipeline, "_mlb", None)
			if mlb is not None:
				self.Classes = list(mlb.classes_)
			else:
				self.Classes = self.Pipeline.named_steps["mlpclassifier"].classes_
		else:
			self.Classes = None

	def _is_multilabel(self):
		"""True when model is multilabel (trained with list-of-lists y; pipeline has _mlb)."""
		if self.Pipeline is not None:
			return getattr(self.Pipeline, "_mlb", None) is not None
		return bool(self.Meta.get("multilabel"))

	def _get_class_names_for_meta(self):
		"""Returns list of class/label names for Meta and scriptCHOP (one per output)."""
		return list(self.Classes) if self.Classes is not None else []

	def _build_extra_meta(self):
		"""Adds classes and, for multilabel, n_outputs and multilabel=True."""
		extra = {"classes": self._get_class_names_for_meta()}
		if self._is_multilabel() and self.Classes:
			extra["n_outputs"] = len(self.Classes)
			extra["multilabel"] = True
		return extra

	def _clear_subclass(self):
		"""Resets Classes and labelOut text."""
		self.Classes = None

	def _read_y_data(self):
		"""Y from y_table as 1D labels or list-of-lists for multilabel."""
		return hf._labels_or_multilabel_from_table(self.y_table)

	def _build_and_fit(self, x, y, w, params, use_w):
		"""Pipeline: StandardScaler -> MLPClassifier. Multilabel: y list-of-lists, fit MultiLabelBinarizer and attach as pipeline._mlb. Supports sample_weight when use_w."""
		# Multilabel: y is list of lists of label names -> MultiLabelBinarizer
		if isinstance(y, list) and y and isinstance(y[0], (list, set)):
			mlb = MultiLabelBinarizer()
			y_binary = mlb.fit_transform(y)
			clf = make_pipeline(StandardScaler(), MLPClassifier(**params))
			try:
				clf.fit(x, y_binary, mlpclassifier__sample_weight=w) if use_w and w is not None else clf.fit(x, y_binary)
			except TypeError:
				clf.fit(x, y_binary)
			setattr(clf, "_mlb", mlb)
			return clf
		# Single-label (multiclass): y is 1D array of label strings
		clf = make_pipeline(StandardScaler(), MLPClassifier(**params))
		try:
			clf.fit(x, y, mlpclassifier__sample_weight=w) if use_w and w is not None else clf.fit(x, y)
		except TypeError:
			clf.fit(x, y)
		return clf

	def _on_train_complete(self, pipeline):
		"""Sets self.Classes from pipeline (_mlb or mlpclassifier) and self.Nfeat from scaler."""
		mlb = getattr(pipeline, "_mlb", None)
		if mlb is not None:
			self.Classes = list(mlb.classes_)
		else:
			self.Classes = pipeline.named_steps["mlpclassifier"].classes_
		self.Nfeat = pipeline.named_steps["standardscaler"].scale_.shape[0]

# Model-specific ----------------------------------------------------------
	# Partial_Fit: incremental fit (multilabel not supported, delegates to Train). Predict: (class_names, values) per Predictmethod.

	def Partial_Fit(self):
		"""Incremental fit. If no pipeline or multilabel data/model, delegates to Train(). Otherwise partial_fit on scaler and MLPClassifier."""
		if self.Pipeline is None:
			return self.Train()

		X_new = hf._table_to_numpy(self.x_table)
		y_new = hf._labels_or_multilabel_from_table(self.y_table)
		# MLPClassifier.partial_fit accepts only 1D y; multilabel not supported
		if isinstance(y_new, list) and y_new and isinstance(y_new[0], (list, set)):
			print("Partial_Fit: multi-label not supported by partial_fit; running full Train().")
			return self.Train()
		if getattr(self.Pipeline, "_mlb", None) is not None:
			print("Partial_Fit: multi-label model; running full Train().")
			return self.Train()
		clf = self.Pipeline.named_steps["mlpclassifier"]

		use_w = self.ownerComp.par.Useweights.eval()
		if use_w and self.w_table is not None:
			self.w = hf._weights_from_table(self.w_table)
		else:
			self.w = None

		scaler = self.Pipeline.named_steps["standardscaler"]

		# X: partial_fit scaler or replace with new fit on exception
		try:
			scaler.partial_fit(X_new)
			X_scaled = scaler.transform(X_new)
		except Exception:
			new_scaler = StandardScaler().fit(X_new)
			self.Pipeline.set_params(standardscaler=new_scaler)
			X_scaled = new_scaler.transform(X_new)

		if hasattr(clf, "partial_fit"):
			try:
				clf.partial_fit(
					X_scaled, y_new,
					classes=getattr(clf, "classes_", np.unique(y_new)),
					sample_weight=self.w if self.w is not None else None
				)
			except TypeError:
				clf.partial_fit(
					X_scaled, y_new,
					classes=getattr(clf, "classes_", np.unique(y_new))
				)

		self.Classes = clf.classes_
		self.Nfeat = getattr(self.Pipeline.named_steps["standardscaler"], "scale_", np.zeros(X_new.shape[1])).shape[0]

		self.setMeta()
		self.UpdateInfo('internal stored')

	def Load(self, name='default'):
		"""Load then restore self.Classes from pipeline (_mlb or mlpclassifier)."""
		super().Load(name)
		if self.Pipeline is not None:
			mlb = getattr(self.Pipeline, "_mlb", None)
			self.Classes = list(mlb.classes_) if mlb is not None else self.Pipeline.named_steps["mlpclassifier"].classes_

	def Predict(self, arr=None):
		"""Returns (class_names, values). values: float32 array, one per class. Predictmethod: predict -> one-hot or 0/1 per class; predict_proba -> probability; predict_log_proba -> log-probability."""
		if self.Pipeline is None or self.Nfeat is None:
			print("No model found. Click TRAIN to create one.")
			return None
		if arr is None:
			return None
		method = self.ownerComp.par.Predictmethod.eval()

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

		scaler = self.Pipeline.named_steps["standardscaler"]
		clf = self.Pipeline.named_steps["mlpclassifier"]
		Xs = scaler.transform(vals.reshape(1, -1))

		if self._is_multilabel():
			# Multilabel: class_names from _mlb.classes_; one value per class (0/1 or probability)
			class_names = list(self.Classes)
			n_outputs = len(class_names)
			if method == "predict":
				pred = clf.predict(Xs)[0]
				out = pred.astype(np.float32)
			elif method == "predict_proba":
				proba_list = clf.predict_proba(Xs)
				out = np.array([_prob_positive_class(proba_list[i]) for i in range(n_outputs)], dtype=np.float32)
			elif method == "predict_log_proba":
				log_proba_list = clf.predict_log_proba(Xs)
				out = np.array([_prob_positive_class(log_proba_list[i]) for i in range(n_outputs)], dtype=np.float32)
			else:
				raise ValueError(f"Unknown Predictmethod: {method!r}")
		else:
			# Single-label (multiclass): one-hot or proba per class
			class_names = [str(c) for c in self.Classes]
			if method == "predict":
				pred_label = clf.predict(Xs)[0]
				out = np.zeros(len(self.Classes), dtype=np.float32)
				out[list(self.Classes).index(pred_label)] = 1.0
			elif method == "predict_proba":
				out = clf.predict_proba(Xs)[0].astype(np.float32)
			elif method == "predict_log_proba":
				out = clf.predict_log_proba(Xs)[0].astype(np.float32)
			else:
				raise ValueError(f"Unknown Predictmethod: {method!r}")

		return class_names, out
