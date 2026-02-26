"""
TouchDesigner extension: accessed as ext.extumapper from inside the component.
When promoted, capitalized attributes are callable from outside (e.g. Fit, Predict).
See wiki: Extensions.
"""


import umap

import joblib
import _helper_modules as hf
from sklearn.preprocessing import StandardScaler

from TDStoreTools import StorageManager
import TDFunctions as TDF
class extumapper:
	"""UMAP fit/transform plus persistence (Scaler, Reducer) and optional threaded fit."""
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp
		self.params = {}
		self.ThreadManager = op.TDResources.ThreadManager
		self.Working = False  # guards concurrent Fit
		self._fit_result = None  # (scaler, reducer, emb, n_components) from worker
		# StorageManager: Reducer and Scaler exposed as component storage
		storedItems = [
			{'name': 'Reducer', 'default': None, 'readOnly': False,
			 						'property': True, 'dependable': True},
			{'name': 'Scaler', 'default': None, 'readOnly': False,
			 						'property': True, 'dependable': True},
		]		
		self.stored = StorageManager(self, ownerComp, storedItems)

	def Fit(self, table=None, outOp=None):
		if self.Working:
			print('Fit already in progress')
			return
		table = table or self.ownerComp.op('null1')
		outOp = outOp or self.ownerComp.op('umap2D_DAT')

		X = hf._table_to_numpy(table)
		self.loadParams()
		n_components = int(self.ownerComp.par.Ncomponents.eval())

		if self.ownerComp.par.Traininthread.eval():
			# Background fit: show loading bar, enqueue worker
			self.ownerComp.par.opviewer = './loading_bar'
			self.ownerComp.op('loading_bar').par.Active = True
			task = self.ThreadManager.TDTask(
				target=self._fit_worker,
				SuccessHook=self._fit_success,
				ExceptHook=self._fit_except,
				args=(X, dict(self.params), n_components)
			)
			self.ThreadManager.EnqueueTask(task)
			self.Working = True
			print('UMAP fit enqueued (background thread)')
		else:
			# Main-thread fit
			scaler = StandardScaler()
			X_scaled = scaler.fit_transform(X)
			reducer = umap.UMAP(**self.params)
			emb = reducer.fit_transform(X_scaled)
			self.Scaler = scaler
			self.Reducer = reducer
			self.fill_components_dat(emb, n_components, 'umap2D_DAT')
			n_samples = X.shape[0]
			labels_op = self.ownerComp.op('labelsDAT')
			if labels_op and (labels_op.numRows == n_samples or labels_op.numRows == n_samples + 1):
				self.ownerComp.par.Generatecolor.pulse()
			print('UMAP fit completed (main thread)')

	def _fit_worker(self, X, params, n_components):
		"""Worker thread only. No TouchDesigner op/par access."""
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)
		reducer = umap.UMAP(**params)
		emb = reducer.fit_transform(X_scaled)
		self._fit_result = (scaler, reducer, emb, n_components)

	def _fit_success(self):
		"""Main-thread callback when background fit finishes. Applies result and clears Working."""
		scaler, reducer, emb, n_components = self._fit_result
		self._fit_result = None
		self.Scaler = scaler
		self.Reducer = reducer
		self.fill_components_dat(emb, n_components, 'umap2D_DAT')
		n_samples = emb.shape[0]
		labels_op = self.ownerComp.op('labelsDAT')
		if labels_op and (labels_op.numRows == n_samples or labels_op.numRows == n_samples + 1):
			self.ownerComp.par.Generatecolor.pulse()
		self.Working = False
		self.ownerComp.op('loading_bar').par.Active = False
		self.ownerComp.par.opviewer = './out1'
		print('UMAP fit completed successfully')

	def _fit_except(self, *args):
		"""Main-thread callback when background fit raises. Resets Working and logs."""
		self.Working = False
		print('UMAP fit failed:', args)


	def Predict(self, arr):
		"""Transform one sample (or first row) with stored Scaler and Reducer."""
		sample = arr[0,:]
		sample_scaled = self.Scaler.transform(sample.reshape(1, -1))
		return self.Reducer.transform(sample_scaled)[0]

	def loadParams(self):
		"""Refresh self.params from the component's params Table DAT (sklearn-style keys)."""
		paramDAT = self.ownerComp.op('params')
		self.params = hf._sk_param_table_to_dict(paramDAT)


	def fill_components_dat(self, emb, num, dat_op=None):
		"""Write embedding array to a Table DAT: header + one row per point. dat_op is name/path."""
		if num == 2:
			header = ['x', 'y']
		elif num == 3:
			header = ['x', 'y', 'z']
		else:
			# num > 3: x,y,z then c4..cN
			header = ['x', 'y', 'z'] + [f'c{i+1}' for i in range(3, num)]

		out = op(dat_op)
		if not out:
			raise RuntimeError(f"DAT '{dat_op}' not found")
		out.clear()
		out.appendRow(header)
		for point in emb:
			row = list(point)[:num]
			if len(row) < num:
				row += [''] * (num - len(row))
			out.appendRow(row)

	def Save(self, name='umap_default'):
		"""Persist Reducer and Scaler to name.joblib (no path; relative to project/script)."""
		joblib.dump({
			'reducer': self.Reducer,
			'scaler': self.Scaler,
		}, name + '.joblib')

	def Load(self, name='umap_default'):
		"""Load Reducer (and Scaler if present) from name.joblib. Supports legacy single-object dumps."""
		data = joblib.load(name + '.joblib')
		if isinstance(data, dict):
			self.Reducer = data['reducer']
			self.Scaler = data.get('scaler', None)
		else:
			self.Reducer = data
			self.Scaler = None

	def onDestroyTD(self):
		"""Clear thread ref and Working so callbacks don't touch torn-down state."""
		self.ThreadManager = None
		self.Working = False

