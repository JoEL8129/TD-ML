"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF
import json
import ast
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import joblib
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
#import asyncio
## custom
import helper_modules as hf

class extmlpclassifier():
	"""
	extmlpclassifier description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		self.last_params = None

		storedItems = [
			# Only 'name' is required...
			{'name': 'Pipeline', 'default': None, 'readOnly': False,
									 'property': True, 'dependable': True},
			{'name': 'Meta', 'default': {}, 'readOnly': False,
									 'property': True, 'dependable': True},									 									 								 						
		]

		# attributes:
		self.x_table = self.ownerComp.op('x') 
		self.y_table = self.ownerComp.op('y')
		self.w_table = self.ownerComp.op('weight')
		self.x_chan_names = self.ownerComp.op('x_chan_names')
		self.y_chan_names = self.ownerComp.op('y_chan_names')
		self.menuSourceOp = self.ownerComp.op('models_table')
		self.param_table = self.ownerComp.op('training_params_no_header')
		self.use_weights = self.ownerComp.par.Useweights.eval()


		self.x = None
		self.y = None
		self.w = None
		self.stored = StorageManager(self, ownerComp, storedItems)
		self.UpdateMenu()
		if self.Pipeline:
			self.Classes = self.Pipeline.named_steps["mlpclassifier"].classes_
			self.Nfeat = int(self.Pipeline.named_steps["standardscaler"].scale_.shape[0])
		else:
			self.Classes = None
			self.Nfeat = None

	def OnPulse(self, par):
		match par.name:
			case "Train" | "Train2":
				self.Train()
				#self.Runasync()
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

	def OnValueChange(self,par,prev):
		match par.name:
			case "Modelmenu":
				if prev!=None and prev!='None':
					self.LoadFromMenu()

			case "Modelfolder":
				self.UpdateMenu()

			case "Savecopyfile":
				self.Save(par.eval())

			case "Loadfromfile":
				self.Load(par.eval())

# Core

	def Save(self, name='default'):
		#self.updateMeta()
		path = Path(str(name))
		if path.suffix.lower() != '.joblib':
			path = path.with_suffix('.joblib')
		path.parent.mkdir(parents=True, exist_ok=True)
		if self.Pipeline is None:
			print('Save aborted: no trained Pipeline.')
			return
		payload = {"pipeline": self.Pipeline, "meta": self.Meta}
		joblib.dump(payload, path)  # consider compress=("xz", 3) if files are huge
		print('saved', path.name, 'to', str(path))
		self.UpdateMenu()

	def Load(self, name='default'):
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
		
		

		# Optionally infer Feat for UI/meta
		try:
			if self.Pipeline is not None:
				self.Nfeat = int(self.Pipeline.named_steps["standardscaler"].scale_.shape[0])
		except Exception:
			pass

		self.loadMeta(loaded_meta)
		#self.updateChanNames()
		print('loaded', path.name,'from',path)
		self.UpdateInfo(path.name)

	def Clear(self):
		"""
		Clear the stored model by resetting Pipeline and Meta to their defaults.
		Also resets dependent attributes (Classes, Nfeat).
		"""
		self.Pipeline = None
		self.Meta.clear()
		self.Classes = None
		self.Nfeat = None
		self.UpdateInfo('None')
		self.ownerComp.op("labelOut").text = 'No model loaded. Click TRAIN to create one.'
		print('Model cleared: Pipeline and Meta reset to defaults')


# Updaters

	def UpdateInfo(self,modelactive):
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

		
		#debug('update menu ')

	def setMeta(self):
		n_outputs = int(self.y.shape[1]) if self.y.ndim == 2 else 1
		
		x_names = [self.x_chan_names[0, c].val for c in range(self.x_chan_names.numCols)]
		y_names = [self.y_chan_names[0, c].val for c in range(self.y_chan_names.numCols)]
		# Build a plain dict, then copy it into the existing DependDict:
		new_meta = {
			"created_utc": datetime.now(timezone.utc).isoformat(),
			"n_samples": int(self.x.shape[0]) if getattr(self, 'x', None) is not None else None,
			"n_features": self.Nfeat,
			"n_outputs": n_outputs,
			"params": self.last_params,
			"sklearn": sklearn.__version__,
			"numpy": np.__version__,
			"op_path": self.ownerComp.path,
			"x_channel_names": list(x_names),
			"y_channel_names": list(y_names),
			"classes": list(self.Classes),
		}

		# Mutate in place to preserve the DependDict wrapper
		self.Meta.clear()
		self.Meta.update(new_meta)


	def loadMeta(self, new_meta=None):
		"""
		Merge/normalize meta loaded from disk with live info.
		- Prefer values from new_meta when provided.
		- Fill missing fields from the current pipeline/x/y/tables.
		- Preserve the DependDict wrapper in self.Meta.
		"""
		#print(new_meta)
		meta_in = new_meta or {}

		# ----- infer/fall back values
		# n_features: prefer meta value, else pipeline, else existing self.Nfeat
		n_feat = meta_in.get("n_features")
		if n_feat is None:
			try:
				n_feat = int(self.Pipeline.named_steps["standardscaler"].scale_.shape[0])
			except Exception:
				n_feat = getattr(self, "Nfeat", None)
		else:
			n_feat = int(n_feat)

		# n_outputs: prefer meta value, else infer from current y if present
		n_outputs = meta_in.get("n_outputs")
		if n_outputs is None:
			try:
				n_outputs = int(self.y.shape[1]) if (self.y is not None and self.y.ndim == 2) else (1 if self.y is not None else None)
			except Exception:
				n_outputs = None
		else:
			n_outputs = int(n_outputs)

		# n_samples: prefer meta value, else infer from current x if present
		n_samples = meta_in.get("n_samples")
		if n_samples is None:
			try:
				n_samples = int(self.x.shape[0]) if getattr(self, "x", None) is not None else None
			except Exception:
				n_samples = None
		else:
			n_samples = int(n_samples)

		# channel names: prefer meta; else take from UI tables; else empty list
		x_names = meta_in.get("x_channel_names")
		y_names = meta_in.get("y_channel_names")

		# params: prefer meta; else last_params
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

		# mutate in place to preserve DependDict wrapper

		self.Meta.clear()
		self.Meta.update(final)

		# keep convenience mirror
		if n_feat is not None:
			self.Nfeat = int(n_feat)







# FileIO High Level

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




	""" async def train_async(self):
		await asyncio.sleep(2)
		self.x = hf._table_to_numpy(self.x_table)
		self.y = hf._labels_from_table(self.y_table)

		use_w = self.ownerComp.par.Useweights.eval()
		self.w = hf._weights_from_table(self.w_table) if (use_w and self.w_table is not None) else None

		params = hf._sk_param_table_to_dict(self.param_table)
		print(params)
		self.last_params = dict(params)

		clf = make_pipeline(StandardScaler(), MLPClassifier(**params))
		try:
			clf.fit(self.x, self.y, mlpclassifier__sample_weight=self.w) if use_w is not None else clf.fit(self.x, self.y)
		except TypeError:
			clf.fit(self.x, self.y)

		self.Pipeline = clf
		self.Classes = clf.named_steps["mlpclassifier"].classes_
		self.Nfeat = clf.named_steps["standardscaler"].scale_.shape[0]

		self.setMeta()
		self.UpdateInfo('internal stored')

	def Runasync(self):
		# Run coroutine
		coroutines = [self.train_async()]
		op.TDAsyncIO.Run(coroutines) """




	def Train(self):
		self.x = hf._table_to_numpy(self.x_table)
		self.y = hf._labels_from_table(self.y_table)

		use_w = self.ownerComp.par.Useweights.eval()
		self.w = hf._weights_from_table(self.w_table) if (use_w and self.w_table is not None) else None

		params = hf._sk_param_table_to_dict(self.param_table)
		print(params)
		self.last_params = dict(params)

		clf = make_pipeline(StandardScaler(), MLPClassifier(**params))
		try:
			clf.fit(self.x, self.y, mlpclassifier__sample_weight=self.w) if use_w is not None else clf.fit(self.x, self.y)
		except TypeError:
			clf.fit(self.x, self.y)

		self.Pipeline = clf
		self.Classes = clf.named_steps["mlpclassifier"].classes_
		self.Nfeat = clf.named_steps["standardscaler"].scale_.shape[0]

		self.setMeta()
		self.UpdateInfo('internal stored')

	def Retrain(self):


		if self.Pipeline is None:
			# Backstop: if no model exists, behave like train() using default params DAT
			return self.Train()

		X_new = hf._table_to_numpy(self.x_table)
		y_new = hf._labels_from_table(self.y_table)
		use_w = self.ownerComp.par.Useweights.eval()

		if use_w and self.w_table is not None:
			self.w = hf._weights_from_table(self.w_table)
		else:
			self.w = None

		scaler = self.Pipeline.named_steps["standardscaler"]
		clf = self.Pipeline.named_steps["mlpclassifier"]

		try:
			scaler.partial_fit(X_new)
			X_scaled = scaler.transform(X_new)
		except Exception:
			new_scaler = StandardScaler().fit(X_new)
			#self.Pipeline.named_steps["standardscaler"] = new_scaler
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
				# fallback for estimators without sample_weight
				clf.partial_fit(
					X_scaled, y_new,
					classes=getattr(clf, "classes_", np.unique(y_new))
				)
		
		self.Classes = clf.classes_
		self.Nfeat = getattr(self.Pipeline.named_steps["standardscaler"], "scale_", np.zeros(X_new.shape[1])).shape[0]


		self.setMeta()
		self.UpdateInfo('internal stored')


	def Predict(self, arr = None):
		if self.Pipeline is None or self.Nfeat is None:
			print("No model found. Click TRAIN to create one.")
			return
		if arr is None:
			return
		method = self.ownerComp.par.Predictmethod.eval()

		#arr = scriptOp.inputs[0].numpyArray().astype(np.float32)
		if arr.ndim == 2 and arr.shape[1] == 1:
			vals = arr[:, 0]  # shape (channels,)
		elif arr.ndim == 2:
			vals = arr[:, 0]  # (channels, samples>1): first sample per channel
		elif arr.ndim == 1:
			vals = arr
		else:
			raise ValueError("Unexpected input shape: %s" % (arr.shape,))

		# 2) Pad/truncate to n_feat
		L = vals.shape[0]
		if L < self.Nfeat:
			padded = np.zeros((self.Nfeat,), dtype=np.float32)
			padded[:L] = vals
			vals = padded
		elif L > self.Nfeat:
			vals = vals[:self.Nfeat]

		# 3) Scale and predict
		scaler = self.Pipeline.named_steps["standardscaler"]
		clf = self.Pipeline.named_steps["mlpclassifier"]
		Xs = scaler.transform(vals.reshape(1, -1))

		if method == "predict":
			pred_label = clf.predict(Xs)[0]
			out_text = str(pred_label)
		elif method == "predict_proba":
			proba = clf.predict_proba(Xs)[0]
			best_i = int(np.argmax(proba))
			pred_label = self.Classes[best_i]
			out_text = "\n".join(f"{cls}: {p*100:.1f}%" for cls, p in zip(self.Classes, proba))

		elif method == "predict_log_proba":
			logp = clf.predict_log_proba(Xs)[0]
			best_i = int(np.argmax(logp))
			pred_label = self.Classes[best_i]
			out_text = "\n".join(f"{cls}: {v:.3f}" for cls, v in zip(self.Classes, logp))
		else:
			raise ValueError(f"Unknown Predictmethod: {method!r}")

		# 4) Write text & CHOP out
		try:
			td = self.ownerComp.op("labelOut")
			if td.text != out_text:
				td.text = out_text
		except Exception:
			pass

		
