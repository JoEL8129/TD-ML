"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""




import umap
from umap import UMAP

import pandas as pd
import ast
import joblib
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from TDStoreTools import StorageManager
import TDFunctions as TDF
class extumapper:
	"""
	umap description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		# properties
		TDF.createProperty(self, 'MyProperty', value=0, dependable=True,
						   readOnly=False)

		# attributes:
		self.a = 0 # attribute
		self.B = 1 # promoted attribute
		self.params = {}
		# stored items (persistent across saves and re-initialization):
		storedItems = [
			# Only 'name' is required...
			{'name': 'Reducer', 'default': None, 'readOnly': False,
			 						'property': True, 'dependable': True},
		]
		# Uncomment the line below to store StoredProperty. To clear stored
		# 	items, use the Storage section of the Component Editor
		
		self.stored = StorageManager(self, ownerComp, storedItems)

	def myFunction(self, v):
		debug(v)

	def PromotedFunction(self, v):
		debug(v)

	def Fit(self, table=None, outOp=None):
		if table==None:
			table = self.ownerComp.op('null1')
		if outOp==None:
			outOp = self.ownerComp.op('umap2D_DAT')
		data = []
		for row in table.rows():  
			data.append([float(c.val) for c in row])
		df = pd.DataFrame(data)
		self.loadParams()
		#print(self.params)
		self.Reducer = umap.UMAP(**self.params)
		emb = self.Reducer.fit_transform(df.values)
		self.fill_components_dat(emb, self.ownerComp.par.Ncomponents, 'umap2D_DAT')
		#self.ownerComp.op('script1_callbacks').module.load_models()
		self.ownerComp.par.Generatecolor.pulse()

	def Predict(self, arr):
		sample = arr[0,:]            # shape (60,)
		return self.Reducer.transform(sample.reshape(1, -1))[0]  # e.g. [x, y]


	def loadParams(self):
		paramDAT = self.ownerComp.op('params')
		for row in paramDAT.rows()[1:]:
			key = row[0].val
			raw = row[1].val
			try:
				val = ast.literal_eval(raw)
			except (ValueError, SyntaxError):
				val = raw
			self.params[key] = val	
		print(self.params)


	def fill_components_dat(self, emb, num, dat_op=None):
		if num == 2:
			header = ['x', 'y']
		elif num == 3:
			header = ['x', 'y', 'z']
		else:
			# for num > 3: first 3 are x,y,z, then c4…cN
			header = ['x', 'y', 'z'] + [f'c{i+1}' for i in range(3, num)]
		
		# resolve the DAT operator
		out = op(dat_op)
		if not out:
			raise RuntimeError(f"DAT '{dat_op}' not found")
		
		# clear existing contents and write header
		out.clear()
		out.appendRow(header)

		# write each point, slicing or padding to exactly n entries
		for point in emb:
			row = list(point)[:num]
			if len(row) < num:
				row += [''] * (num - len(row))
			out.appendRow(row)

	def Save(self, name='umap_default'):
		joblib.dump(self.Reducer,name+'.joblib')

