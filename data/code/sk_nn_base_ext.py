# BaseExt.py
from base_ext import base_ext
from ParTemplate import ParTemplate
from TDStoreTools import StorageManager
import TDFunctions as TDF
import json
import ast
from datetime import datetime, timezone
from pathlib import Path
import helper_modules as hf

class sklearntest(base_ext):
	"""
	Minimal reusable base for TouchDesigner extensions.
	Provides GetPage() to get-or-create a custom parameter page.
	"""
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp

		storedItems = [
			# Only 'name' is required...
			{'name': 'Pipeline', 'default': None, 'readOnly': False,
			 						'property': True, 'dependable': True},
			{'name': 'Meta', 'default': {}, 'readOnly': False,
			 						'property': True, 'dependable': True},									 									 								 						
		]




		self.x = None
		self.y = None
		self.w = None


		#self.PreLoad()
		

		self.stored = StorageManager(self, ownerComp, storedItems)
		self.SetupFileIOPars()


	def Log(self, *args):
		print(f"[{self.ownerComp.path}]", *args)

	def SetupPars(self):
		page = self.GetPage('ctrl')
		pars = [
			ParTemplate('Train', par_type='pulse', label='Train', callback=True),
			ParTemplate('Myfloat', par_type='float', label='My Float', size = int(2), default=0.5, norm_min=0.0, norm_max=1.5, min = 0, max = 2 , clamp=True, callback=True),
			ParTemplate('Mode', par_type='menu', label='Mode',
						menu_names=['a', 'b', 'c'], menu_labels=['Alpha', 'Beta', 'Gamma'], default='b',callback=True),
			ParTemplate('Tint', par_type='rgb', label='Tint', default=(1.0, 0.5, 0.2),callback=True),
			ParTemplate('Tinte', par_type='rgb', label='Tinte', default=(1.0, 0.5, 0.2),callback=True),
			ParTemplate('Flo', par_type = 'float', size = 2)
		]

		for p in pars:
			p.createPar(page)

	def SetupFileIOPars(self):
		page = self.GetPage('ctrl')
		pars = [
			ParTemplate('Save', par_type='pulse', label='Save', callback=True),
			ParTemplate('Modelmenu', par_type='menu', label='Load Model',
						menu_source=tdu.TableMenu(self.ownerComp.op('models_table'), includeFirstRow=True),callback=True),
			ParTemplate('Tint', par_type='rgb', label='Tint', default=(1.0, 0.5, 0.2),callback=True),
			ParTemplate('Tinte', par_type='rgb', label='Tinte', default=(1.0, 0.5, 0.2),callback=True),
			ParTemplate('Flo', par_type = 'float', size = 2)
		]

		for p in pars:
			p.createPar(page)

