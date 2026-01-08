# extinhlearn (Extension DAT)
from TDStoreTools import StorageManager
import TDFunctions as TDF
from BaseExt import BaseExt
from ParTemplate import ParTemplate

class extinhlearn(BaseExt):
	def __init__(self, ownerComp):
		super().__init__(ownerComp)



		#TDF.createProperty(self, 'MyProperty', value=0, dependable=True, readOnly=False)
		#self.a = 0
		#self.B = 1  # promoted
		self.SetupPars()
		#self.UpdatePars()

	
	def SetupPars(self):
		page = self.GetPage('ctrl')
		pars = [
			ParTemplate('Name', par_type='pulse', label='Name', callback=True),
			ParTemplate('Myfloat', par_type='float', label='My Float', size = int(2), default=0.5, norm_min=0.0, norm_max=1.5, min = 0, max = 2 , clamp=True, callback=True),
			ParTemplate('Mode', par_type='menu', label='Mode',
						menu_names=['a', 'b', 'c'], menu_labels=['Alpha', 'Beta', 'Gamma'], default='b',callback=True),
			ParTemplate('Tint', par_type='rgb', label='Tint', default=(1.0, 0.5, 0.2),callback=True),
			ParTemplate('Tinte', par_type='rgb', label='Tinte', default=(1.0, 0.5, 0.2),callback=True),
			ParTemplate('Flo', par_type = 'float', size = 2)
		]

		for p in pars:
			p.createPar(page)

	def UpdatePars(self):
		page = self.GetPage('ctrl')
		page.destroy()
		self.SetupPars()

	def OnMyfloat(self, par):
		val = par.eval()
		print(val)

	def OnName(self,par):
		#val = par.eval()
		print(par.eval())
	def OnTint(self,par):
		#val = par.eval()
		print(par.eval())


	def OnMode(self,par):
		print(par.eval())

	def OnTinte(self,par):
		print(par.eval())