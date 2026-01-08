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
from text1 import ParTemplate

class channelstorage:
	"""
	channelstorage description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp



		# stored items (persistent across saves and re-initialization):
		storedItems = [
			# Only 'name' is required...
			{'name': 'Storedchannels', 'default': [], 'readOnly': False,
			 						'property': True, 'dependable': True},
		]
		# Uncomment the line below to store StoredProperty. To clear stored
		# 	items, use the Storage section of the Component Editor
		self.Inputchop = self.ownerComp.op('in1')
		self.SetupPars()
		self.stored = StorageManager(self, ownerComp, storedItems)

	def SetupPars(self):
		page2 = self.GetPage('Main')
		pars2 = [
				ParTemplate(
					'Addchannel', 
					par_type='pulse', 
					label='Add Channel',
					callback=True),										

		]

		for p in pars2:
			p.createPar(page2) 


	def OnAddchannel(self, par):
		for chan in self.Inputchop.chans():
			self.addChannel(chan.name)
			

	def addChannel(self, channelName):
		if channelName not in self.Storedchannels:
			self.Storedchannels.append(channelName)
			print('chan added')
		else:
			print(f'Channel "{channelName}" already exists in Storedchannels')

	def myFunction(self, v):
		debug(v)

	def PromotedFunction(self, v):
		debug(v)

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