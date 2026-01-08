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

class presentext:
	"""
	presenter description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		# properties
		TDF.createProperty(self, 'Activepage', value='page0', dependable=True,
						   readOnly=False)

		# attributes:
		#lself.Activepage = 'page0' # attribute
		self.Lastpage = 'page1' # promoted attribute
		self.ResetPages()
		# stored items (persistent across saves and re-initialization):
		storedItems = [
			# Only 'name' is required...
			{'name': 'StoredProperty', 'default': None, 'readOnly': False,
			 						'property': True, 'dependable': True},
		]
		# Uncomment the line below to store StoredProperty. To clear stored
		# 	items, use the Storage section of the Component Editor
		
		# self.stored = StorageManager(self, ownerComp, storedItems)

	def ResetPages(self):
		pagesOp = self.ownerComp.op('sort1')
		for cell in pagesOp.rows():
			self.ownerComp.op(str(cell[0].val)).par.display = 0
			self.ownerComp.op(str(cell[0].val)).par.enable = 0

		self.Activepage = 'page0'
		self.ownerComp.op(self.Activepage).par.display = 1
		self.ownerComp.op(self.Activepage).par.enable = 1

		


	def SetActivePage(self, s='page0'):
		self.Lastpage = self.Activepage
		self.Activepage = s
		self.ownerComp.op(self.Activepage).par.display = 1
		self.ownerComp.op(self.Activepage).par.enable = 1

		self.ownerComp.op(self.Lastpage).par.display = 0
		self.ownerComp.op(self.Lastpage).par.enable = 0



	def PromotedFunction(self, v):
		debug(v)