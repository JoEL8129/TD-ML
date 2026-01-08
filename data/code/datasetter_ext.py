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
import _helper_modules as hf

class datasetterext:
	"""
	datasetterext description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp


		self.UseWeights = self.ownerComp.par.Useweights

		self.RecordChop = self.ownerComp.op('record1')
		self.Switches = [self.ownerComp.op('switch1'),self.ownerComp.op('switch2'),self.ownerComp.op('switch3') if self.UseWeights==True else self.ownerComp.op('switch1'),self.ownerComp.op('switch2')]


		self.inputs = {
			'X': {
				'chop': self.ownerComp.op('X'),
				'dat':  self.ownerComp.op('X_dat'),
				'dest': self.ownerComp.op('table_X'),
				'switch': self.ownerComp.par.Input1
			},
			'Y': {
				'chop': self.ownerComp.op('y'),
				'dat':  self.ownerComp.op('y_dat'),
				'dest': self.ownerComp.op('table_y'),
				'switch': self.ownerComp.par.Input2
			},
			'W': {
				'chop': self.ownerComp.op('weight'),
				'dat':  self.ownerComp.op('weight_dat'),
				'dest': self.ownerComp.op('table_weight'),
				'switch': self.ownerComp.par.Input3
			},
		}








		# stored items (persistent across saves and re-initialization):
		storedItems = [
			# Only 'name' is required...
			{'name': 'StoredProperty', 'default': None, 'readOnly': False,
			 						'property': True, 'dependable': True},
		]
		# Uncomment the line below to store StoredProperty. To clear stored
		# 	items, use the Storage section of the Component Editor
		
		# self.stored = StorageManager(self, ownerComp, storedItems)


	def SetupParameters(self):
		page = self.GetPage('controls')


# Callbacks

	def OnValueChange(self, par, prev):
		print(par.name, par.eval(), prev)
		if par.name == 'Record':
			if prev == 0:
				self.StartRecord()
			elif prev == 1:				
				self.StopRecord()


	def OnPulse(self,par):
		if par.name == 'Snap':
			self.Add()		

# Core1

	def Add(self):
		self._appendInput(self.inputs['X'], key='X')
		self._appendInput(self.inputs['Y'], key='Y')
		if self.UseWeights:
			self._appendInput(self.inputs['W'], key='W', is_weight=True)


	def AddY(self):
		self._appendInput(self.inputs['Y'], key='Y')

	def AddX(self):
		self._appendInput(self.inputs['X'], key='X')
		

	def StartRecord(self):
		self.RecordChop.par.reset.pulse()
		#self.ownerComp.op('table_y_label_rec').clear()

		#for switchOp in self.Switches:
		#	switchOp.par.index = 0
		self.RecordChop.par.record = 1
		#for row in self.ownerComp.op('y_dat').rows():
		#	self.ownerComp.op('table_y_label_rec').appendRow(row)
		print('started recording')

	def WriteRecordingToBuffers(self):
		xRec = self.ownerComp.op('x_rec')
		yRec = self.ownerComp.op('y_rec')

		xBuf = self.ownerComp.op('x_buffer')
		yBuf = self.ownerComp.op('y_buffer')
		xBuf.clear()
		yBuf.clear()


		if self.ownerComp.par.Input1.eval() == 'chop':
			self.ownerComp.op('sample_names_x').par.snap.pulse()
			src = self.ownerComp.op('select3')
			row = src.row(0)
			xBuf.appendRow(row)
			hf._append_chop_to_table(xRec,xBuf,False)

		if self.ownerComp.par.Input2.eval() == 'chop':
			self.ownerComp.op('sample_names_y').par.snap.pulse()
			src = self.ownerComp.op('select4')
			row = src.row(0)
			yBuf.appendRow(row)
			hf._append_chop_to_table(yRec,yBuf,False)

		elif self.ownerComp.par.Input2.eval() == 'dat':
			self.ownerComp.op('table_y_label_rec').clear()
			row = self.ownerComp.op('y_dat').rows()[0]
			for i in range(self.ownerComp.op('select_x_from_rec').numSamples):
				self.ownerComp.op('table_y_label_rec').appendRow(row)
			self._appendDatToTable(self.ownerComp.op('table_y_label_rec'),yBuf)
			#header = [f"Label {i+1}" for i in range(dat.numCols)]
			#table.appendRow(header)		

	def ThreshRecording(self, thresh: float):
		"""
		Remove rows from x_buffer / y_buffer whose corresponding value in thresh_rec
		is below `thresh`. We collect indices first and then delete in reverse order
		to avoid index shifts.
		"""
		threshRec = self.ownerComp.op('thresh_rec')
		xBuf = self.ownerComp.op('x_buffer')
		yBuf = self.ownerComp.op('y_buffer')

		# Build the list of row indices to delete
		# (Assumes buffers are aligned so that row i corresponds to sample i.)
		to_delete = []
		ns = int(threshRec.numSamples)
		for i in range(ns):
			# Access first channel's ith sample
			val = float(threshRec[0][i])
			if val < thresh:
				to_delete.append(i)

		# Delete in reverse so remaining indices stay valid
		for i in reversed(to_delete):
			self._deleteFromBuffersAt(i, xBuf, yBuf)

	def _deleteFromBuffersAt(self, i: int, xBuf, yBuf):
		"""
		Deletes row i from both buffers if it exists.
		Adjust this if your buffers have headers or different shapes.
		"""
		# If your buffers include a header row, offset by +1:
		# i_eff = i + 1
		i_eff = i + 1

		# Guard against out-of-range (buffers can be different lengths)
		if hasattr(xBuf, 'numRows') and i_eff < xBuf.numRows:
			xBuf.deleteRow(i_eff)
		if hasattr(yBuf, 'numRows') and i_eff < yBuf.numRows:
			yBuf.deleteRow(i_eff)

	def StopRecord(self):
		#self.RecordChop.par.reset.pulse()
		self.RecordChop.par.record = 0
		self.WriteRecordingToBuffers()
		print(self.ownerComp.op('x_buffer').numRows)
		if self.ownerComp.par.Threshrec==True:
			self.ThreshRecording(self.ownerComp.par.Threshold.eval())

		self._appendDatToTable(self.ownerComp.op('x_buffer'),self.ownerComp.op('table_X'),True)
		self._appendDatToTable(self.ownerComp.op('y_buffer'),self.ownerComp.op('table_y'),True)
		
		print(self.ownerComp.op('x_buffer').numRows)
		#self.

		#self._append_dat_rows(self.ownerComp.op('table_y_label_rec'),self.ownerComp.op('table_y'))

		#for switchOp in self.Switches:
		#	switchOp.par.index = 1
		#self.RecordChop.par.record = 1

		print('stopped recording')
	




# Helpers


	def _appendRows(self, row_values, targetTableOp=None, number=50):
		"""
		Append the same data row multiple times to a target Table DAT.

		Args:
			row_values (list/tuple): Cells for the data *after* the sample label
				(i.e., these map to the feature columns; the first column will be
				auto-filled with 'sample n').
			targetTableOp (OP or str or None): Target Table DAT OP or OP path.
				Defaults to self.ownerComp.op('table_y') if None.
			number (int): How many identical rows to append. Must be >= 1.

		Behavior:
			- If the table has no header, a placeholder header is created:
			  ["Label", "Label 1", "Label 2", ...].
			- If header width doesn't match len(row_values)+1, the row will be
			  padded with '' or truncated to fit the header.
			- The first cell of each appended row is an auto-generated sample label
			  ("sample n") using _nextSampleLabel().
		"""
		# Resolve target table
		table = targetTableOp
		if table is None:
			table = self.ownerComp.op('table_y')
		elif isinstance(table, str):
			table = self.ownerComp.op(table)

		if table is None:
			print("⚠️ _appendRows: target table operator not found.")
			return

		# Normalize row values
		if row_values is None:
			row_values = []
		elif not isinstance(row_values, (list, tuple)):
			# allow single scalar; make it a one-element list
			row_values = [row_values]

		if not isinstance(number, int) or number < 1:
			print("⚠️ _appendRows: 'number' must be an integer >= 1.")
			return

		# Ensure there's a header; if none, create a generic header matching row width
		expected_data_cols = len(row_values)
		if table.numRows == 0:
			header = ["Label"] + [f"Label {i+1}" for i in range(expected_data_cols)]
			table.appendRow(header)
		else:
			# Check header width and reconcile row width accordingly
			header_cols = table.numCols
			# header should be 1 (label col) + data cols
			target_width = max(1, header_cols)  # guard
			target_data_cols = target_width - 1

			if target_data_cols != expected_data_cols:
				# Adjust row to match header width
				if expected_data_cols < target_data_cols:
					# pad with blanks
					row_values = list(row_values) + [''] * (target_data_cols - expected_data_cols)
				else:
					# truncate extra values
					row_values = list(row_values)[:target_data_cols]

		# Append rows
		for _ in range(number):
			row = [self._nextSampleLabel(table)] + list(row_values)
			table.appendRow(row)

	def _switch_mode(self, par):
		"""
		Robustly detect CHOP vs DAT from a menu/string parameter.
		Treat anything containing 'dat' or 'table' (case-insensitive) as DAT; else CHOP.
		"""
		try:
			v = str(par.eval())
		except Exception:
			v = str(par)
		v = v.lower()
		if 'dat' in v or 'table' in v:
			return 'dat'
		return 'chop'
	
	def _appendInput(self,src, key='X', is_weight=False):
		"""
		Append one input to its destination table based on its switch mode.
		- src['switch'] should be a menu param with values like 'CHOP' / 'DAT'.
		- When 'CHOP': use _appendChopToTable (weights use _appendWeightToTable).
		- When 'DAT' : use _appendDatToTable (generic, header in row 0).
		"""
		mode = self._switch_mode(src['switch'])

		if mode == 'dat':
			if src['dat'] is None:
				print(f"⚠️ {key}: DAT operator not found.")
				return
			self._appendDatToTableSimple(src['dat'], src['dest'])

		else:
			if src['chop'] is None:
				print(f"⚠️ {key}: CHOP operator not found.")
				return
			if is_weight:
				self._appendWeightToTable(src['chop'], src['dest'])
			else:
				self._appendChopToTable(src['chop'], src['dest'])

	def _appendChopToTable(self, chop, table):
			"""
			CHOP → table with the 4 requested cases:

			1) 1 channel, 1 sample
			- add 1 data sample
			- feature size = 1 (channel)
			- header from channel names

			2) 1 channel, many samples
			- add 1 data sample
			- feature size = #samples
			- header from sample indices (s1..sN)

			3) many channels, 1 sample
			- add 1 data sample
			- feature size = #channels
			- header from channel names

			4) many channels, many samples
			- add many data samples (one per channel)
			- feature size = #samples
			- header from sample indices (s1..sN)
			"""
			nC, nS = chop.numChans, chop.numSamples

			# Case 1: 1C, 1S  → one row, features = channels (size 1)
			if nC == 1 and nS == 1:
				self._ensureHeaderFromChop(table, chop)  # header ['', <chanName>]
				row = [self._nextSampleLabel(table), chop[0].eval()]
				table.appendRow(row)
				return

			# Case 2: 1C, many S → one row, features = samples
			if nC == 1 and nS > 1:
				self._ensureHeaderFromSamples(table, chop)  # header ['', s1..sN]
				row = [self._nextSampleLabel(table)] + [chop[0].eval(i) for i in range(nS)]
				table.appendRow(row)
				return

			# Case 3: many C, 1S → one row, features = channels
			if nC > 1 and nS == 1:
				self._ensureHeaderFromChop(table, chop)  # header ['', chan1..chanC]
				row = [self._nextSampleLabel(table)] + [chan.eval() for chan in chop.chans()]
				table.appendRow(row)
				return

			# Case 4: many C, many S → many rows (one per channel), features = samples
			if nC > 1 and nS > 1:
				self._ensureHeaderFromSamples(table, chop)  # header ['', s1..sN]
				for chan in chop.chans():
					row = [self._nextSampleLabel(table)] + [chan.eval(i) for i in range(nS)]
					table.appendRow(row)
				return

	def Chop_to_table(self,chopOp, tableOp, *, channels='rows', samples='cols', add_column_headers=True):
		"""
		Write a CHOP into a Table DAT with headers only for columns.

		Parameters
		----------
		chopOp : OP
			The CHOP operator (e.g. op('noise1')).
		tableOp : OP
			The Table DAT operator (e.g. op('table1')).
		channels : {'rows','cols'}
			Whether CHOP channels should be written as rows or columns.
		samples : {'rows','cols'}
			Whether CHOP samples should be written as rows or columns.
			Must be the opposite of `channels`.
		add_column_headers : bool
			If True, writes a single header row naming the columns only.
			- If channels == 'cols', header = channel names
			- If samples  == 'cols', header = sample indices
		"""
		# Validate orientation
		if (channels, samples) not in {('rows','cols'), ('cols','rows')}:
			raise ValueError("Use either channels='rows', samples='cols' OR channels='cols', samples='rows'.")

		num_chans   = chopOp.numChans
		num_samples = chopOp.numSamples
		chan_names  = [ch.name for ch in chopOp.chans()]
		chan_arrays = [chopOp[i].numpyArray() for i in range(num_chans)]

		if channels == 'rows' and samples == 'cols':
			# Columns are samples → optional header = sample indices
			if add_column_headers:
				tableOp.appendRow([str(i) for i in range(num_samples)])

			# Each row is one channel's samples (no row headers)
			for ci in range(num_chans):
				tableOp.appendRow(chan_arrays[ci].tolist())

		else:
			# channels == 'cols' and samples == 'rows'
			# Columns are channels → optional header = channel names
			if add_column_headers:
				tableOp.appendRow(chan_names)

			# Each row is one sample across all channels (no row headers)
			for si in range(num_samples):
				tableOp.appendRow([arr[si] for arr in chan_arrays])


	def _append_dat_rows(self, src, dst):
		"""Append all rows from src DAT to dst DAT."""
		for row in src.rows():
			dst.appendRow([cell.val for cell in row])

	def _appendDatToTable(self, dat, table, has_header_row = False, header_row=0):
		"""
		Generic DAT → table append.
		Assumes the DAT has a header row at `header_row` (default 0) containing column names.
		Appends each subsequent row as one 'sample n'.
		If no header is present (Input2header == False), creates:
		Label | Label 1 | Label 2 | ... | Label N
		"""
		if dat.numRows == 0:
			print("⚠️ DAT is empty; nothing to append.")
			return
		if has_header_row == True:
			# Build header from the DAT's header row
			names = []
			for c in range(dat.numCols):
				cell = dat[header_row, c]
				names.append(cell.val if cell is not None else f'col{c+1}')

			# Ensure table has a header (first column is the sample label column)
			self._ensureHeaderFromNames(table, names)

			# Append all data rows after the header row from the DAT
			for r in range(header_row + 1, dat.numRows):
				values = []
				for c in range(dat.numCols):
					cell = dat[r, c]
					values.append(cell.val if cell is not None else '')
				table.appendRow([self._nextSampleLabel(table)] + values)

		else:
			# Create header if table is empty: "Label | Label 1 | Label 2 | ... | Label N"
			if table.numRows == 0:
				# Make a single header row with (1 + dat.numCols) cells
				header = [f"Label {i+1}" for i in range(dat.numCols)]
				table.appendRow(header)

			# Append all DAT rows as samples
			for r in range(0, dat.numRows):
				values = []
				for c in range(dat.numCols):
					cell = dat[r, c]
					values.append(cell.val if cell is not None else '')
				table.appendRow(values)

	def _appendDatToTableSimple(self, dat, table):
		"""
		Append *all* cells from `dat` to `table` exactly as-is.
		Ignores headers and doesn't add sample labels; simply copies rows.
		"""
		if dat.numRows == 0 or dat.numCols == 0:
			print("⚠️ DAT is empty; nothing to append.")
			return

		for r in range(dat.numRows):
			row_vals = []
			for c in range(dat.numCols):
				cell = dat[r, c]
				row_vals.append(cell.val if cell is not None else '')
			table.appendRow(row_vals)


	def _ensureHeaderFromChop(self,table, chop):
		chan_names = [c.name for c in chop.chans()]
		self._ensureHeaderFromNames(table, chan_names)

	def _ensureHeaderFromSamples(self,table, chop):
		"""
		Ensure header is ['', s1, s2, ..., sN] where N = chop.numSamples.
		"""
		names = [f's{i+1}' for i in range(chop.numSamples)]
		self._ensureHeaderFromNames(table, names)

	def _ensureHeaderFromNames(self,table, names):
		"""
		Ensure header is ['', <name1>, <name2>, ...].
		"""
		desired = [''] + list(names)

		if table.numRows == 0:
			table.appendRow(desired)
			return

		# read existing header row
		existing = []
		for col in range(table.numCols):
			cell = table[0, col]
			existing.append(cell.val if cell is not None else '')

		if len(existing) != len(desired) or any(a != b for a, b in zip(existing, desired)):
			table.clear()
			table.appendRow(desired)

	def _nextSampleLabel(self,table):
		"""
		Returns 'sample n' where n = data-row index (header excluded) + 1.
		Header is row 0 → first data row is 'sample 1'.
		"""
		n = 1 if table.numRows == 0 else table.numRows  # because header consumes row 0
		return f"sample {n}"

	def _appendWeightToTable(self,chop, table):
		"""
		Special append for W, but keep the same table shape:
		• ERROR if both nC>1 and nS>1 (invalid).
		• Multiple channels (single sample) → one row, columns are channel names.
		• Multiple samples (single channel) → one row per sample, single param column.
		• Single channel & sample → one row.
		Always includes blank top-left header and 'sample n' in the first column.
		"""
		nC, nS = chop.numChans, chop.numSamples

		if nC > 1 and nS > 1:
			print("⚠️ Weight CHOP has both multiple channels and multiple samples — cannot append.")
			return

		self._ensureHeaderFromChop(table, chop)

		if nC > 1:
			# single row, many params
			row = [self._nextSampleLabel(table)] + [chan.eval() for chan in chop.chans()]
			table.appendRow(row)

		elif nS > 1:
			# many rows, single param
			for i in range(nS):
				row = [self._nextSampleLabel(table), chop[0].eval(i)]
				table.appendRow(row)

		else:
			# single row, single param
			row = [self._nextSampleLabel(table), chop[0].eval()]
			table.appendRow(row)
