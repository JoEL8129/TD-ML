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
import data_sanit as ds
import os
import shutil

class datalogic:
	"""
	datalogic description
	"""
	def __init__(self, ownerComp):
		self.ownerComp = ownerComp

		self.x_input_chop = self.ownerComp.op('chop_x')
		self.y_input_chop = self.ownerComp.op('chop_y')
		self.w_input_chop = self.ownerComp.op('chop_w')
		self.x_input_dat = self.ownerComp.op('dat_x')
		self.y_input_dat = self.ownerComp.op('dat_y')
		self.w_input_dat = self.ownerComp.op('dat_w')

		self.x_buffer = self.ownerComp.op('x_buffer')
		self.y_buffer = self.ownerComp.op('y_buffer')
		self.w_buffer = self.ownerComp.op('w_buffer')

		self.x_table = self.ownerComp.op('x_table')
		self.y_table = self.ownerComp.op('y_table')
		self.w_table = self.ownerComp.op('w_table')
		self.seqid_table = self.ownerComp.op('seqid_table')

		self.use_w = self.ownerComp.par.Useweights.eval()
		storedItems = [
			{'name': 'X_names', 'default': [], 'readOnly': False,
			 						'property': True, 'dependable': True},
			{'name': 'Y_names', 'default': [], 'readOnly': False,
			 						'property': True, 'dependable': True},
		]
		
		self.RecordChop = self.ownerComp.op('record1')
		self.CallbacksComp = self.ownerComp.op('par_callbacks')
		self.stored = StorageManager(self, ownerComp, storedItems)


	def AddChannelNamesFromInputs(self):
		for chan in self.x_input_chop.chans():
			if chan.name not in self.X_names:
				self.X_names.append(chan.name)
				print('chan added')

		for chan in self.y_input_chop.chans():
			if chan.name not in self.Y_names:
				self.Y_names.append(chan.name)
				print('chan added')
		

	def StartRecord(self):

		if self.x_input_chop.numSamples>1:
			self.ownerComp.op('shuffle_if_multi_sample_x_rec').par.Index = 1
		else:
			self.ownerComp.op('shuffle_if_multi_sample_x_rec').par.Index = 0

		if self.y_input_chop.numSamples>1:
			self.ownerComp.op('shuffle_if_multi_sample_y_rec').par.Index = 1
		else:
			self.ownerComp.op('shuffle_if_multi_sample_y_rec').par.Index = 0

		if self.w_input_chop.numSamples>1:
			self.ownerComp.op('shuffle_if_multi_sample_w_rec').par.Index = 1
		else:
			self.ownerComp.op('shuffle_if_multi_sample_w_rec').par.Index = 0



		self.RecordChop.par.reset.pulse()
		self.RecordChop.par.record = 1

		print('started recording')

	def StopRecord(self):
		self.ClearBuffers()
		self.RecordChop.par.record = 0
		x_rec = self.ownerComp.op('x_rec')
		y_rec = self.ownerComp.op('y_rec')
		w_rec = self.ownerComp.op('w_rec')

		# Check if Matchchanname is enabled
		use_match = self.ownerComp.par.Matchchanname.eval()

		if use_match:
			# Use channels_to_row_by_name for direct channel name matching
			self.channels_to_row_by_name(x_rec, self.x_buffer)
			if self.ownerComp.par.Inputtype2.eval()=='chop':
				self.channels_to_row_by_name(y_rec, self.y_buffer)
			elif self.ownerComp.par.Inputtype2.eval()=='dat':
				for i in range(self.x_buffer.numRows):
					self.Dat_to_table(self.y_input_dat,self.y_buffer)
			
			if self.use_w:
				if self.ownerComp.par.Inputtype3.eval()=='chop':
					self.channels_to_row_by_name(w_rec, self.w_buffer)
				elif self.ownerComp.par.Inputtype3.eval()=='dat':
					for i in range(self.x_buffer.numRows):
						self.Dat_to_table(self.w_input_dat,self.w_buffer)
		else:
			# Use original Chop_to_table method
			self.Chop_to_table(x_rec,self.x_buffer,channels='cols', samples='rows')
			if self.ownerComp.par.Inputtype2.eval()=='chop':
				self.Chop_to_table(y_rec,self.y_buffer,channels='cols', samples='rows')
			elif self.ownerComp.par.Inputtype2.eval()=='dat':
				for i in range(self.x_buffer.numRows):
					self.Dat_to_table(self.y_input_dat,self.y_buffer)
			
			if self.use_w:
				if self.ownerComp.par.Inputtype3.eval()=='chop':
					self.Chop_to_table(w_rec,self.w_buffer,channels='cols', samples='rows')
				elif self.ownerComp.par.Inputtype3.eval()=='dat':
					for i in range(self.x_buffer.numRows):
						self.Dat_to_table(self.w_input_dat,self.w_buffer)			

		self._deleteFromBuffersAt(0,self.x_buffer)
		self._deleteFromBuffersAt(0,self.y_buffer)
		if self.use_w:
			self._deleteFromBuffersAt(0,self.w_buffer)	

		if self.ownerComp.par.Threshrec==True:
			self.ThreshRecording(self.ownerComp.par.Threshold.eval())
		


		self.Dat_to_table(self.x_buffer,self.x_table)
		self.Dat_to_table(self.y_buffer,self.y_table)
		if self.use_w:
			self.Dat_to_table(self.w_buffer,self.w_table)

		# Add sequential IDs - use number of rows in buffer (after deletions/threshold)
		num_rows_added = self.x_buffer.numRows
		self.AddSeqIdToTable(num_rows_added, self.seqid_table)

		print('stopped recording')
		self.AddChannelNamesFromInputs()



	def ThreshRecording(self, thresh: float):
		threshRec = self.ownerComp.op('t_rec')
		xBuf = self.x_buffer
		yBuf = self.y_buffer
		wBuf = self.w_buffer


		# Build the list of row indices to delete
		to_delete = []
		ns = int(threshRec.numSamples)
		for i in range(ns):
			# Access first channel's ith sample
			val = float(threshRec[0][i])
			if val < thresh:
				to_delete.append(i)

		for i in reversed(to_delete):
			self._deleteFromBuffersAt(i, xBuf)
			self._deleteFromBuffersAt(i, yBuf)
			self._deleteFromBuffersAt(i, wBuf)

	def _deleteFromBuffersAt(self, i: int, buf):
		i_eff = i + 1
		if hasattr(buf, 'numRows') and i_eff < buf.numRows:
			buf.deleteRow(i)



	def Chop_to_table(self,chopOp, tableOp, *, channels='rows', samples='cols', add_column_headers=False):
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

		elif channels == 'cols' and samples == 'rows':
			# channels == 'cols' and samples == 'rows'
			# Columns are channels → optional header = channel names
			if add_column_headers:
				tableOp.appendRow(chan_names)

			# Each row is one sample across all channels (no row headers)
			for si in range(num_samples):
				tableOp.appendRow([arr[si] for arr in chan_arrays])

	def channels_to_row_by_name(self, chopOp, tableOp):
		"""
		Convert CHOP channels to table rows, mapping channels to columns by name
		based on X_names or Y_names order. Each sample becomes a row.
		
		The function determines which name list to use based on the destination table:
		- x_buffer or x_table -> uses X_names
		- y_buffer or y_table -> uses Y_names
		- w_buffer or w_table -> uses Y_names (default)
		
		Args:
			chopOp: The CHOP operator to read from
			tableOp: The table operator to write to
		"""
		# Determine which name list to use based on destination table
		if tableOp == self.x_buffer or tableOp == self.x_table:
			name_list = self.X_names
		elif tableOp == self.y_buffer or tableOp == self.y_table:
			name_list = self.Y_names
		elif tableOp == self.w_buffer or tableOp == self.w_table:
			# For weights, default to Y_names (could be changed if W_names is added)
			name_list = self.Y_names
		else:
			# Fallback: try to infer from table name or use X_names as default
			table_name = tableOp.name.lower()
			if 'x' in table_name:
				name_list = self.X_names
			elif 'y' in table_name:
				name_list = self.Y_names
			else:
				name_list = self.X_names
		
		# Get channel data from CHOP
		num_samples = chopOp.numSamples
		chop_channels = {ch.name: ch for ch in chopOp.chans()}
		chop_channel_names = list(chop_channels.keys())
		
		# If name_list is empty, initialize it from CHOP channels
		if not name_list:
			name_list.extend(chop_channel_names)
			print(f"⚠️ Name list was empty, initialized from CHOP channels: {chop_channel_names}")
		
		# Create mapping: for each name in name_list, get the corresponding channel index
		# and build a list of channel arrays in the order of name_list
		ordered_channels = []
		missing_channels = []
		
		for name in name_list:
			if name in chop_channels:
				ordered_channels.append(chop_channels[name].numpyArray())
			else:
				ordered_channels.append(None)  # Placeholder for missing channel
				missing_channels.append(name)
		
		# Warn about missing channels
		if missing_channels:
			print(f"⚠️ Warning: Channels in name list but not in CHOP: {missing_channels}")
		
		# Warn about extra channels in CHOP not in name list
		extra_channels = [name for name in chop_channel_names if name not in name_list]
		if extra_channels:
			print(f"⚠️ Warning: Channels in CHOP but not in name list: {extra_channels}")
		
		# Create rows: each sample becomes a row with values in columns ordered by name_list
		for si in range(num_samples):
			row_values = []
			for chan_array in ordered_channels:
				if chan_array is not None:
					# Channel exists, get the sample value
					row_values.append(float(chan_array[si]))
				else:
					# Channel missing, leave cell empty
					row_values.append('')
			tableOp.appendRow(row_values)

	def Dat_to_table(self, datOp, tableOp):
		if datOp.numRows == 0:
			print(datOp, " is empty; nothing to append.")
			return		
		for r in range(0, datOp.numRows):
			values = []
			for c in range(datOp.numCols):
				cell = datOp[r, c]
				values.append(cell.val if cell is not None else '')
			tableOp.appendRow(values)		

	def AddSeqIdToTable(self, num_rows: int, tableOp):
		"""
		Add sequential ID rows to the seqid table.
		
		Args:
			num_rows: Number of rows to add (should match rows added to x/y/w tables)
			tableOp: The table operator to write to (seqid_table)
		"""
		# Check if Addsequentialid toggle is enabled
		if not self.ownerComp.par.Addsequentialid.eval():
			return
		
		# Get current Seqid parameter value
		seqid = self.ownerComp.par.Seqid.eval()
		
		# Append num_rows rows, each containing the seqid value
		for _ in range(num_rows):
			tableOp.appendRow([seqid])



	def ClearBuffers(self):
		self.x_buffer.clear()
		self.y_buffer.clear()
		self.w_buffer.clear()

	def ClearTables(self):

		self.x_table.clear()
		self.y_table.clear()
		self.w_table.clear()
		self.seqid_table.clear()
		# Clear stored names when tables are cleared
		self.X_names = []
		self.Y_names = []

		


	def UpdateNamesFromLoad(self):
		self.X_names = self.ownerComp.op('load_x_names').row(0)
		self.Y_names = self.ownerComp.op('load_y_names').row(0)	






	def Snap(self):
		# Check if Matchchanname is enabled
		use_match = self.ownerComp.par.Matchchanname.eval()
		
		# Store row count before processing to calculate how many rows were added
		rows_before = self.x_table.numRows
		
		# Process x input
		input_type_x = self.ownerComp.par.Inputtype1.eval()
		if input_type_x == 'chop':
			if use_match:
				# Use channels_to_row_by_name for direct channel name matching
				self.channels_to_row_by_name(self.x_input_chop, self.x_table)
			else:
				# Use original Chop_to_table method
				channels_x = self.ownerComp.par.Inputchannelsto1.eval()
				samples_x = self.ownerComp.par.Inputsamplesto1.eval()
				if channels_x == samples_x:
					print("⚠️ Warning: Inputchannelsto1 and Inputsamplesto1 have the same value. They must be different.")
				else:
					self.Chop_to_table(self.x_input_chop, self.x_table, channels=channels_x, samples=samples_x)
		elif input_type_x == 'dat':
			self.Dat_to_table(self.x_input_dat, self.x_table)
		
		# Process y input
		input_type_y = self.ownerComp.par.Inputtype2.eval()
		if input_type_y == 'chop':
			if use_match:
				# Use channels_to_row_by_name for direct channel name matching
				self.channels_to_row_by_name(self.y_input_chop, self.y_table)
			else:
				# Use original Chop_to_table method
				channels_y = self.ownerComp.par.Inputchannelsto2.eval()
				samples_y = self.ownerComp.par.Inputsamplesto2.eval()
				if channels_y == samples_y:
					print("⚠️ Warning: Inputchannelsto1 and Inputsamplesto1 have the same value. They must be different.")
				else:
					self.Chop_to_table(self.y_input_chop, self.y_table, channels=channels_y, samples=samples_y)
		elif input_type_y == 'dat':
			self.Dat_to_table(self.y_input_dat, self.y_table)
		
		# Process w input (only if use_w flag is set)
		if self.use_w:
			input_type_w = self.ownerComp.par.Inputtype3.eval()
			if input_type_w == 'chop':
				if use_match:
					# Use channels_to_row_by_name for direct channel name matching
					self.channels_to_row_by_name(self.w_input_chop, self.w_table)
				else:
					# Use original Chop_to_table method
					channels_w = self.ownerComp.par.Inputchannelsto3.eval()
					samples_w = self.ownerComp.par.Inputsamplesto3.eval()
					if channels_w == samples_w:
						print("⚠️ Warning: Inputchannelsto3 and Inputsamplesto3 have the same value. They must be different.")
					else:
						self.Chop_to_table(self.w_input_chop, self.w_table, channels=channels_w, samples=samples_w)
			elif input_type_w == 'dat':
				self.Dat_to_table(self.w_input_dat, self.w_table)

		# Add sequential IDs - calculate rows added
		rows_after = self.x_table.numRows
		num_rows_added = rows_after - rows_before
		if num_rows_added > 0:
			self.AddSeqIdToTable(num_rows_added, self.seqid_table)
		
		self.AddChannelNamesFromInputs()


	def Save(self,name=None):
		path = self.ownerComp.par.Folder
		suffix = '.dat'
		self.ownerComp.op('fileout5').par.file = path+'/'+name+'/'+name+'_x'+suffix
		self.ownerComp.op('fileout6').par.file = path+'/'+name+'/'+name+'_y'+suffix
		self.ownerComp.op('fileout7').par.file = path+'/'+name+'/'+name+'_w'+suffix
		self.ownerComp.op('fileout5').par.write.pulse()
		self.ownerComp.op('fileout6').par.write.pulse()
		self.ownerComp.op('fileout7').par.write.pulse()
		print('saved: '+name+suffix+' at: '+path)		
		run(self.ownerComp.op('folder1').par.refreshpulse.pulse(), delayFrames = 5)




	def Load(self):
		self.ClearTables()
		path = self.ownerComp.par.Folder.eval()
		name = self.ownerComp.par.Items.eval()
		suffix = '.dat'
		self.ownerComp.op('load_file_x').par.file = path+'/'+name+'/'+name+'_x'+suffix
		self.ownerComp.op('load_file_y').par.file = path+'/'+name+'/'+name+'_y'+suffix
		self.ownerComp.op('load_file_w').par.file = path+'/'+name+'/'+name+'_w'+suffix

		#op('w_table').par.file = path+'/'+name+'/'+name+'_w'+suffix
		self.ownerComp.op('load_file_x').par.syncfile.pulse()
		self.ownerComp.op('load_file_y').par.syncfile.pulse()
		self.ownerComp.op('load_file_w').par.syncfile.pulse()

		self.Dat_to_table(self.ownerComp.op('load_x'),self.x_table)
		self.Dat_to_table(self.ownerComp.op('load_y'),self.y_table)
		self.Dat_to_table(self.ownerComp.op('load_w'),self.w_table)

		self.UpdateNamesFromLoad()





	def DeleteSelectedDataset(self):
		relpath = self.ownerComp.par.Folder+'/'+self.ownerComp.par.Items.eval() # relative folder path to delete
		project_dir = project.folder
		folder_path = os.path.join(project_dir, relpath)
			# check if folder exists before deleting
		if os.path.exists(folder_path):
			shutil.rmtree(folder_path)
			print(f"Deleted: {folder_path}")
			run(self.ownerComp.op('folder1').par.refreshpulse.pulse(), delayFrames = 5)
		else:
			print(f"Folder not found: {folder_path}")
		return

	def DeleteLastEntry(self):
		tables = [self.x_table, self.y_table, self.w_table]
		for table in tables:
			if table.numRows > 0:  # make sure it’s not empty
				last_row_index = table.numRows - 1
				table.deleteRow(last_row_index)
	
	def HeaderFromChannels(self, chop):
		chan_names = [c.name for c in chop.chans()]
		return chan_names

	def HeaderFromSamples(self,chop):
		names = [f's{i+1}' for i in range(chop.numSamples)]
		return names

	def destroyCallbacks(self):
		
		for ope in self.CallbacksComp.ops('*'):
			ope.destroy()

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


	# ============================================================================
	# Data Sanitization Convenience Methods
	# These methods wrap the data_sanitizer module functions for easy access
	# ============================================================================

	def _get_tables(self, table_names):
		"""
		Convert table names ('x', 'y', 'w') to actual table operators.
		
		Args:
			table_names: List of table names or 'all'
			
		Returns:
			List of table operators
		"""
		if table_names == 'all' or (isinstance(table_names, list) and 'all' in table_names):
			return [self.x_table, self.y_table, self.w_table]
		
		if not isinstance(table_names, list):
			table_names = [table_names]
		
		tables = []
		for name in table_names:
			if name == 'x':
				tables.append(self.x_table)
			elif name == 'y':
				tables.append(self.y_table)
			elif name == 'w':
				tables.append(self.w_table)
			else:
				print(f"⚠️ Unknown table name: {name}")
		
		return tables

	def FillEmptyCells(self, tables=['x', 'y'], fill_value=0.0, columns=None, preserve_header=True):
		"""
		Fill empty cells in specified tables with a given value.
		
		Args:
			tables: List of table names ('x', 'y', 'w') or 'all'
			fill_value: Value to use for filling (float or str)
			columns: Optional list of column indices to target
			preserve_header: If True, skip first row
		"""
		table_ops = self._get_tables(tables)
		return ds.fill_empty_cells(table_ops, fill_value, columns, preserve_header)

	def NormalizeTable(self, tables=['x'], columns=None, preserve_header=True):
		"""
		Normalize numeric columns to [0, 1] range using min-max scaling.
		
		Args:
			tables: List of table names ('x', 'y', 'w') or 'all'
			columns: Optional list of column indices to process
			preserve_header: If True, skip first row
		"""
		table_ops = self._get_tables(tables)
		return ds.normalize_table(table_ops, columns, preserve_header)

	def StandardizeTable(self, tables=['x'], columns=None, preserve_header=True):
		"""
		Standardize numeric columns to mean=0, std=1 using z-score.
		
		Args:
			tables: List of table names ('x', 'y', 'w') or 'all'
			columns: Optional list of column indices to process
			preserve_header: If True, skip first row
		"""
		table_ops = self._get_tables(tables)
		return ds.standardize_table(table_ops, columns, preserve_header)

	def RemoveRowsWithMissing(self, tables=['x', 'y'], threshold=1, mode='threshold', preserve_header=True):
		"""
		Remove rows where cells are empty based on configurable criteria.
		
		Args:
			tables: List of table names ('x', 'y', 'w') or 'all'
			threshold: Float (0.0-1.0) for percentage, or int for absolute count
			mode: 'any' (any missing), 'all' (all missing), or 'threshold' (percentage)
			preserve_header: If True, skip first row
		"""
		table_ops = self._get_tables(tables)
		return ds.remove_rows_with_missing(table_ops, threshold, mode, preserve_header)

	def RemoveColumnsWithMissing(self, tables=['x', 'y'], threshold=0.5, mode='threshold', preserve_header=True):
		"""
		Remove columns where cells are empty based on configurable criteria.
		
		Args:
			tables: List of table names ('x', 'y', 'w') or 'all'
			threshold: Float (0.0-1.0) for percentage, or int for absolute count
			mode: 'any' (any missing), 'all' (all missing), or 'threshold' (percentage)
			preserve_header: If True, skip first row
		"""
		table_ops = self._get_tables(tables)
		return ds.remove_columns_with_missing(table_ops, threshold, mode, preserve_header)

	def DetectOutliers(self, tables=['x'], method='iqr', threshold=1.5, columns=None, preserve_header=True):
		"""
		Detect outliers in numeric columns using IQR or Z-score method.
		
		Args:
			tables: List of table names ('x', 'y', 'w') or 'all'
			method: 'iqr' or 'zscore'
			threshold: Multiplier for IQR (default 1.5) or z-score (default 3.0)
			columns: Optional list of column indices to process
			preserve_header: If True, skip first row
		"""
		table_ops = self._get_tables(tables)
		return ds.detect_outliers(table_ops, method, threshold, columns, preserve_header)

	def HandleOutliers(self, tables=['x'], method='iqr', threshold=1.5, action='cap', columns=None, preserve_header=True):
		"""
		Detect and handle outliers by capping, removing, or logging them.
		
		Args:
			tables: List of table names ('x', 'y', 'w') or 'all'
			method: 'iqr' or 'zscore'
			threshold: Multiplier for IQR (default 1.5) or z-score (default 3.0)
			action: 'cap' (cap to bounds), 'remove' (remove rows), 'log' (log only), or 'report' (return report)
			columns: Optional list of column indices to process
			preserve_header: If True, skip first row
		"""
		table_ops = self._get_tables(tables)
		return ds.handle_outliers(table_ops, method, threshold, action, columns, preserve_header)

	def AssignWeights(self, condition='missing', weight_strategy=None, tables=['x', 'y'], default_weight=1.0, preserve_header=True):
		"""
		Assign or update weights in w_table based on conditions in source tables.
		
		Args:
			condition: 'missing', 'outliers', or 'custom'
			weight_strategy: Dict mapping conditions to weights (e.g., {'missing': 0.5, 'normal': 1.0})
			tables: List of table names to analyze for conditions
			default_weight: Default weight for rows not matching conditions
			preserve_header: If True, skip first row
		"""
		table_ops = self._get_tables(tables)
		return ds.assign_weights(self.w_table, condition, weight_strategy, table_ops, default_weight, preserve_header)

	def SanitizeData(self, tables=['x', 'y'], fill_value=0.0, remove_missing=True, normalize=False, standardize=False):
		"""
		Convenience method to apply common sanitization steps in sequence.
		
		Args:
			tables: List of table names ('x', 'y', 'w') or 'all'
			fill_value: Value to use for filling empty cells
			remove_missing: If True, remove rows with missing values
			normalize: If True, normalize numeric columns
			standardize: If True, standardize numeric columns (overrides normalize)
		"""
		results = {}
		
		# Fill empty cells
		results['fill'] = self.FillEmptyCells(tables, fill_value)
		
		# Remove rows with missing values
		if remove_missing:
			results['remove_rows'] = self.RemoveRowsWithMissing(tables, threshold=0.5, mode='threshold')
		
		# Normalize or standardize
		if standardize:
			results['standardize'] = self.StandardizeTable(tables)
		elif normalize:
			results['normalize'] = self.NormalizeTable(tables)
		
		return results

	
	# ============================================================================
	# Sanitization Callback Methods (called by ParTemplate callbacks)
	# ============================================================================

	def OnFillemptycells(self, par):
		"""Callback for Fill Empty Cells pulse parameter."""
		table_val = self.ownerComp.par.Filltables.eval()
		tables = table_val if table_val == 'all' else [table_val]
		fill_value = self.ownerComp.par.Fillvalue.eval()
		result = self.FillEmptyCells(tables=tables, fill_value=fill_value)
		print(f'Fill Empty Cells: {result["cells_filled"]} cells filled in {result["tables_processed"]} table(s)')

	def OnNormalizetable(self, par):
		"""Callback for Normalize Table pulse parameter."""
		table_val = self.ownerComp.par.Normtables.eval()
		tables = table_val if table_val == 'all' else [table_val]
		result = self.NormalizeTable(tables=tables)
		print(f'Normalize Table: Processed {result["tables_processed"]} table(s)')

	def OnStandardizetable(self, par):
		"""Callback for Standardize Table pulse parameter."""
		table_val = self.ownerComp.par.Normtables.eval()
		tables = table_val if table_val == 'all' else [table_val]
		result = self.StandardizeTable(tables=tables)
		print(f'Standardize Table: Processed {result["tables_processed"]} table(s)')

	def OnRemoverowswithmissing(self, par):
		"""Callback for Remove Rows With Missing pulse parameter."""
		table_val = self.ownerComp.par.Removetables.eval()
		tables = table_val if table_val == 'all' else [table_val]
		threshold = self.ownerComp.par.Removethreshold.eval()
		mode = self.ownerComp.par.Removemode.eval()
		result = self.RemoveRowsWithMissing(tables=tables, threshold=threshold, mode=mode)
		print(f'Remove Rows With Missing: {result["rows_removed"]} rows removed from {result["tables_processed"]} table(s)')

	def OnRemovecolumnswithmissing(self, par):
		"""Callback for Remove Columns With Missing pulse parameter."""
		table_val = self.ownerComp.par.Removetables.eval()
		tables = table_val if table_val == 'all' else [table_val]
		threshold = self.ownerComp.par.Removethreshold.eval()
		mode = self.ownerComp.par.Removemode.eval()
		result = self.RemoveColumnsWithMissing(tables=tables, threshold=threshold, mode=mode)
		print(f'Remove Columns With Missing: {result["columns_removed"]} columns removed from {result["tables_processed"]} table(s)')

	def OnDetectoutliers(self, par):
		"""Callback for Detect Outliers pulse parameter."""
		table_val = self.ownerComp.par.Outliertables.eval()
		tables = table_val if table_val == 'all' else [table_val]
		method = self.ownerComp.par.Outliermethod.eval()
		threshold = self.ownerComp.par.Outlierthreshold.eval()
		result = self.DetectOutliers(tables=tables, method=method, threshold=threshold)
		print('Outlier Detection Results:', result)

	def OnHandleoutliers(self, par):
		"""Callback for Handle Outliers pulse parameter."""
		table_val = self.ownerComp.par.Outliertables.eval()
		tables = table_val if table_val == 'all' else [table_val]
		method = self.ownerComp.par.Outliermethod.eval()
		threshold = self.ownerComp.par.Outlierthreshold.eval()
		action = self.ownerComp.par.Outlieraction.eval()
		result = self.HandleOutliers(tables=tables, method=method, threshold=threshold, action=action)
		print('Outlier Handling Results:', result)

	def OnAssignweights(self, par):
		"""Callback for Assign Weights pulse parameter."""
		condition = self.ownerComp.par.Weightcondition.eval()
		table_val = self.ownerComp.par.Weighttables.eval()
		tables = table_val if table_val == 'all' else [table_val]
		default_weight = self.ownerComp.par.Defaultweight.eval()
		weight_missing = self.ownerComp.par.Weightmissing.eval()
		weight_outliers = self.ownerComp.par.Weightoutliers.eval()
		weight_strategy = {
			'missing': weight_missing,
			'outliers': weight_outliers,
			'normal': default_weight
		}
		result = self.AssignWeights(condition=condition, weight_strategy=weight_strategy, 
								   tables=tables, default_weight=default_weight)
		print(f'Weight Assignment: {result["weights_assigned"]} weights assigned')

	def OnSanitizedata(self, par):
		"""Callback for Sanitize Data pulse parameter."""
		table_val = self.ownerComp.par.Sanitizetables.eval()
		tables = table_val if table_val == 'all' else [table_val]
		fill_value = self.ownerComp.par.Sanitizefillvalue.eval()
		remove_missing = bool(self.ownerComp.par.Sanitizeremovemissing.eval())
		normalize = bool(self.ownerComp.par.Sanitizenormalize.eval())
		standardize = bool(self.ownerComp.par.Sanitizestandardize.eval())
		result = self.SanitizeData(tables=tables, fill_value=fill_value, 
								 remove_missing=remove_missing, normalize=normalize, 
								 standardize=standardize)
		print('Sanitize Data Results:', result)

