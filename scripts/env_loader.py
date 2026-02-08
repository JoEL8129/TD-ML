



import sys
import os
import platform
import importlib
from pathlib import Path  # <— add once at top





def onStart():
# replace just these lines inside your onStart():
	user = parent().par.User.eval()      # conda root, e.g. C:/Users/js812/miniconda3
	condaEnv = parent().par.Env.eval()   # env name, e.g. td-ml

	if platform.system() == 'Windows':
		base  = Path(user) / 'envs' / condaEnv
		dlls  = str(base / 'DLLs')           # Windows-native backslashes
		libbin = str(base / 'Library' / 'bin')

		os.add_dll_directory(dlls)
		os.add_dll_directory(libbin)

		env_sp = str(base / 'Lib' / 'site-packages')
			# 1) remove *all* traces of any previously loaded numpy
		for mod in list(sys.modules):
			if mod.startswith('numpy'):
				sys.modules.pop(mod)
			# 2) put your env first on the import path
		sys.path.insert(0, env_sp)
		importlib.invalidate_caches()

			# 3) now import for real
		import numpy as np
		print("NumPy version:", np.__version__)
	
	return