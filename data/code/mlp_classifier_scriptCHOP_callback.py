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




# ——— TouchDesigner callbacks ———

def onSetupParameters(scriptOp):
    return


def cook(scriptOp):
    scriptOp.clear()
    scriptOp.isTimeSlice = False
    scriptOp.rate = parent().par.Samplerate
    arr = scriptOp.inputs[0].numpyArray().astype(np.float32)
    parent.ps.Predict(arr)
    scriptOp.numSamples = 1
    chan = scriptOp.appendChan('empty')
    chan[0] = 0

# ——— Optional: try loading on drop ———
