import numpy as np

def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):
    scriptOp.clear()
    scriptOp.isTimeSlice = False
    arr = scriptOp.inputs[0].numpyArray().astype(np.float32)
    preds = parent.ps.Predict(arr)
    scriptOp.numSamples = preds.size
    scriptOp.rate = 60
    chan = scriptOp.appendChan('predictions')
    chan.copyNumpyArray(preds)

