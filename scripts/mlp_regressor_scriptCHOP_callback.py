import numpy as np

def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):
    scriptOp.clear()
    scriptOp.isTimeSlice = False
    
    # Check if model exists before calling Predict
    if parent.ps.Pipeline is None or parent.ps.Nfeat is None:
        # No model loaded - output zeros
        # Try to get n_outputs from Meta, default to 1
        try:
            n_outputs = parent.ps.Meta.get('n_outputs', 1) or 1
        except:
            n_outputs = 1
        preds = np.zeros((n_outputs,), dtype=np.float32)
    else:
        arr = scriptOp.inputs[0].numpyArray().astype(np.float32)
        preds = parent.ps.Predict(arr)
        # Handle case where Predict returns None (shouldn't happen if check above works, but safety)
        if preds is None:
            try:
                n_outputs = parent.ps.Meta.get('n_outputs', 1) or 1
            except:
                n_outputs = 1
            preds = np.zeros((n_outputs,), dtype=np.float32)
    
    scriptOp.numSamples = preds.size
    scriptOp.rate = 60
    chan = scriptOp.appendChan('predictions')
    chan.copyNumpyArray(preds)

