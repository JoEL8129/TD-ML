# me - this DAT
# par - the Par object that has changed
# val - the current value
# prev - the previous value
# Make sure the corresponding toggle is enabled in the Parameter Execute DAT.
import umap
from umap import UMAP
import pandas as pd
import ast
import joblib
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
#reducer = None

def onValueChange(par, prev):
	# use par.eval() to get current value
	return

# The changes are a list of named tuples, where each tuple is (Par, previous value)
def onValuesChanged(changes):
	for c in changes:
		# use par.eval() to get current value
		par = c.par
		prev = c.prev
	return
def onPulse(par):
    # 1) Grab the table
    #global reducer
    table = op('null1')  # replace with your DAT path
    data = []
    for row in table.rows():  # skip header if any
        data.append([float(c.val) for c in row])
    df = pd.DataFrame(data)
    #df2 = pd.DataFrame(data)


    # 2) Build params dict from your params DAT
    paramDAT = op('params')
    params = {}
    for row in paramDAT.rows()[1:]:  # skip header
        key = row[0].val            # e.g. 'n_components'
        raw = row[1].val            # e.g. '3', '0.25', '(15,)', '42'
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            val = raw
        params[key] = val

    
    #pipeline = make_pipeline(
        #PCA(n_components=20),
    #    UMAP(**params),
    #)

    #pipeline.fit(df2.values)
    #emb_full = pipeline.transform(df2.values)
    #emb_full = pipeline.fit_transform(df2.values)

    # 3) Fit UMAP
    reducer = umap.UMAP(**params)
    
    emb = reducer.fit_transform(df.values)
    joblib.dump(reducer,'umap_pca_pipeline.joblib')

    fill_components_dat(emb, parent().par.Ncomponents, 'umap2D_DAT')

    # 6) Trigger downstream pulse
    #parent().par.Generatecolor.pulse()
    op('script1_callbacks').module.load_models()
    return



def fill_components_dat(emb, n, dat_op='umap2D_DAT'):
    """
    Write embedding points into a DAT with dynamic headers based on n components.
    
    Args:
        emb (iterable of sequences): each element is a sequence of length >= n
        n (int): number of components/columns to output
        dat_op (str or OP): name or OP reference of the DAT to write into
    """
    if n == 2:
        header = ['x', 'y']
    elif n == 3:
        header = ['x', 'y', 'z']
    else:
        # for n > 3: first 3 are x,y,z, then c4…cN
        header = ['x', 'y', 'z'] + [f'c{i+1}' for i in range(3, n)]
    
    # resolve the DAT operator
    out = op(dat_op)
    if not out:
        raise RuntimeError(f"DAT '{dat_op}' not found")
    
    # clear existing contents and write header
    out.clear()
    out.appendRow(header)

    # write each point, slicing or padding to exactly n entries
    for point in emb:
        row = list(point)[:n]
        if len(row) < n:
            row += [''] * (n - len(row))
        out.appendRow(row)

