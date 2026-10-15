"""Match sklearn float32 input / double threshold comparison explicitly."""
import math
def predict(model,x):
    probabilities=[]
    for tree in model['trees']:
        node=0
        while tree['left'][node]!=-1:
            node=tree['left'][node] if float(x[tree['feature'][node]])<=float(tree['threshold'][node]) else tree['right'][node]
        probabilities.append(tree['positive'][node])
    return math.fsum(probabilities)/len(probabilities)
def outside(model,x):
    return any(float(x[i])<lo-1e-6 or float(x[i])>hi+1e-6 for i,lo,hi in model['support'])
def decide(model,x):
    p=predict(model,x);ood=outside(model,x)
    return bool(ood or p>=model['threshold']),dict(probability=p,threshold=model['threshold'],outside_support=ood)
