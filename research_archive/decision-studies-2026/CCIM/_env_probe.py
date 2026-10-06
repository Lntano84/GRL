import importlib
import platform
import sys

print("python  :", sys.version.replace("\n", " "))
print("exe     :", sys.executable)
print("platform:", platform.platform())
print()

for m in ["numpy", "scipy", "pandas", "highspy", "pulp", "mip", "ortools", "gurobipy", "matplotlib"]:
    try:
        mod = importlib.import_module(m)
        print("%-12s OK      %s" % (m, getattr(mod, "__version__", "?")))
    except Exception as e:
        print("%-12s MISSING (%s)" % (m, type(e).__name__))
