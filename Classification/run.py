import joblib
import sys
import numpy as np
filename = 'model.sav'
loaded_model = joblib.load(filename)


data_point = np.loadtxt(sys.stdin)

if (data_point.shape == (2352,)):
     data = [data_point]
     sys.stdout.write(str(int(loaded_model.predict(data))))
else:
     for d in data_point:
          data = [d]
          sys.stdout.write(str(int(loaded_model.predict(data))))
