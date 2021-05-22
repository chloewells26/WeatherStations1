
import pandas as pd
import numpy as np
df1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df2 = pd.DataFrame(data=df1, columns=['a', 'b', 'c'])

print(df2)