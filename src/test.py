
import numpy as np
import scipy.stats as st
  
# define sample data
gfg_data = [1, 1, 1, 2, 2, 2, 3, 3, 3,
            3, 3, 4, 4, 5, 5, 5, 6,
            7, 8, 10]
  
# create 99% confidence interval
a = st.t.interval(confidence=0.99,
              df=len(gfg_data)-1,
              loc=np.mean(gfg_data), 
              scale=st.sem(gfg_data))
print(a)