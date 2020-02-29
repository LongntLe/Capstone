# import related libraries
import numpy as np
import cvxpy as cvx 
from sklearn.linear_model import LinearRegression
import time

s = ob_df.ms.tolist()[:1000]# load data into list

# fit trend line on moving window
w = 15 # parameter for calibrating period. I.e., the bot look into the data 15 data points before the current point to project market moving in the next window.
ord_list = [0] #start with 0 orders.
# start moving window
for i in range(w, len(s)):
  start = time.time()
  x = np.array(list(range(w))).reshape((-1, 1))
  y = np.array(s[i-w:i])
  model = LinearRegression() # the bot use the data 15 seconds before the current time and fit a linear regression to project price movement in the next window
  model.fit(x, y)
  #print (model.coef_, model.intercept_)

  # create a set of projected data
  proj_s = np.array([model.coef_*i + model.intercept_ for i in range(15, 31)])
  proj_s = proj_s[1:] - proj_s[:-1]
  proj_s = proj_s.T

  # solve optimization problem on a window of time T=15
  #print (sum(ord_list))
  """
  The code block below makes use of CVXPy to solve an optimization.
  """
  # inputs:
  N = 15 # size of moving window
  max_pos = 0.1 # max position size
  
  # matrices A and B as shown in MPC model formulation
  A = np.matrix('1,1;0,0') 
  B = np.matrix('0;1')
  
  X = cvx.Variable((2,N+1)) # combination state vectors by time steps
  U = cvx.Variable((1,N)) # combination of actions vectors by time steps

  con = [X[:,0] == np.array([sum(ord_list),0])] # constraint 1: starting condition, specified to be changed based on the current inventory of the bot itself
  con.extend([X[:,1:16] == A*X[:,0:15] + B*U]) # constraint 2: update condition
  con.extend([cvx.norm(U[0,j],'inf')<=max_pos for j in range(0,N)]) # constraint 3: maximum trading position
  con.extend([cvx.norm(X[0,:],'inf')<=1]) # constraint 4: maximum inventory size

  if abs(sum(ord_list)) < 0.9:
    con.extend([sum(U[0,:]) == 0]) # constraint 5a: neutral accumulation
  else:
    #print (sum(ord_list))
    con.extend([sum(U[0,:]) == -sum(ord_list)]) # constraint 5b: exposure reduction
  obj = cvx.Maximize(sum(proj_s*X[0,1:16])) # objective function

  prob = cvx.Problem(obj, con) # insert objective function and constraints into cvxpy
  
  #output
  prob.solve() # solve using cvxpy magic
  ord_list.append(U.value[0][0]) # the bot executes only the first action, the window moves the process repeats
    
ord_list = ord_list[1:] # discard the first (zero place-holder) action

# # plot result (inventory over time)
# plt.plot(list(range(14, 30)), X.value[0])

# plot the pnl of the bot
act_s = ob_df.ms.tolist()[1:201] # data moves up 1 step to account for latency.
ord_list = [0]*15 + ord_list
ds = np.array(act_s[1:]) - np.array(act_s[:-1])
pnl = np.cumsum(np.cumsum(np.array(ord_list[:-1]))*ds)
plt.plot(pnl)

# plot inventory over time
plt.plot(np.cumsum(ord_list))

# plot price time series
plt.plot(act_s)
