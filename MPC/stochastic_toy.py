# import related libraries
import numpy as np
import cvxpy as cvx 
from sklearn.linear_model import LinearRegression
import time

w = 15 # parameter for calibrating period. I.e., the bot look into the data 15 data points before the current point to project market moving in the next window.

s = ob_df.ms.tolist()[:1000]# load data into list
act_s = ob_df.ms.tolist()[1:1001] # data moves up 1 step to account for latency.
ds = np.array(act_s[1:]) - np.array(act_s[:-1])

# fit trend line on moving window
ord_list = [0] #start with 0 orders.
# start moving window
for i in range(w, len(s)):
    start = time.time()
    x = np.array(list(range(w))).reshape((-1, 1))
    y = np.array(s[i-w:i])
    model = LinearRegression() # the bot use the data 15 seconds before the current time and fit a linear regression to project price movement in the next window
    model.fit(x, y)
    ds_t = ds[i-w:i-1]
    #print (model.coef_, model.intercept_)
    tmp_ord_list = []
    print (np.mean(ds_t), np.std(ds_t))

  # create a set of projected data
    projs = []
    for _ in range(50):
        # create projected time series
        nxt_inc = np.random.normal(loc=np.mean(ds_t), scale=np.std(ds_t), size=w+1)
        proj_s = nxt_inc + np.array([model.coef_[0]*i + model.intercept_ for i in range(w, 2*w+1)])
        # process time series for analysis later
        proj_s_org = proj_s
        proj_s_org = proj_s_org.T
        proj_s = proj_s[1:] - proj_s[:-1]
        proj_s = proj_s.T

        # solve optimization problem on a window of time T=15
        """
        The code block below makes use of CVXPy to solve an optimization.
        """
        # inputs:
        N = 15 # size of moving window
        max_pos = 1 # max position size

        # matrices A and B as shown in MPC model formulation
        A = np.matrix('1,1;0,0') 
        B = np.matrix('0;1')

        X = cvx.Variable((2,N+1)) # combination state vectors by time steps
        U = cvx.Variable((1,N)) # combination of actions vectors by time steps

        con = [X[:,0] == np.array([sum(ord_list),0])] # constraint 1: starting condition, specified to be changed based on the current inventory of the bot itself
        con.extend([X[:,1:w+1] == A*X[:,0:w] + B*U]) # constraint 2: update condition
        con.extend([cvx.norm(U[0,j],'inf')<=max_pos for j in range(0,N)]) # constraint 3: maximum trading position
        con.extend([cvx.norm(X[0,:],'inf')<=1]) # constraint 4: maximum inventory size

        con.extend([X[0, -1] == 0]) #neutral exposure
      
        obj = cvx.Maximize(X[0,1:w+1]@proj_s-cvx.tv(cvx.diag(proj_s_org[1:]*X[0,1:w+1])))

        prob = cvx.Problem(obj, con) # insert objective function and constraints into cvxpy

        #output
        prob.solve() # solve using cvxpy magic
        tmp_ord_list.append(U.value[0][0])
        
    v = np.mean(tmp_ord_list) # record the mean of all optimal controls
    #print (v)

    ord_list.append(v) # the bot executes only the first action, the window moves the process repeats
    
ord_list = ord_list[1:] # discard the first (zero place-holder) action

# plot the pnl of the bot
ord_list = [0]*w + ord_list
pnl = np.cumsum(np.cumsum(np.array(ord_list[:-1]))*ds)

fig = plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(pnl)

# plot inventory over time
plt.subplot(1,3,2)
plt.plot(np.cumsum(ord_list))

# plot price time series
plt.subplot(1,3,3)
plt.plot(act_s)
