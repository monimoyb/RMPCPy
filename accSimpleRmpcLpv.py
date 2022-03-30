# Code implementing "A simple robust MPC for LPV systems". 
# From the paper: 
# 
# Bujarbaruah M, Rosolia U, Stürz YR, Borrelli F. A simple robust MPC for linear 
# systems with parametric and additive uncertainty. American Control Conference, 
# 2021, pp. 2108-2113, IEEE.
# 
# This code does not do closed-loop MPC. It generates an inner approx. of the ROA.  
import numpy as np
import problemDef as pdef
import polytope as pc 
from solveMPC import solveMpcAccSimpleRobust
import matplotlib.pyplot as plt

# Load the parameters of the problem here.
params = pdef.ProblemParams()

# Assign the horizon of the MPC.
params.setHorizon(5)

# Assign the number of initial condition samples to use.
params.setx0SampleCount(576)

# Compute the terminal set Xn.	
params.computeMaxRobustPosInvariantLPV()

# Compute the big matrices needed for all horizons in distFeedback.  
dictOfMatricesDf = []
for i in range(1,params.N+1):
	dictOfMatricesDf.append(params.formDfMatrices(i, "LPV"))

# Solve MPC from a bunch of sampled initial conditions. 
# Store feasible initial conditions as ROA samples. 
xs = params.getInitialStateMesh() 
xFeas = np.zeros([1, params.nx])

# The matrices required for horizon 1 problem.
boldAbar = params.Anom
boldBbar = params.Bnom
Fx = params.XnLPV.A
fx = params.XnLPV.b
boldHw = params.W.A
boldhw = params.W.b
boldHu = params.U.A
boldhu = params.U.b

dictofMatricesH1 = dict(boldAbar=boldAbar,
				  	  	boldBbar=boldBbar,
					  	Fx=Fx,
				      	fx=fx,
				      	boldHw=boldHw,
				      	boldhw=boldhw,
				      	boldHu=boldHu,
				      	boldhu=boldhu)

for j in range(params.Nx):
	solverSuccessFlag = False
	for i in range(1,params.N+1):
		solverSuccessFlag += solveMpcAccSimpleRobust(xs[:,j], 
													 params, 
													 dictOfMatricesDf[i-1], 
													 dictofMatricesH1,
													 i)

	# If feasible, add to the ROA sample collection set. 
	if (solverSuccessFlag == True):
		xFeas = np.vstack((xFeas, xs[:,j].T))

if (xFeas.shape[0] == 1):
	print("Nothing Feasible!")
else:
	# Finally form the approximate ROA.
	approxRoa = pc.qhull(xFeas)

	# Plot a non-empty returned approx. ROA.
	if (approxRoa.b.shape[0] !=0):
		fig = plt.figure()
		ax = fig.gca()
		approxRoa.plot(ax)
		ax.relim()
		ax.autoscale_view()
		plt.grid(True)
		plt.show()
	else:
		print("CVX hull not full dimensional. Increase Nx.")
	