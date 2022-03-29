# Code implementing the robust MPC with optimization-based constraint tightenings.
# For LPV systems; from the preprint: arxiv.org/abs/2007.00930. 
# This code does not do closed-loop MPC. It generates an inner approx. of the ROA.  
import numpy as np
from scipy import linalg
import sys
import problemDef as pdef
import polytope as pc 
from solveMPC import solveRobustMpcOptimalTightening
import matplotlib.pyplot as plt

# Load the parameters of the problem here.
params = pdef.ProblemParams()

# Assign the horizon of the MPC.
params.setHorizon(3)

# If the horizon is larger than N=4, terminate. 
# For this case I suggest using accSimpleRobust.py instead. 
if params.N > 4:
	print("Lower horizon to N<=4.")
	sys.exit()

# Assign the number of initial condition samples to use. 
params.setx0SampleCount(576)

# Compute the terminal set Xn.	
params.computeMaxRobustPosInvariantLPV()

# Compute the bounds needed. 
dictofBounds = params.computeOfflineBounds()

# The matrices required for the method.
if (params.N > 1):
	boldA1Bar = np.zeros([params.nx*params.N, params.nx*params.N])
	for j in range(1,params.N+1):
		for k in range(1,j+1):
			tmpMat = np.linalg.matrix_power(params.Anom, j-k) if k==1 else \
			   np.hstack((tmpMat, np.linalg.matrix_power(params.Anom, j-k)))

		boldA1Bar[params.nx*(j-1): params.nx*j, 0:params.nx*j] = tmpMat 

	boldAbar = np.kron(np.eye(params.N), params.Anom)
	boldBbar = np.kron(np.eye(params.N), params.Bnom)
	Fx = linalg.block_diag(np.kron(np.eye(params.N-1), params.X.A), 
								   params.XnLPV.A)
	fx = np.vstack((np.kron(np.ones([params.N-1, 1]), params.X.b), 
					params.XnLPV.b))
	boldHw = np.kron(np.eye(params.N), params.W.A)
	boldhw = np.kron(np.ones([params.N,1]), params.W.b)
	boldHu = np.kron(np.eye(params.N), params.U.A)
	boldhu = np.kron(np.ones([params.N,1]), params.U.b) 
else:
	boldAbar = params.Anom
	boldBbar = params.Bnom
	Fx = params.XnLPV.A
	fx = params.XnLPV.b
	boldHw = params.W.A
	boldhw = params.W.b
	boldHu = params.U.A
	boldhu = params.U.b

dictofMatrices = dict(boldA1Bar=boldA1Bar, 
				  	  boldAbar=boldAbar,
					  boldBbar=boldBbar,
					  Fx=Fx,
					  fx=fx,
					  boldHw=boldHw,
					  boldhw=boldhw,
					  boldHu=boldHu,
					  boldhu=boldhu)

# Solve MPC from a bunch of sampled initial conditions. 
# Store feasible initial conditions as ROA samples. 
xs = params.getInitialStateMesh() 
xFeas = np.zeros([1, params.nx])

for i in range(params.Nx):
	solverSuccessFlag = solveRobustMpcOptimalTightening(xs[:,i], 
														params, 
														dictofBounds,
														dictofMatrices)

	# If feasible, add to the ROA sample collection set. 
	if (solverSuccessFlag == True):
		xFeas = np.vstack((xFeas, xs[:,i].T))

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
