# Code implementing the disturbance feedback based robust MPC for LTI systems.
# This code does not do closed-loop MPC. It generates an inner approx. of the ROA. 
import numpy as np
import problemDef as pdef
import polytope as pc 
from solveMPC import solveMpcDisturbanceFeedback
import matplotlib.pyplot as plt

# Load the parameters of the problem here.
params = pdef.ProblemParams()

# Assign the horizon of the MPC.
params.setHorizon(5)

# Assign the number of initial condition samples to use. 
params.setx0SampleCount(576)

# Compute the terminal set Xn.	
params.computeMaxRobustPosInvariantLTI()

# Compute the big matrices needed. 
dictOfMatrices = params.formDfMatrices(params.N, "LTI")

# Solve MPC from a bunch of sampled initial conditions. 
# Store feasible initial conditions as ROA samples. 
xs = params.getInitialStateMesh() 
xFeas = np.zeros([1, params.nx])

for i in range(params.Nx):
	solverSuccessFlag = solveMpcDisturbanceFeedback(xs[:,i], 
													params, 
													dictOfMatrices,
													params.N,
													"LTI")

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
