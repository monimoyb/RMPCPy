# Functions to solve the associated robust MPC problems at any state xt.
from sys import path
# This path to be modified by the user.
path.append(r"</Users/monimoybujarbaruah>/casadi-py27-v3.4.4")
from casadi import *
from scipy import linalg
import setOperations as sop
import matplotlib.pyplot as plt

# MPC for shrinking tube (LTI system).
def solveMpcShrinkingTube(xt, params):	
	# Form the closed-loop matrix.
	Acl = params.A + params.B @ params.K

	# Optimization variables.
	opti = casadi.Opti()
	x = opti.variable(params.nx, params.N+1)
	u = opti.variable(params.nu, params.N)	
	
	# Initial cost and constraints.
	nominalCost = 0.
	opti.subject_to( x[:,0] == xt)

	# MPC problem over a horizon of N.
	for i in range(params.N):

		# Dynamics of the nominal states.
		opti.subject_to( x[:,i+1] == Acl @ x[:,i] + params.B @ u[:,i])

		# Impose the nominal constraints.
		if (i == 0):
			nomimalStateConstraintSet = params.X
			nomimalInputConstraintSet = params.U
		elif (i == 1):
			stateConstraintTightening = params.W 
			inputConstraintTightening = sop.transformP(params.K, params.W) 
			nomimalStateConstraintSet = params.X - stateConstraintTightening
			# Using my own Pontryagin difference function for a 1D input case. 			
			nomimalInputConstraintSet = sop.pontryaginDifference(params.U, 
																 inputConstraintTightening)
		else:
			stateConstraintTightening = stateConstraintTightening + \
										sop.transformP(np.linalg.matrix_power(Acl,i-1),
													   						  params.W)
			# Using my own Minkowski sum function for a 1D input case.
			inputConstraintTightening = sop.minkowskiSum(inputConstraintTightening, 
														 sop.transformP(params.K @ \
														 np.linalg.matrix_power(Acl,i-1), 
														 						params.W))
			nomimalStateConstraintSet = params.X - stateConstraintTightening
			# Using my own Pontryagin difference function for a 1D input case.
			nomimalInputConstraintSet = sop.pontryaginDifference(params.U, 
																 inputConstraintTightening)

		opti.subject_to( nomimalStateConstraintSet.A @ x[:,i] 
										 <= nomimalStateConstraintSet.b )
		opti.subject_to( nomimalInputConstraintSet.A @ (params.K @ x[:,i] + u[:,i]) 
										 <= nomimalInputConstraintSet.b )

		# Update the cost.
		nominalCost += x[:,i].T @ params.Q @ x[:,i] + u[:,i].T @ params.R @ u[:,i]

	# Include the terminal ingredients.
	terminalTightening = stateConstraintTightening + \
						 sop.transformP(np.linalg.matrix_power(Acl, params.N-1), params.W)
	nominalTerminalConstraintSet = params.XnLTI - terminalTightening
	
	opti.subject_to( nominalTerminalConstraintSet.A @ x[:,params.N] 
									 <= nominalTerminalConstraintSet.b )
	nominalCost += x[:,params.N].T @ params.PN @ x[:,params.N] 

	# Solve the MPC problem.
	opti.minimize(nominalCost)
	opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
	opti.solver('ipopt', opts)
	
	try:
		sol = opti.solve()
	except:
		return False

	# Return the MPC feasibility flag.
	return sol.stats()['return_status'] == 'Solve_Succeeded'

# MPC for rigid tube (LTI system).
def solveMpcRigidTube(xt, params):	
	# Form the closed-loop matrix. 
	Acl = params.A + params.B @ params.K

	# Optimization variables.
	opti = casadi.Opti()
	x = opti.variable(params.nx, params.N+1)
	u = opti.variable(params.nu, params.N)	

	# Initial cost and constraint.
	nominalCost = 0.
	opti.subject_to(params.E.A @ (xt-x[:,0]) <= params.E.b)
	
	# Form the tightened constraints. 
	nomimalStateConstraintSet = params.X - params.E
	# Using my own Pontryagin difference function for a 1D input case.
	nomimalInputConstraintSet = sop.pontryaginDifference(params.U, 
														 sop.transformP(params.K,params.E))

	# MPC problem over a horizon of N.
	for i in range(params.N):
		# Dynamics of the nominal states.
		opti.subject_to( x[:,i+1] == Acl @ x[:,i] + params.B @ u[:,i])

		# Impose the nominal constraints.
		opti.subject_to( nomimalStateConstraintSet.A @ x[:,i] 
										 <= nomimalStateConstraintSet.b )
		opti.subject_to( nomimalInputConstraintSet.A @ (params.K @ x[:,i] + u[:,i]) 
										 <= nomimalInputConstraintSet.b )

		# Update the cost.
		nominalCost += x[:,i].T @ params.Q @ x[:,i] + u[:,i].T @ params.R @ u[:,i]

	# Include the terminal ingredients.	
	opti.subject_to( params.XnBar.A @ x[:,params.N] <= params.XnBar.b )
	nominalCost += x[:,params.N].T @ params.PN @ x[:,params.N] 

	# Solve the MPC problem.
	opti.minimize(nominalCost)
	opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
	opti.solver('ipopt', opts)

	try:
		sol = opti.solve()
	except:
		return False

	# Return the MPC feasibility flag.
	return sol.stats()['return_status'] == 'Solve_Succeeded'

# MPC with disturbance feedback policy parametrization.
def solveMpcDisturbanceFeedback(xt, 
								params, 
								dictOfMatrices, 
								sHorizon, 
								sysFlag):

	boldF = dictOfMatrices["bF"]
	boldG = dictOfMatrices["bG"]
	boldH = dictOfMatrices["bH"]
	smallC = dictOfMatrices["sc"]
	WAstacked = dictOfMatrices["wA"]
	Wbstacked = dictOfMatrices["wb"]
	dim_t = dictOfMatrices["dim_t"]
	dim_a = dictOfMatrices["dim_a"]

	# Parse the system and check horizon size.
	N = sHorizon
	if (sysFlag=="LTI"):
		A = params.A
		B = params.B
		PN = params.PN
	else:
		A = params.Anom
		B = params.Bnom
		PN = params.PNv

	# Optimization variables.
	opti = casadi.Opti()
	
	# Policy parametrization matrix.
	M = opti.variable(params.nu*N, params.nx*N)
	for j in range(params.nu*N):
		for k in range(2*j,params.nx*N):
			opti.subject_to( M[j,k] == 0.0)
	
	# Nominal states, inputs, and the dual variables.
	x = opti.variable(params.nx, N+1)
	v = opti.variable(params.nu*N, 1)
	Z = opti.variable(dim_a, dim_t)

	# Initial cost and constraint. 
	opti.subject_to(x[:,0] == xt.reshape(-1,1))
	opti.subject_to(params.X.A @ x[:,0] <= params.X.b)
	nominalCost = 0.
	
	# Stage cost.
	for k in range(N):
		x[:,k+1] = A @ x[:, k] + B @ v[k*params.nu:(k+1)*params.nu]
		nominalCost += x[:,k].T @ params.Q @ x[:,k] \
					 + v[k*params.nu:(k+1)*params.nu].T @ params.R @ v[k*params.nu:(k+1)*params.nu]
	
	# Include the terminal cost.
	nominalCost += x[:,N].T @ PN @ x[:,N] 

	# Robust state and input constraints. 
	opti.subject_to(boldF @ v + Z.T @ Wbstacked <= smallC + boldH @ xt.reshape(-1,1))
	opti.subject_to(vec(Z)>=0.)
	opti.subject_to(boldF@M + boldG == Z.T @ WAstacked)

	# Solve the MPC problem.
	opti.minimize(nominalCost)
	opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
	opti.solver('ipopt', opts)
	
	try:
		sol = opti.solve()
	except:
		return False

	# Return the MPC feasibility flag.
	return sol.stats()['return_status'] == 'Solve_Succeeded'

# A simple robust MPC for LPV systems. ACC 2021 (Bujarbaruah M., Rosolia R., Sturz Y., Borrelli F.).
def solveMpcAccSimpleRobust(xt, 
							params, 
							dictofDFMatrices,
							dictofMatricesH1,
							sHorizon):

	if (sHorizon > 1):
		return solveMpcDisturbanceFeedback(xt, 
										   params, 
										   dictofDFMatrices,
										   sHorizon,
										   "LPV")
	
	# The matrices required. 
	boldAbar = dictofMatricesH1["boldAbar"]
	boldBbar = dictofMatricesH1["boldBbar"]
	Fx = dictofMatricesH1["Fx"]
	fx = dictofMatricesH1["fx"]
	boldHw = dictofMatricesH1["boldHw"]
	boldhw = dictofMatricesH1["boldhw"]
	boldHu = dictofMatricesH1["boldHu"]
	boldhu = dictofMatricesH1["boldhu"]
	
	# Variables and constraints.
	opti = casadi.Opti()
	x = opti.variable(params.nx, 2)
	v = opti.variable(params.nu,1)
	Lambda = opti.variable(Fx.shape[0], boldHw.shape[0]) 
	
	opti.subject_to(x[:,0] == xt)
	opti.subject_to(vec(Lambda)>=0.)
	opti.subject_to(boldHu@v <= boldhu)                       
	
	# Enumerate the set of vertices here. 
	for i in range(params.delAv.shape[0]):
		bolddelA = params.delAv[i]
		for j in range(params.delBv.shape[0]): 
			bolddelB = params.delBv[j]
			opti.subject_to(Fx@((boldAbar+bolddelA)@xt +  
						   (boldBbar+bolddelB)@v) + Lambda@boldhw <= fx)

	opti.subject_to(Lambda@boldHw == Fx)  

	# Solve the MPC problem.
	nominalCost = v.T @ params.R @ v + x[:,1].T @ params.PNv @ x[:,1]
	opti.minimize(nominalCost)
	opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
	opti.solver('ipopt', opts)

	try:
		sol = opti.solve()
	except:
		return False

	# Return the MPC feasibility flag.
	return sol.stats()['return_status'] == 'Solve_Succeeded'  

# Robust MPC for LPV systems with optimization based constraint tightening. 
# From draft: arxiv.org/abs/2007.00930. 
def solveRobustMpcOptimalTightening(xt, 
									params, 
									dictofBounds,
									dictofMatrices):
	N = params.N

	if (N ==1):
		return solveMpcAccSimpleRobust(xt, params, {}, dictofMatrices, N)
	
	# Bounds computed offline.  
	t_1 = dictofBounds["t_1"]
	t_2 = dictofBounds["t_2"]
	t_3 = dictofBounds["t_3"]
	t_w = dictofBounds["t_w"]
	t_delTaB = dictofBounds["t_delTaB"]

	# The matrices required. 
	boldA1Bar = dictofMatrices["boldA1Bar"]
	boldAbar = dictofMatrices["boldAbar"]
	boldBbar = dictofMatrices["boldBbar"]
	Fx = dictofMatrices["Fx"]
	fx = dictofMatrices["fx"]
	boldHw = dictofMatrices["boldHw"]
	boldhw = dictofMatrices["boldhw"]
	boldHu = dictofMatrices["boldHu"]
	boldhu = dictofMatrices["boldhu"]

	# Variables and constraints. 
	opti = casadi.Opti()

	# Nominal state and inputs.
	x = opti.variable(params.nx*(N+1), 1)
	v = opti.variable(params.nu*N,1)
	
	# Policy and dual matrices. 
	Lambda = opti.variable(Fx.shape[0], boldHw.shape[0])   
	M = opti.variable(params.nu*N, params.nx*N)
	for j in range(params.nu*N):
		for k in range(2*j,params.nx*N):
			opti.subject_to(M[j,k] == 0.0)
	gamma = opti.variable(boldHw.shape[0], boldHu.shape[0])   

	opti.subject_to(vec(Lambda)>=0.)
	
	# Input constraints.
	opti.subject_to(vec(gamma)>=0.)
	opti.subject_to(gamma.T@boldhw <= boldhu - boldHu@v)
	opti.subject_to((boldHu@M).T == boldHw.T@gamma)   

	# Nominal dynamics evolution. 
	opti.subject_to(x[0:params.nx] == xt.reshape(-1,1))
	for k in range(1,N+1):
		x[k*params.nx:(k+1)*params.nx] = params.Anom@x[(k-1)*params.nx:k*params.nx] +\
										 params.Bnom@v[(k-1)*params.nu:k*params.nu]

	# These are done to extract the ininity norms of these opti vectors/matrices. 
	epsilon_v = opti.variable(1,1)
	opti.subject_to(epsilon_v >= 0.)
	
	# This is for ||v||_inf.
	for i in range(v.shape[0]):
		# Bounding abs. of each entry of the vector.
		opti.subject_to(-epsilon_v <= v[i]) 
		opti.subject_to(v[i] <= epsilon_v)
	
	# This is for ||x||_inf.
	epsilon_x = opti.variable(1,1)
	opti.subject_to(epsilon_x >= 0.)
	xNoTermS = x[0:-params.nx]
	for i in range(xNoTermS.shape[0]):
		# Bounding abs. of each entry of the vector.
		opti.subject_to(-epsilon_x  <= xNoTermS[i]) 
		opti.subject_to(xNoTermS[i] <= epsilon_x)
	
	# This is for ||M||_inf, with matrix M. 
	epsilon_Mij = opti.variable(params.nu*N, params.nx*N)
	epsilon_M = opti.variable(1,1)
	opti.subject_to(vec(epsilon_Mij)>= 0.)
	opti.subject_to(epsilon_M>=0.)
	
	# Bounding abs. of each entry of the rows and summing.
	rSumVec = opti.variable(M.shape[0], 1)
	for i in range(M.shape[0]):
		rSum = 0.
		for j in range(M.shape[1]):
			opti.subject_to(-epsilon_Mij[i,j] <= M[i,j])
			opti.subject_to(M[i,j] <= epsilon_Mij[i,j])
			rSum += epsilon_Mij[i,j]
		opti.subject_to(rSumVec[i] == rSum) 
	
	# Bounding the row sum.
	opti.subject_to(rSumVec <= epsilon_M*np.ones([M.shape[0], 1]))

	# Verex enumeration here for two terms linear in model mismatches.
	for i in range(params.delAv.shape[0]):
		bolddelA = np.kron(np.eye(N),params.delAv[i])
		for j in range(params.delBv.shape[0]): 
			bolddelB = np.kron(np.eye(N),params.delBv[j])
			opti.subject_to(Fx @ boldAbar @ xNoTermS + 
							Fx @ boldBbar @ v + 
							Fx @ boldA1Bar @ bolddelA @ xNoTermS + 
							Fx @ boldA1Bar @ bolddelB @ v +
							t_1*epsilon_x + 
							(t_2+t_delTaB)*(epsilon_M*params.wub) + 
							t_2*epsilon_v + 
							t_3*epsilon_M*params.wub + 
							t_w*params.wub + 
							Lambda@boldhw <= fx)

	# Dual state constraints.
	opti.subject_to(Fx@boldBbar@M + Fx@(boldA1Bar@boldBbar - boldBbar)@M + 
					Fx@np.eye(boldA1Bar.shape[0]) == Lambda@boldHw)

	# Stage cost and the terminal cost. 
	nominalCost  = x.T @ linalg.block_diag(np.kron(np.eye(N),params.Q), params.PNv) @ x \
				  + v.T @ np.kron(np.eye(N), params.R) @ v 
	
	# Solve the MPC problem. 
	opti.minimize(nominalCost)
	opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
	opti.solver('ipopt', opts)

	try:
		sol = opti.solve()
	except:
		return False

	# Return the MPC feasibility flag. 
	return sol.stats()['return_status'] == 'Solve_Succeeded'  	
