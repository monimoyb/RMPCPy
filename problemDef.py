# Define the parameters of the problem and all associated functions here.
import numpy as np
import scipy.signal
from scipy import linalg
from pytope import Polytope
import polytope as pc
import itertools
from controlpy.synthesis import controller_lqr_discrete_time as dlqr
import setOperations as sop

class ProblemParams:
	def __init__(self): 
		# Dimensions of the state and input spaces.
		self.nx = 2
		self.nu = 1

		# True dynamics matrices. Used for all LTI examples. 
		self.A = np.array([[1.0, 0.05], 
						  [0.0, 1.0]])
		self.B = np.array([[0.0],
						  [1.1]])
		
		# Nominal Dynamics matrices. Used for all LPV examples. 
		self.Anom = np.array([[1.0, 0.15], 
							  [0.1, 1.0]])
		self.Bnom = np.array([[0.1],
							  [1.1]])
		
		# Lists of DeltaA and DeltaB matrices (Draft: arxiv.org/abs/2007.00930). 
		self.epsA = 0.1
		self.epsB = 0.1

		self.delAv = np.array([ [ [0., self.epsA], [self.epsA, 0.] ],
								[ [0., self.epsA], [-self.epsA, 0.] ],
								[ [0., -self.epsA], [self.epsA, 0.] ],
								[ [0., -self.epsA], [-self.epsA, 0.] ] ])
		
		self.delBv = np.array([ [ [0.],[-self.epsB] ], 
								[ [0.],[self.epsB]], 
								[ [self.epsB], [0.] ],
								[ [-self.epsB], [0.]] ]) 

		# State constraints.
		self.Hx = np.array([[1.0, 0.0], 
							[-1.0, 0.0],
							[0.0, 1.0],
							[0.0, -1.0]])
		self.hx = np.array([[8.0], [8.0], [8.0], [8.0]])

		self.X = Polytope(self.Hx, self.hx)

		# Input constraints.
		self.Hu = np.array([[1.0], [-1.0]])
		self.hu = np.array([[4.0], [4.0]])
		
		self.U = Polytope(self.Hu, self.hu)

		# Express the constraints as Cx + Du <= b format. 
		self.C = np.vstack((self.Hx, np.zeros([self.Hu.shape[0], self.nx])))
		self.D = np.vstack( (np.zeros([self.Hx.shape[0], self.nu]), self.Hu) )
		self.b = np.vstack( (self.hx, self.hu) ) 

		# Disturbance set.
		self.wub = 0.1
		self.Hw = np.array([[1.0, 0.0], 
							[-1.0, 0.0],
							[0.0, 1.0],
							[0.0, -1.0]])
		self.hw = np.array([[self.wub], [self.wub], [self.wub], [self.wub]])
		
		self.W = Polytope(self.Hw, self.hw)

		# Stage cost weight matrices.
		self.Q = np.array([[10.0, 0.0], 
						   [0.0, 10.0]])
		self.R = 2.0

		# MPC horizon, initial condition sample size. Set by user.
		self.N = None
		self.Nx = None

		# Compute the stabilizing gain K, weight PN (LTI system).
		K,self.PN,_ = dlqr(self.A, self.B, 
						   self.Q, self.R)
		self.K = -K

		# Compute the stabilizing gain K, weight PN (LPV system).
		KvS = scipy.signal.place_poles(self.Anom, 
									   self.Bnom, 
									   np.array([0.745, 0.75]))
		self.Kv = -KvS.gain_matrix
		
		self.PNv = linalg.solve_discrete_lyapunov((self.Anom+\
				   self.Bnom@self.Kv).T, self.Q+self.R*self.Kv.T@self.Kv)
	
		# These invariant sets will be computed when required.
		self.E = None
		self.XnBar = None
		self.XnLTI = None
		self.XnLPV = None

	# Assign the horizon of the MPC problem to solve.
	def setHorizon(self, N):
		self.N = N
	
	# Assign the number of initial condition samples to use. 
	def setx0SampleCount(self, Nx):
		self.Nx = Nx

	# Generate the initial condition samples via a mesh.
	def getInitialStateMesh(self):  
		x = np.linspace(-9.0, 9.0, int(np.sqrt(self.Nx)))
		y = np.linspace(-9.0, 9.0, int(np.sqrt(self.Nx)))
		xs = np.empty([self.nx,1])

		for i in x:
			for j in y:
				xs = np.hstack((xs, np.array([[i],[j]])))

		return xs

	# Form the matrices along the horizon needed for disturbance feedback MPC.
	# Following the notations of the paper: 
	#
	# Goulart PJ, Kerrigan EC, Maciejowski JM: Optimization over state feedback 
	# policies for robust control with constraints. Automatica, vol 42, pp 523-33. 
	# 
	def formDfMatrices(self, sHorizon, sysFlag):
		if (sysFlag == "LTI"):
			Xn = self.XnLTI
			A = self.A
			B = self.B
			W = self.W
		else:
			Xn = self.XnLPV
			A = self.Anom
			B = self.Bnom
			addWBound = self.epsA*self.hx[0].item() +  \
						self.epsB*self.hu[0].item() + self.hw[0].item();  
			
			W = Polytope(lb=-addWBound*np.ones([self.nx,1]),
						 ub= addWBound*np.ones([self.nx,1]))                                                      

		N = sHorizon
		dim_t = self.C.shape[0]*N + Xn.A.shape[0]
		
		boldA = np.eye(self.nx)
		for k in range(1,N+1):
			boldA = np.vstack( (boldA, np.linalg.matrix_power(A, k) ))

		matE = np.hstack( ( np.eye(self.nx), np.zeros([self.nx, self.nx*(N-1)]) ) ) 
		boldE = np.vstack( ( np.zeros([self.nx, self.nx*N]), matE ) )

		for k in range(2,N+1):
			matE_updated = np.hstack( (np.linalg.matrix_power(A, k-1), matE[:,0:-self.nx]) )
			boldE = np.vstack( (boldE, matE_updated) )
			matE = matE_updated

		boldB = boldE @ np.kron(np.eye(N), B)
		boldC = linalg.block_diag( np.kron( np.eye(N), self.C ), Xn.A )
		szD = np.kron(np.eye(N), self.D).shape[0]
		boldD = np.vstack( ( np.kron( np.eye(N), self.D ), np.zeros([dim_t-szD, self.nu*N]) ) )

		boldF = boldC @ boldB + boldD
		boldG = boldC @ boldE
		boldH = -boldC @ boldA
		smallC = np.vstack( (np.kron(np.ones([N, 1]),self.b), Xn.b ) )

		# Matrices associated to the stacked disturbances along the horizon.
		WAstacked = np.kron(np.eye(N), W.A)
		Wbstacked = np.kron(np.ones([N, 1]), W.b)
		dim_a = WAstacked.shape[0]

		d = dict(bF=boldF, bG=boldG, bH=boldH, sc=smallC, 
				 wA=WAstacked, wb=Wbstacked, dim_t = dim_t, 
				 dim_a = dim_a)

		return d

	# Compute the minimal RPI set for error (LTI system).    
	def computeMinRobustPositiveInvariantLTI(self):
		maxIter = 100
		i = 0
		O_v = Polytope(lb = np.zeros((self.nx,1)), ub = np.zeros((self.nx,1)))
		
		while (i < maxIter): 
			O_vNext = sop.transformP(self.A + self.B @ self.K, O_v) + self.W;     

			# Check if the algorithm has covnerged.
			if (i> 0 and O_vNext == O_v):
				invariantSet = O_vNext     
				return invariantSet 

			O_v = O_vNext
			i = i + 1
		
		self.E = O_vNext

	# Compute the maximal positive invariant terminal set (Nominal system, LTI).
	def computeMaxPosInvariantLTI(self):
		# Form the dictionary needed for the precursor function.
		dMat = dict(Acl=self.A + self.B @ self.K, 
					K=self.K, 
					Hu=self.Hu, 
					hu=self.hu)

		# Setting a max bound for quitting the iterations.
		maxIter = 10
		S = self.X - self.E
		i = 0

		while (i < maxIter):
			pre = sop.preAutLTI(S, dMat)
			preIntersectS = pre & S 

			if(S == preIntersectS):
				return S
			else:
				S = preIntersectS
				i = i + 1
		
		self.XnBar = S

	# Compute the maximal robust positive invariant terminal set (LTI System).
	def computeMaxRobustPosInvariantLTI(self):
		# Form the dictionary needed for the precursor function.
		dMat = dict(Acl=self.A + self.B @ self.K, 
					K=self.K, 
					Wv=self.W.V, 
					Hu=self.Hu, 
					hu=self.hu)

		# Setting a max bound for quitting the iterations.
		maxIter = 10
		S = self.X
		i = 0

		while (i < maxIter):
			robPre = sop.robustPreAutLTI(S, dMat)
			robPreIntersectS = robPre & S

			if(S == robPreIntersectS):
				self.XnLTI = S
				break
			else:
				S = robPreIntersectS
				i = i + 1

		self.XnLTI = S

	# Compute the maximal robust positive invariant terminal set (LPV System).
	def computeMaxRobustPosInvariantLPV(self):
		# Form the dictionary needed for the precursor function.
		dMat = dict(Anom=self.Anom,
					Bnom=self.Bnom, 
					delAv=self.delAv,
					delBv=self.delBv,
					K=self.Kv, 
					Wv=self.W.V, 
					Hu=self.Hu, 
					hu=self.hu)

		# Setting a max bound for quitting the iterations.
		maxIter = 10
		S = self.X
		i = 0

		while (i < maxIter):
			robPre = sop.robustPreAutLPV(S, dMat)
			robPreIntersectS = robPre & S

			if(S == robPreIntersectS):
				self.XnLPV = S
				break
			else:
				S = robPreIntersectS
				i = i + 1
	
		self.XnLPV = S 

	# Compute the nonconvex offline bounds required for arxiv.org/abs/2007.00930.
	def computeOfflineBounds(self): 
		if self.N == 1:
			Fx = self.XnLPV.A
			t_w = np.zeros([Fx.shape[0],1])
			t_1 = t_w
			t_2 = t_w 
			t_3 = t_w
			t_delTaB = t_w

			# Return all the bounds. 
			return dict(t_1=t_1, 
						t_2=t_2, 
						t_3=t_3, 
						t_w=t_w, 
						t_delTaB=t_delTaB)
		
		# Set of all the possible A matrix vertices.
		setA = np.zeros([self.delAv.shape[0], self.nx, self.nx])
		for i in range(self.delAv.shape[0]):
			setA[i] = self.Anom + self.delAv[i]

		# Forming the boldA1Bar matrix. 
		boldA1Bar = np.zeros([self.nx*self.N, self.nx*self.N])
		for j in range(1,self.N+1):
			for k in range(1,j+1):
				tmpMat = np.linalg.matrix_power(self.Anom, j-k) if k==1 else \
						 np.hstack((tmpMat, np.linalg.matrix_power(self.Anom, j-k)))

			boldA1Bar[self.nx*(j-1): self.nx*j, 0:self.nx*j] = tmpMat 

		# Forming the boldAvBar matrix.
		tmpMat = np.zeros([self.N-1, self.nx*self.N, self.nx*self.N])

		for n in range(1,self.N):
			for j in range(1,self.N+1):
				if ((j-1)*self.nx + n*self.nx +1 <= self.nx*self.N):
					tmpMat[n-1][(j-1)*self.nx + n*self.nx: j*self.nx + n*self.nx, 
							    (j-1)*self.nx: j*self.nx] = np.eye(self.nx)
	
		for k in range(tmpMat.shape[0]):
				boldAvbar = tmpMat[k] if k==0 else np.hstack((boldAvbar, tmpMat[k]))

		# Form the tdelA and tdelB bounds.
		t_dela = float('-inf')
		t_delb = float('-inf')

		for j in range(self.delAv.shape[0]):
			t_dela = max(t_dela, 
						 np.linalg.norm(np.kron(np.eye(self.N),self.delAv[j]),np.inf))
		
		for j in range(self.delBv.shape[0]):
			t_delb = max(t_delb, 
						 np.linalg.norm(np.kron(np.eye(self.N),self.delBv[j]),np.inf)) 

		# Form the t_delTaB bound. 
		Fx = linalg.block_diag(np.kron(np.eye(self.N-1), self.X.A), self.XnLPV.A)
		t_delTaB = np.zeros([Fx.shape[0], 1])
		for row in range(Fx.shape[0]):
			t_delTaB[row] = float('-inf')
			for i in range(self.delBv.shape[0]):
				t_delTaB[row] = max(t_delTaB[row], t_delb*np.linalg.norm(Fx[row,:]@boldA1Bar@\
												   np.kron(np.eye(self.N),self.delBv[i]), 1)) 
				
		# Form all the combinatorial powers of matrices. 
		APowerMatrices = {int(1):setA}
		var = list(range(self.delAv.shape[0]))
		for i in range(2,self.N):
			lis = [p for p in itertools.product(var, repeat=i)]
			APow = np.zeros([len(lis), self.nx, self.nx])
			for j in range(len(lis)):
				APow[j] = np.eye(self.nx)
				for k in lis[j]:
					APow[j] = APow[j] @ setA[k] 
			
			APowerMatrices[int(i)] = APow

		# (N-1)-tuples of indices that are all combinations to pick from APowerMatrices[1-> N-1]. 
		listOfCombinationsIdx = []
		for i in range(1, self.N):
			tmp = APowerMatrices[int(i)]
			listOfCombinationsIdx.append(list(range(tmp.shape[0])))

		combinationMatricesIdx = list(itertools.product(*listOfCombinationsIdx))
		
		# Now form the combination (N-1)-tuples of matrices along the horizon using the above ids. 
		combinationMatrices = np.zeros([len(combinationMatricesIdx), self.nx, (self.N-1)*self.nx]) 
		for j in range(len(combinationMatricesIdx)): 
			for k in range(1,self.N):
				tmpMat = APowerMatrices[int(k)][combinationMatricesIdx[j][k-1]] if k==1 else \
						 np.hstack((tmpMat, 
									APowerMatrices[int(k)][combinationMatricesIdx[j][k-1]]))
			
			combinationMatrices[j] = tmpMat
		
		# Forming all the stacked matrices here for all the above matrix combinations. 
		delmat = np.zeros([len(combinationMatricesIdx), self.N*self.nx*(self.N-1), self.N*self.nx]) 
		for j in range(len(combinationMatricesIdx)):
			for k in range(1,self.N):
				tmpMat = np.kron(np.eye(self.N), combinationMatrices[j][:,(k-1)*self.nx: k*self.nx] \
								-np.linalg.matrix_power(self.Anom,k)) if k==1 else \
						 np.vstack((tmpMat, np.kron(np.eye(self.N), 
												 combinationMatrices[j][:,(k-1)*self.nx: k*self.nx] \
												 -np.linalg.matrix_power(self.Anom,k))))             
			
			delmat[j] = tmpMat 

		delmat_tw = np.zeros([len(combinationMatricesIdx), self.N*self.nx*(self.N-1), self.N*self.nx])
		for j in range(len(combinationMatricesIdx)):
			for k in range(1,self.N):
				tmpMat = np.kron(np.eye(self.N), 
									combinationMatrices[j][:,(k-1)*self.nx: k*self.nx]) if k==1 else \
						 np.vstack((tmpMat, np.kron(np.eye(self.N), 
												 combinationMatrices[j][:,(k-1)*self.nx: k*self.nx])))
			delmat_tw[j] = tmpMat 

		# Find the bounds by row-wise trying all vertex combinations.
		t_0 = np.zeros([Fx.shape[0], 1])
		t_1 = np.zeros([Fx.shape[0], 1])
		t_2 = np.zeros([Fx.shape[0], 1])
		t_3 = np.zeros([Fx.shape[0], 1])
		t_w = np.zeros([Fx.shape[0], 1])
		
		for row in range(Fx.shape[0]):
			t_w[row] = float('-inf')
			t_0[row] = float('-inf')
			t_3[row] = float('-inf')

			for j in range(len(combinationMatricesIdx)):
				t_w[row] = max(t_w[row], np.linalg.norm(Fx[row,:]@boldAvbar@delmat_tw[j], 1))

			for j in range(len(combinationMatricesIdx)):
				t_0[row] = max(t_0[row], np.linalg.norm(Fx[row,:]@boldAvbar@delmat[j], 1))

			t_1[row] = t_0[row]*t_dela
			t_2[row] = t_0[row]*t_delb 

			for j in range(len(combinationMatricesIdx)):
				t_3[row] = max(t_3[row], np.linalg.norm(Fx[row,:]@boldAvbar@delmat[j]@ 
										 np.kron(np.eye(self.N), self.Bnom), 1))

		# Return all the bounds. 
		return dict(t_1=t_1, 
					t_2=t_2, 
					t_3=t_3, 
					t_w=t_w, 
					t_delTaB=t_delTaB)
