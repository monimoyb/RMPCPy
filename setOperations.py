# Define all the set operations here. 
from pytope import Polytope 
import polytope as pc
import numpy as np
import matplotlib.pyplot as plt

# Function to compute the matrix transformation of a polytope.
def transformP(M, P):
	pVertices = P.V
	transformedVertices = M@pVertices[0]

	for i in range(pVertices.shape[0]):
		ans = np.vstack((transformedVertices, M@pVertices[i]))
		transformedVertices = ans

	transformedPolytope = Polytope(transformedVertices)
	return transformedPolytope

# Function to compute the Minkowski sum of two polytopes.
def minkowskiSum(P1, P2):
	V_sum = []
	V1 = P1.V
	V2 = P2.V

	for i in range(V1.shape[0]):
		for j in range(V2.shape[0]):
			V_sum.append(V1[i,:] + V2[j,:])
	
	return Polytope(np.asarray(V_sum))

# Function to compute the Pontryagin difference of two polytopes.
def pontryaginDifference(P1, P2):
	Px = P1.A
	px = P1.b
	p2Vertices = P2.V
	hMax = np.zeros([px.size, 1])

	# Compute the max values row-wise.
	for i in range(px.size):
		maxVal = float('-inf')
		# iterate through P2 vertices.
		for j in range(p2Vertices.shape[0]):
			maxVal = max(maxVal, Px[i,:] @ p2Vertices[j, :])

		hMax[i] = maxVal

	pontryaginDifferencePolytope = Polytope(Px, px - hMax)
	return pontryaginDifferencePolytope

# Function to compute the robust precursor set (LTI system).
def robustPreAutLTI(S, dMat):
	# Unpack the dictionary. 
	Acl = dMat["Acl"] 
	K = dMat["K"]
	Wv = dMat["Wv"]
	Hu = dMat["Hu"]
	hu = dMat["hu"]

	# Initialize.
	H = S.A
	h = S.b	
	hTight = np.zeros([h.size, 1])

	# Compute the tightenings row-wise.
	for i in range(h.size):
		minTightening = float('inf')
		# iterate through W vertices.
		for j in range(Wv.shape[0]):
			minTightening = min(minTightening, h[i]- H[i,:] @ Wv[j, :])

		hTight[i] = minTightening

	prePolytope = Polytope(H@Acl, hTight) & Polytope(Hu@K, hu) 

	return prePolytope

# Function to compute the robust precursor set (LPV system).
def robustPreAutLPV(S, dMat):
	# Unpack the dictionary. 
	Anom = dMat["Anom"]
	Bnom = dMat["Bnom"] 
	delAv = dMat["delAv"]
	delBv = dMat["delBv"]
	K = dMat["K"]
	Wv = dMat["Wv"]
	Hu = dMat["Hu"]
	hu = dMat["hu"]

	# Form all the closed-loop matrices' options. 
	nx = delAv.shape[1]
	clMat = np.zeros([delAv.shape[0]*delBv.shape[0], nx, nx])
	count = 0
	for i in range(delAv.shape[0]):
		for j in range(delBv.shape[0]):
			clMat[count] = (Anom + delAv[i]) + (Bnom+delBv[j]) @ K
			count +=1

	# Initialize.
	H = S.A
	h = S.b	

	# Compute the tightenings row-wise for each closed-loop matrix. 
	for k in range(clMat.shape[0]):
		hTight = np.zeros([h.size, 1])
		for i in range(h.size):
			minTightening = float('inf')
			# iterate through W vertices.
			for j in range(Wv.shape[0]):
				minTightening = min(minTightening, h[i]- H[i,:] @ Wv[j, :])

			hTight[i] = minTightening

		prePolytope = Polytope(H@clMat[k], hTight) & Polytope(Hu@K, hu) 

		# Do the intersections for all models.
		prePolytopeLPV = prePolytope if k==0 else prePolytopeLPV & prePolytope


	return prePolytopeLPV

# Function to compute the nominal precursor set (LTI).
def preAutLTI(S, dMat):
	# Unpack the dictionary.
	Acl = dMat["Acl"] 
	K = dMat["K"]
	Hu = dMat["Hu"]
	hu = dMat["hu"]	

	# Initialize.
	H = S.A
	h = S.b	

	prePolytope = Polytope(H@Acl, h) & Polytope(Hu@K, hu) 

	return prePolytope
