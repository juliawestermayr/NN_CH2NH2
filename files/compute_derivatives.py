# -*- coding: utf-8 -*-
import numpy as np

#================================
# Compute derivative of NN wrt xyz
#================================

def compute_dNNdxyz( dNNdD, dD_dxyz, scaling_factors, n_atoms ):
    # get number of states
    #formula: dE(NN)/dG * dG/dxyz --> dNNdD*descriptors + E(NN)* dG/dxyz
    n_states = dNNdD.shape[0]
    # get number of atoms (from x derivatives of descriptors, which are N by N
    #n_atoms = dD_dxyz[0].shape[0]
    # construct Cartesian derivatives for every state
    all_dNNdxyz = []
    for state in range( n_states ):
        # reconstruct matrix from upper triangle
        curr_dNNdD = np.zeros( (n_atoms,n_atoms) )
        curr_dNNdD[np.triu_indices(n_atoms,1)] = dNNdD[state] / scaling_factors[1]* scaling_factors[3][state]
        curr_dNNdD = curr_dNNdD + curr_dNNdD.T
        curr_dNNdxyz = np.zeros( (n_atoms,3) )
        for xyz in range( 3 ):
            curr_dNNdxyz[:,xyz] = np.sum( curr_dNNdD*dD_dxyz[xyz], axis=1 )
        all_dNNdxyz.append( curr_dNNdxyz )
    return all_dNNdxyz

#================================
# Compute derivative of descriptors wrt xyz
#================================

def get_dD_dxyz( xyz_coords, descriptors, n_atoms ):
    dD_dx = np.zeros( (n_atoms,n_atoms) )
    dD_dy = np.zeros( (n_atoms,n_atoms) )
    dD_dz = np.zeros( (n_atoms,n_atoms) )
    D = np.zeros( (n_atoms,n_atoms) )
    # Get distance vectors in x, y and z components
    for i in range( n_atoms ):
        for j in range( i+1, n_atoms ):
            xyz_ij = xyz_coords[i] - xyz_coords[j]
            dD_dx[i,j] =  xyz_ij[0] #dD_dx[0,1] for 2-atom molecule
            dD_dx[j,i] = -xyz_ij[0] #dD_dx[1,0] 
            dD_dy[i,j] =  xyz_ij[1]
            dD_dy[j,i] = -xyz_ij[1]
            dD_dz[i,j] =  xyz_ij[2]
            dD_dz[j,i] = -xyz_ij[2]
            D[i,j] = 1.0 / np.linalg.norm( xyz_ij )
            D[j,i] = D[i,j]
    D_3 = D * D * D
    dD_dx = -D_3 * dD_dx
    dD_dy = -D_3 * dD_dy
    dD_dz = -D_3 * dD_dz
    dD_dxyz = ( dD_dx, dD_dy, dD_dz ) #tensor with 3 times (for each coordinate) the matrix (natomsxnatoms)
    return dD_dxyz

#================================
# Compute derivative of descriptors wrt xyz for Coulomb Matrix
#================================

def get_dD_dxyz_coulomb( xyz_coords, descriptors, n_atoms,charges ):
    dD_dx = np.zeros( (n_atoms,n_atoms) )
    dD_dy = np.zeros( (n_atoms,n_atoms) )
    dD_dz = np.zeros( (n_atoms,n_atoms) )
    D = np.zeros( (n_atoms,n_atoms) )
    # Get distance vectors in x, y and z components
    for i in range( n_atoms ):
        for j in range( i+1, n_atoms ):
            xyz_ij = xyz_coords[i] - xyz_coords[j]
            dD_dx[i,j] =  xyz_ij[0] #dD_dx[0,1] for 2-atom molecule
            dD_dx[j,i] = -xyz_ij[0] #dD_dx[1,0] 
            dD_dy[i,j] =  xyz_ij[1]
            dD_dy[j,i] = -xyz_ij[1]
            dD_dz[i,j] =  xyz_ij[2]
            dD_dz[j,i] = -xyz_ij[2]
            D[i,j] = (charges[i]*charges[j]) / np.linalg.norm( xyz_ij )
            D[j,i] = D[i,j]
    for i in range(n_atoms):
        D[i,i] = 0.5*charges[i]**(2.4)
    # reconstruct coulomb matrix from vector of
    # upper triangle
    # Construct full matrix of derivatives
    D_3 = D * D * D
    dD_dx = -D_3 * dD_dx
    dD_dy = -D_3 * dD_dy
    dD_dz = -D_3 * dD_dz
    dD_dxyz = ( dD_dx, dD_dy, dD_dz ) #tensor with 3 times (for each coordinate) the matrix (natomsxnatoms)
    #fill the diagonal with constants again
    return dD_dxyz

#================================
# Compute derivative of NN wrt xyz
#================================

def compute_dNNdxyz_coulomb( dNNdD, dD_dxyz, scaling_factors, n_atoms ):
    # get number of states
    #formula: dE(NN)/dG * dG/dxyz --> dNNdD*descriptors + E(NN)* dG/dxyz
    n_states = dNNdD.shape[0]
    # get number of atoms (from x derivatives of descriptors, which are N by N
    # construct Cartesian derivatives for every state
    all_dNNdxyz = []
    for state in range( n_states ):
        # reconstruct matrix from upper triangle
        curr_dNNdD = np.zeros( (n_atoms,n_atoms) )
        curr_dNNdD[np.triu_indices(n_atoms,k=0)] = dNNdD[state] / scaling_factors[1]* scaling_factors[3][state]
        curr_dNNdD = curr_dNNdD + curr_dNNdD.T
        #now we have the diagonal twice - divide by 2
        for i in range(n_atoms):
            curr_dNNdD[i][i] = curr_dNNdD[i][i]/2
        curr_dNNdxyz = np.zeros( (n_atoms,3) )
        for xyz in range( 3 ):
            curr_dNNdxyz[:,xyz] = np.sum( curr_dNNdD*dD_dxyz[xyz], axis=1 )
        all_dNNdxyz.append( curr_dNNdxyz )
    return all_dNNdxyz

#
#===================================================================================
#==================================HESSIAN FROM NNs=================================
#===================================================================================


def get_hessian(nmstates,n_atoms,state,dNNdD,ddNNdDdD,scaling_factors,xyz_coords,dim):

    #shape descriptor
    dNNdD = dNNdD / scaling_factors[1] * scaling_factors[3][state]
    #shape Descriptor x Descriptor
    for idim in range(dim):
      for jdim in range(dim):
        ddNNdDdD[idim][jdim] = ddNNdDdD[idim][jdim] / scaling_factors[1][idim] /scaling_factors[1][jdim] * scaling_factors[3][state]

    #FIRST PART = d²NN_dD²*dD_dxyz*dD_cxyz
    first_part = compute_firstpart_hessian(dNNdD,n_atoms,xyz_coords[0],dim)

    #SECOND PART = dNN/dD*d²D/d(xyz)²
    second_part = compute_secondpart_hessian(xyz_coords[0],n_atoms,ddNNdDdD,dim)

    #add both parts
    Hessian=np.add(first_part,second_part)
    return Hessian

def compute_firstpart_hessian(dNNdD,n_atoms,xyz_coords,dim):
    #compute second derivative of descriptors wrt xyz
    second_derivative = np.zeros((n_atoms*3,n_atoms*3,dim))
    index = -1
    for i in range( n_atoms ):
        for j in range( i+1, n_atoms ):
            index+=1
            xyz_ij = (xyz_coords[i] - xyz_coords[j])
            D = np.linalg.norm( xyz_ij )
            #derivative wrt to xi xi ...
            xx = 3*xyz_ij[0]**2/(D**5)-1/(D**3)
            yy = 3*xyz_ij[1]**2/(D**5)-1/(D**3)
            zz = 3*xyz_ij[2]**2/(D**5)-1/(D**3)
            xy = 3*xyz_ij[0]*xyz_ij[1]/(D**5)
            xz = 3*xyz_ij[0]*xyz_ij[2]/(D**5)
            yz = 3*xyz_ij[1]*xyz_ij[2]/(D**5)

            #same terms
            second_derivative[i*3+0][i*3+0][index] = xx
            second_derivative[i*3+1][i*3+1][index] = yy
            second_derivative[i*3+2][i*3+2][index] = zz

            second_derivative[j*3+0][j*3+0][index] = xx
            second_derivative[j*3+1][j*3+1][index] = yy
            second_derivative[j*3+2][j*3+2][index] = zz

            second_derivative[i*3+0][i*3+1][index] = xy
            second_derivative[i*3+0][i*3+2][index] = xz
            second_derivative[i*3+1][i*3+0][index] = xy
            second_derivative[i*3+1][i*3+2][index] = yz
            second_derivative[i*3+2][i*3+0][index] = xz
            second_derivative[i*3+2][i*3+1][index] = yz

            second_derivative[j*3+0][j*3+1][index] = xy
            second_derivative[j*3+0][j*3+2][index] = xz
            second_derivative[j*3+1][j*3+0][index] = xy
            second_derivative[j*3+1][j*3+2][index] = yz
            second_derivative[j*3+2][j*3+0][index] = xz
            second_derivative[j*3+2][j*3+1][index] = yz

            #mixed terms                          
            second_derivative[i*3+0][j*3+0][index] = -xx
            second_derivative[i*3+1][j*3+1][index] = -yy
            second_derivative[i*3+2][j*3+2][index] = -zz

            second_derivative[j*3+0][i*3+0][index] = -xx
            second_derivative[j*3+1][i*3+1][index] = -yy
            second_derivative[j*3+2][i*3+2][index] = -zz

            second_derivative[i*3+0][j*3+1][index] = -xy
            second_derivative[i*3+0][j*3+2][index] = -xz
            second_derivative[i*3+1][j*3+0][index] = -xy
            second_derivative[i*3+1][j*3+2][index] = -yz
            second_derivative[i*3+2][j*3+0][index] = -xz
            second_derivative[i*3+2][j*3+1][index] = -yz

            second_derivative[j*3+0][i*3+1][index] = -xy
            second_derivative[j*3+0][i*3+2][index] = -xz
            second_derivative[j*3+1][i*3+0][index] = -xy
            second_derivative[j*3+1][i*3+2][index] = -yz
            second_derivative[j*3+2][i*3+0][index] = -xz
            second_derivative[j*3+2][i*3+1][index] = -yz
    first_part = np.zeros((n_atoms*3,n_atoms*3))
    for i in range(n_atoms*3):
      for j in range(n_atoms*3):
        first_part[i][j] = np.dot(dNNdD[0],second_derivative[i][j])

    return first_part


def compute_secondpart_hessian( xyz_coords, n_atoms, ddNNdDdD,dim ):
    #compute first derivative of descriptors wrt xyz - dimension: natoms*3 x number_of_descriptors
    index=-1
    Descriptor=np.zeros((dim))
    first_derivative = np.zeros((n_atoms*3,dim))
    for i in range( n_atoms ):
        for j in range( i+1, n_atoms ):
            index+=1
            xyz_ij = (xyz_coords[i] - xyz_coords[j])
            Descriptor[index] = np.linalg.norm(xyz_ij)
            first_derivative[i*3+0][index]=-1/(Descriptor[index]**3)*xyz_ij[0]
            first_derivative[i*3+1][index]=-1/(Descriptor[index]**3)*xyz_ij[1]
            first_derivative[i*3+2][index]=-1/(Descriptor[index]**3)*xyz_ij[2]
            first_derivative[j*3+0][index]=-1/(Descriptor[index]**3)*-xyz_ij[0]
            first_derivative[j*3+1][index]=-1/(Descriptor[index]**3)*-xyz_ij[1]
            first_derivative[j*3+2][index]=-1/(Descriptor[index]**3)*-xyz_ij[2]
    D2 = np.matmul(first_derivative,ddNNdDdD)
    D3 = np.matmul(D2,first_derivative.T)
    return D3




#===========================================================================
#======================HOPPING DIRECTION FROM DELTA HESISAN=================
#===========================================================================


def get_hoppingdirection(Hessian,nmstates,n_atoms):

  #store hessian
  #write_hessian(Hessian)
  #get delta hessian between states
  delta_hessian = get_delta_Hessian(Hessian,nmstates,n_atoms)

  #get hopping direction for states from delta hessian 
  hopping_direction = get_hopping_direction(delta_hessian,nmstates,n_atoms)

  return hopping_direction


def get_delta_Hessian(Hessian,nmstates,n_atoms):

    delta_Hessian=np.zeros((nmstates,nmstates,n_atoms*3,n_atoms*3))

    for i in range(nmstates):
      for j in range(nmstates):
        delta_Hessian[i][j] = Hessian[i]-Hessian[j]

    return delta_Hessian

def get_hopping_direction(delta_Hessian,nmstates,n_atoms):

    #finds the hopping direction via singular value decomposition of each matrix from delta_HEssian

    hopping_direction = np.zeros((nmstates,nmstates,n_atoms*3))
    for i in range(nmstates):
        for j in range(i+1,nmstates):
            current_Hessian = np.zeros((n_atoms*3,n_atoms*3))
            current_Hessian[:][:] = delta_Hessian[i][j][:][:]
            #singular value decomposition
            u,s,vh=np.linalg.svd(current_Hessian)
            #eval,evec = np.linalg.eig(current_Hessian)
            #take  vh -u  is the same but transpose differently ordered (atom1x, atom2x,atom1y,... and vh is atom1x, atom1y,...)
            hopping_direction[i][j] = vh[0]
            hopping_direction[j][i] = -vh[0]
    return hopping_direction


#======================================
#STORE HESSIAN IN PICKLE AND NUMPY FILE

def write_hessian(hessian):

    try:
        pf=open("hessian_nn",'wb')
    except IOError:
        print("Could not write to 'all_weights.pkl'.")
        exit()

    np.save("hessian_nn_numpy",hessian)
