# -*- coding: utf-8 -*-
# Interactive script for the setup of dynamics calculations for SHARC
# 
# usage: python setup_traj.py

import copy
import math
import os
import readline
import sys
import itertools
import contextlib
import theano
import theano.tensor as T
import numpy as np
import pickle
import fcntl
import time
from collections import defaultdict
#============================================================================
#   READ XYZ FILE FOR GEOMETRIES
#============================================================================


def get_asize(mols,pad):
    asize = defaultdict()

    for mol in mols:
        for key, value in mol.natypes.items():
            try:
                asize[key] = max(asize[key], value + pad)
            except KeyError:
                asize[key] = value + pad
    #without the +1 because we have the charge +1 
    size = max(mol.nuclear_charges.size for mol in mols) 
    return asize, size

def get_descriptors_coulomb( sharc_dat, rng, train_frac,n_molecules ):
    """get descriptor matrices by processing data from SHARC-xyz-output, return list of elements in every upper triangular descriptor matrix
    PARAMETERS:
    sharc_xyz: "output.xyz" produced by SHARC
    train_frac: fraction of training set, e.g. 0.9

    FUNCTION:
    1.) read data from xyz-file
    2.) get descriptor matrices
    3.) transform every upper triangular descriptor matrix to list
    4.) return list of all lists from 3.)
    """
    des="descriptors"
    # read atypes, geometries from "output.xyz"
    dict_properties = read_output_dat(sharc_dat)
    dict_properties.update({'Coulomb': True})
    #os.system('python /user/julia/SHARC/bin/get_xyz.py %s' %sharc_dat)
    #cwd=os.getcwd()
    path = '/user/julia/BaselVisit/Methylenimmonium/dataset/xyz-files'
    all_atypes = dict_properties["AllAtomTypes"]
    all_geoms = dict_properties["AllGeometries"]
    natoms = dict_properties["NumberOfAtoms"]
    all_molecules = dict_properties["Stepsize"]
    dict_indices = {"Index"	: False }
    # calculate descriptor matrices from xyz-geometries
    descriptors = []
    # transform every upper triangular descriptor matrix to list, append to list of descriptors for every state
    mols = []
    for i in range(all_molecules):
      filename = "%s/%04d.xyz" %(path,i+1)
      mol = Compound(filename)
      mols.append(mol)
    asize,size = get_asize(mols,1)
    for i in range(all_molecules):
      #coulomb: dimension 21 for 6 atoms - diagonal matrix + diagonal
      mols[i].generate_coulomb_matrix(size,sorting="row-norm")
      #bob (dimension 45 for 6 atoms)
      #mols[i].generate_bob(size=size,asize=asize)
      descriptors.append(np.asarray(mols[i].representation))
    descriptors = np.array(descriptors)
    dict_properties.update( {'Charges': mols[0].nuclear_charges})
    # Shuffle indices
    all_indices = dict_properties["Step"]
    steplen=all_indices.shape[0]
    steplen=int(steplen)
    indices_matrix=np.zeros((n_molecules,int(2)))
    n_train = int( np.floor( n_molecules*train_frac ) )
    rng.shuffle( all_indices )
    train_indices= all_indices[:n_train]
    valid_indices=all_indices[(n_train):n_molecules]
    t=int(0)
    v=int(1)
    for i in range(len(train_indices)):
      indices_matrix[i]=train_indices[i],t
    for j in range(len(valid_indices)):
      indices_matrix[len(train_indices)+j]=valid_indices[j],v
    #indices_matrix=np.array(indices_matrix)
    dict_indices["Index"]=indices_matrix
    indices=indices_matrix
    train_descriptors, valid_descriptors, descriptors_mean, descriptors_std, zeroindex = scale(des, descriptors, train_frac, n_molecules, train_indices, valid_indices )
       # Load data into theano shared variables
    train_descriptors = theano.shared( np.asarray(train_descriptors,dtype=theano.config.floatX), borrow=True )
    valid_descriptors = theano.shared( np.asarray(valid_descriptors,dtype=theano.config.floatX), borrow=True )
    # return list of descriptors for every state
      #input dimension:
    input_dim = descriptors.shape[1]
    return descriptors, dict_indices, dict_properties, input_dim, train_descriptors, valid_descriptors, all_indices, valid_indices, train_indices, n_molecules, descriptors_mean, descriptors_std, zeroindex

def transform_coordinates( all_atypes, all_geoms, n_molecules, natoms ):
    """originally from "transform_coords.py"
       get descriptor matrices
    """
    all_dmats = []
#    n_molecules = len( all_atypes )/natoms
   # print "n_molecules: ", n_molecules
    for i in range( n_molecules ):
        elemental_indices = index_elements( all_atypes )
        #print "allatypes", all_atypes
        #print "all geom i", all_geoms[i]
        dist_mat = compute_dist_mat( all_geoms[i] )
        # Mask diagonal for inversion
        np.fill_diagonal( dist_mat, 1.0 )
        print(dist_mat)
        dist_mat = 1.0 / dist_mat
        np.fill_diagonal( dist_mat, 0.0 )
        all_dmats.append( dist_mat )
        print(dist_mat)
        #print all_dmats, "all_dmats", all_dmats[i]
    return all_dmats


def compute_dist_mat( geom ):
    """originally from "transform_coords.py"
       Fast distance matrix computation in numpy
    """
    m, n = geom.shape
    G = np.dot( geom, geom.T )
    H = np.tile( np.diag( G ), (m,1) )
    return np.sqrt( H + H.T - 2*G )
    #return H + H.T - 2*G

def index_elements( all_atypes ):
    """Get indices where atoms of a certain element can be
       found in a molecule.
       Return them in the form of a dict, with the keys being
       nuclear charge and the entry being a numpy array of the indices.
    """
    elemental_indices = {}
    #print all_atypes
    all_atypes = np.array(all_atypes)
    #print all_atypes
   # print np.unique(all_atypes), "element"
    for element in np.unique( all_atypes ):
        elemental_indices[element] = np.where( all_atypes == element )
    #print elemental_indices, "elemental indices"
    return elemental_indices



def get_training( quantity, dict_properties, rng, n_molecules, train_indices, valid_indices, train_frac, controlfile = None ):
    """ get energies (diagonal element of Hamiltonian) from SHARC-dat-output, scale them and return all necessary data for NN-training
    PARAMETERS:
    descriptors: list of descriptors as obtained by get_descriptors
    sharc_dat: "output.dat" produced by SHARC
    train_frac: fraction of training set, e.g. 0.9
    controlfile: if as controlfile is given in inputfile, unscaled descriptors and energies are written to this file

    FUNCTION:
    1.) read data from .dat-file, determine input and output dimensions
    2.) if parameter controlfile is given: write unscaled energies and unscaled descriptors to controlfile_energy
    3.) split and scale energies and descriptors
    4.) load scaled energies and descriptors into Theano shared variables
    5.) return data = ((train_descriptors, train_energies),(valid_descriptors, valid_energies)), input_dim, output_dim, scaling_factors
    """
    """if controlfile:
      try:
        out = open( controlfile + "_quantity.dat", 'w' )
      except IOError:
        print( "Could not create file %s." % controlfile )
        exit()
        for i in range(step):
          out.write( "Quantity %s     " % ( ' '.join(["%f " % q for q in all_quantity[i] ] )))
          out.write( "Descriptors %s \n" % ( ' '.join(["%f " % d for d in descriptors[i] ] )))
        out.close()"""
      # Scale energies    
    if quantity == "energy":
      current_quantity = dict_properties["Energy"]
      train_current_quantity, valid_current_quantity, current_quantity_mean, current_quantity_std, zeroindex = scale( quantity, current_quantity, train_frac, n_molecules, train_indices, valid_indices )
      #print "Energy scaled"
      #print( "ESTD : %s " % (' '.join("%14.8f" % quantity for quantity in current_quantity_std)))
      #print( "EMEAN: %s " % (' '.join("%14.8f" % quantity for quantity in current_quantity_mean)))
    # Load data into theano shared variables
      train_current_quantity    = theano.shared( np.asarray(train_current_quantity,dtype=theano.config.floatX), borrow=True )
      valid_current_quantity    = theano.shared( np.asarray(valid_current_quantity,dtype=theano.config.floatX), borrow=True )
      output_dim = current_quantity.shape[1]
      #print "Quantity", current_quantity, "output_Dim:", output_dim
    elif quantity == "soc":
      current_quantity = dict_properties["SpinOrbitCoupling"]
      train_current_quantity, valid_current_quantity, current_quantity_mean, current_quantity_std, zeroindex = scale( quantity, current_quantity, train_frac, n_molecules, train_indices, valid_indices )
      #print "SOC scaled"
      #print( "SocSTD : %s " % (' '.join("%14.8f" % quantity for quantity in current_quantity_std)))
      #print( "SocMEAN: %s " % (' '.join("%14.8f" % quantity for quantity in current_quantity_mean)))
    # Load data into theano shared variables
      train_current_quantity    = theano.shared( np.asarray(train_current_quantity,dtype=theano.config.floatX), borrow=True )
      valid_current_quantity    = theano.shared( np.asarray(valid_current_quantity,dtype=theano.config.floatX), borrow=True )
      output_dim = current_quantity.shape[1]
      #print "Quantity", current_quantity, "output_Dim:", output_dim
    elif quantity == "dipole":
      current_quantity = dict_properties["Dipole"]
      train_current_quantity, valid_current_quantity, current_quantity_mean, current_quantity_std, zeroindex = scale( quantity, current_quantity, train_frac, n_molecules, train_indices, valid_indices )
      #print "Dipole scaled"
      #print( "DipoleSTD : %s " % (' '.join("%14.8f" % quantity for quantity in current_quantity_std)))
      #print( "DipoleMEAN: %s " % (' '.join("%14.8f" % quantity for quantity in current_quantity_mean)))
    # Load data into theano shared variables
      train_current_quantity    = theano.shared( np.asarray(train_current_quantity,dtype=theano.config.floatX), borrow=True )
      valid_current_quantity    = theano.shared( np.asarray(valid_current_quantity,dtype=theano.config.floatX), borrow=True )
      output_dim = current_quantity.shape[1]
      #print "Quantity", current_quantity, "output_Dim:", output_dim
    elif quantity == "grad":
      current_quantity = dict_properties["Gradient"]
      train_current_quantity, valid_current_quantity, current_quantity_mean, current_quantity_std, zeroindex = scale( quantity, current_quantity, train_frac, n_molecules, train_indices, valid_indices )
      #print "Gradient scaled"
      #print( "GradSTD : %s " % (' '.join("%14.8f" % quantity for quantity in current_quantity_std)))
      #print( "GradMEAN: %s " % (' '.join("%14.8f" % quantity for quantity in current_quantity_mean)))
    # Load data into theano shared variables
      train_current_quantity    = theano.shared( np.asarray(train_current_quantity,dtype=theano.config.floatX), borrow=True )
      valid_current_quantity    = theano.shared( np.asarray(valid_current_quantity,dtype=theano.config.floatX), borrow=True )
      output_dim = current_quantity.shape[1]
      #print "Quantity", current_quantity, "output_Dim:", output_dim
    elif quantity == "nac":
      current_quantity = dict_properties["NonAdiabaticCoupling"]
      train_current_quantity, valid_current_quantity, current_quantity_mean, current_quantity_std, zeroindex = scale( quantity, current_quantity, train_frac, n_molecules, train_indices, valid_indices)
      #print "Nac scaled"
      #print( "NacSTD : %s " % (' '.join("%14.8f" % quantity for quantity in current_quantity_std)))
      #print( "NacMEAN: %s " % (' '.join("%14.8f" % quantity for quantity in current_quantity_mean)))
    # Load data into theano shared variables
      train_current_quantity    = theano.shared( np.asarray(train_current_quantity,dtype=theano.config.floatX), borrow=True )
      valid_current_quantity    = theano.shared( np.asarray(valid_current_quantity,dtype=theano.config.floatX), borrow=True )
      output_dim = current_quantity.shape[1]
    elif quantity == "nac_grad":
      current_quantity = dict_properties["NonAdiabaticCoupling"]
      train_current_quantity, valid_current_quantity, current_quantity_mean, current_quantity_std, zeroindex = scale( quantity, current_quantity, train_frac, n_molecules, train_indices, valid_indices)
      train_current_quantity    = theano.shared( np.asarray(train_current_quantity,dtype=theano.config.floatX), borrow=True )
      valid_current_quantity    = theano.shared( np.asarray(valid_current_quantity,dtype=theano.config.floatX), borrow=True )
      output_dim = current_quantity.shape[1]
    elif quantity == "energy_grad":
      current_quantity = dict_properties["Energy"]
      train_current_quantity, valid_current_quantity, current_quantity_mean, current_quantity_std, zeroindex = scale( quantity, current_quantity, train_frac, n_molecules, train_indices, valid_indices )
      train_current_quantity    = theano.shared( np.asarray(train_current_quantity,dtype=theano.config.floatX), borrow=True )
      valid_current_quantity    = theano.shared( np.asarray(valid_current_quantity,dtype=theano.config.floatX), borrow=True )
      output_dim = current_quantity.shape[1]
    return output_dim, current_quantity_mean, current_quantity_std, train_current_quantity, valid_current_quantity, zeroindex

def scale( quantity, data, train_frac, n_molecules, train_indices, valid_indices):
    """ split given data into training and validation set, scale and return scaled data and scaling factors
    PARAMETERS:
    data: unscaled training data, e.g. matrices of quantities
    steps: n of steps in "output.dat" and "output.xyz" - steps determined in class output_dat
    train_frac: fraction of training set, e.g. 0.9

    FUNCTION:
    1.) split data in training and validation set
    2.) calculate mean and std of training data
    3.) scale training and validation data: ( data - mean )/std
    4.) return scaled data & scaling factors
    """

    # get training and validation indices
    # split data in training and validation set
    for i in range(len(train_indices)):
      train_data = data[train_indices]
    for i in range(len(valid_indices)):
      valid_data = data[valid_indices]
    # calculate mean and standard
    data_mean = np.mean( train_data, axis=0 )# - np.mean( train_descriptors, axis=0 )
    data_std  = np.std( train_data, axis=0 ) #/ np.std( train_descriptors, axis=0 )
    # If standard deviation should be smaller than certain threshold
    # set it to 1
    # zeroindices contains numbers, at which position zeros are in data_std
    zeroindex = np.where( data_std < 1.0e-12 )[0]
    data_std[zeroindex] = 1.0

    # scale data
    train_data = ( train_data - data_mean ) / data_std
    valid_data = ( valid_data - data_mean ) / data_std

    return train_data, valid_data, data_mean, data_std, zeroindex

def write_zeroindices( zeroindices, quantity ):
    zero=open("zeroindices_%s" %(quantity), "w")
    zero.write( "%s" %( zeroindices[:] ) )
    zero.close()

def read_output_dat(sharc_dat):
    """ used class: output_dat
        dictionary containing the matrices with necessary values of the quantities: dict_read - if real and imaginary values are available, they are separated into two matrices
        Hamiltonian matrix: real values: 'Hamiltonian_real', imaginary values: 'Hamiltonian_imag'
                            the diagonal elements contain the energy (only real values)
                            the off-diagonal elements are the SOCs (real+imaginary parts)
        Dipole matrices: only real values, no imagrinary ones; x,y, and z part separated, called: Dipole_x, Dipole_y, Dipole_z, respectively
    all_energy only consists of the real values of the energy = diagnoal elements of the Hamiltonian matrix
     (in dictionary: read_dict of class output_dat called 'Hamiltonian_real')
    #all_soc consists the upper triangular matrix of the Hamiltonian containing the real and imaginary values of the SOCs
    #all_dipole consists of the upper triangular matrices including the diagonal elements of the real values of the dipole matrices - the first line always contains the values of dipole_x, the second the values of dipole_y and the third the values of dipole_z
    """
    dict_properties = { "Step"              : False,
                        "Energy"            : False,
                        "SpinOrbitCoupling"	: False,
                        "Dipole"            : False,
                        "NonAdiabaticCoupling"	: False,
			"Gradient"		: False,
			"Index"			: False }
    y=output_dat(sharc_dat)
    ezero = y.ezero
    all_atype = y.all_atype
    nmstates = y.nmstates
    natoms = y.natoms
    n_singlets = y.states[0]
    n_triplets = y.states[2]
    stepsize = len(y.startlines)
    all_energy=[]
    soc_numberoftriangular=3*(n_triplets*(n_triplets-1))/2
    soc_numberoftriangular=int(soc_numberoftriangular)
    all_soc=np.zeros((stepsize,n_singlets*n_triplets*3+soc_numberoftriangular))
    all_soc_real=[]
    all_soc_imag=[]
    all_dipole_x=[]
    all_dipole_y=[]
    all_dipole_z=[]
    all_grad = []
    all_geom = []
    all_nac = []
    all_step=np.arange(stepsize)
    dipole_numberofmatrix=3*((n_singlets*(n_singlets-1)/2+n_singlets)+(n_triplets*(n_triplets-1)/2+n_triplets))
    dipole_numberofmatrix=int(dipole_numberofmatrix)
    all_dipole=np.zeros((stepsize,dipole_numberofmatrix))
    dipole_range=all_dipole.shape[1]/3
    #iterates over steps in output.dat file, every step is a new row in the desired matrices for the quantities: all_quantity
    for step,read_dict in y:
      all_geom.append(read_dict["Geometries"])
      #step arrays are overwritten after every iteration over one step and written to the matrices all_quantity
      energy_step=[]
      soc_step_real = []
      soc_step_imag = []
      dipole_step_x = []
      dipole_step_y = []
      dipole_step_z = []
      grad_step = []
      nac_step = []
      #all_energy: contains the diagonal elements of the Hamiltonian matrix for each time step in a row
      #all_soc: only the upper triangular matrix is taken into account
      #         for socs there are 6 variables: a, b, c, d, e and f 
      #         a+ib are the couplings between S and T(ms=-1, ms=+1)
      #         ic are the couplings between S and T(ms=0)
      #         e+if are the couplings between T(ms=-1,ms=+1) and T(ms=0)
      #         id are the couplings between T(ms=-1,ms=+1) and T with the same ms
      #         Tx(-1)/Ty(+1) = 0, Tx(0)/yT(0) = 0, Tx/Tx = 0
      #all_dipole: dipole matrices (for x y and z coordinate) are symmetric, thus the upper triangular matrix including the diagonal elements are used for training.
      #            the singlets do not couple with the triplets, moreover the triplets only couple with same multiplicity - the other values are 0 and T1(ms-1)/T2(ms-1)=T1(ms0)/T2(ms0)=T1(ms+1)/T2(ms+1)
      #all_grad: dimension: step x state*3*natoms - contains in the first row the first step: with state((atom1:x,y,z)(atom2:x,y,z))
      #all_nac: dimension: step x state*state*natoms*3 - contains in the first row the first step with: state1 coupling with state1 
      #         for the first atom (x,y, and z values), following the second atom (x,y, and z values),...
      #         state 1 coupling with state 2 for the firs atom and second atom,...

      #because the triplets all have the same energy and are written e.g. for 2 triplets in the way: T1 (ms -1), T2 (ms -1), T1 (ms 0), T2 (ms 0), T1 (ms+1), T2(ms+2); only the first entry of each triplet has to be trained
      for singlets in range(n_singlets):
        energy_step.append(read_dict['Hamiltonian_real'][singlets][singlets])
      for triplets in range(n_singlets,n_singlets+n_triplets):
        energy_step.append(read_dict['Hamiltonian_real'][triplets][triplets])
      all_energy.append(energy_step)

      #get the dipole moments of singlets
      for singlet in range(n_singlets):
        for upper_triangular in range(singlet,n_singlets):
          dipole_step_x.append(read_dict['Dipole_x'][singlet][upper_triangular])
          dipole_step_y.append(read_dict['Dipole_y'][singlet][upper_triangular])
          dipole_step_z.append(read_dict['Dipole_z'][singlet][upper_triangular])
      #get the dipole moments of triplets
      for triplet in range(n_singlets,n_singlets+n_triplets):
        for upper_triangular in range(triplet,n_triplets+n_singlets):
          dipole_step_x.append(read_dict['Dipole_x'][triplet][upper_triangular])
          dipole_step_y.append(read_dict['Dipole_y'][triplet][upper_triangular])
          dipole_step_z.append(read_dict['Dipole_z'][triplet][upper_triangular])
      all_dipole_x.append(dipole_step_x)
      all_dipole_y.append(dipole_step_y)
      all_dipole_z.append(dipole_step_z)

      for soc_state in range(n_singlets):
        #get a
        for a in range(n_singlets,n_singlets+n_triplets):
          soc_step_real.append(read_dict['Hamiltonian_real'][soc_state][a])
        #get b and c
        for b_c in range(n_singlets,n_singlets+n_triplets*2):
          soc_step_imag.append(read_dict['Hamiltonian_imag'][soc_state][b_c])
          #get e
      for soc_state in range(n_singlets,n_singlets+n_triplets-1):
        for e in range(soc_state+n_triplets+1,n_singlets+2*n_triplets):
          soc_step_real.append(read_dict['Hamiltonian_real'][soc_state][e])
        #get d and f
        for d in range(soc_state+1,n_singlets+n_triplets):
          soc_step_imag.append(read_dict['Hamiltonian_imag'][soc_state][d])
        for f in range(soc_state+n_triplets+1,n_singlets+n_triplets*2):
          soc_step_imag.append(read_dict['Hamiltonian_imag'][soc_state][f])

      all_soc_real.append(soc_step_real)
      all_soc_imag.append(soc_step_imag)
      for grad_state in range(natoms*(n_singlets+n_triplets)):
        for g in range(3):
          grad_step.append(read_dict['Gradient'][grad_state][g])
      all_grad.append(grad_step)
      #get all singlet-singlet nacs without the first (S0 with S0 - this is 0)
      #if those are extracted, get all Singlet couplings with the S1
      #couplings for triplets with different ms are the same, except for ms=+1 - it differs in sign
      singlet=0
      while True:
        if singlet<n_singlets:
          for nac_state_singlet in range(natoms*(singlet+1),natoms*n_singlets):
            for n in range(3):
              nac_step.append(read_dict['NonAdiabaticCoupling'][nac_state_singlet+singlet*natoms*nmstates][n])
          singlet+=1
        else:
          break
      triplet=0
      while True:
        if triplet<n_triplets:
          #write couplings with ms=-1
          for nac_state_triplet1 in range(natoms*nmstates*(singlet)+natoms*(singlet+triplet+1),natoms*(singlet+n_triplets)+natoms*(singlet)*nmstates):
            for n in range(3):
              nac_step.append(read_dict['NonAdiabaticCoupling'][nac_state_triplet1+triplet*natoms*nmstates][n])
          triplet+=1
        else:
          break
      all_nac.append(nac_step)
    #transform list into matrix, for all_energy: add zero point energy 
    all_dipole_x=np.array(all_dipole_x)
    all_dipole_y=np.array(all_dipole_y)
    all_dipole_z=np.array(all_dipole_z)
    all_energy = np.array(all_energy)
    all_energy = all_energy + ezero
    all_soc_real = np.array(all_soc_real)
    all_soc_imag = np.array(all_soc_imag)
    all_grad = np.array(all_grad)
    all_nac = np.array(all_nac)
    all_geom = np.array(all_geom)
    #conversion of geometry matrix with values in a.u. to values given in angstrom
    all_geom = all_geom*0.529177211
    for t in range(0,stepsize):
      #put the real values of the SOC matrix and then the imaginary values of the soc matrix into the matrix all_soc
      socvalue=0
      soc_matrixsize_1=n_singlets*n_triplets+n_triplets*(n_triplets-1)/2
      soc_matrixsize_1=int(soc_matrixsize_1)
      for soc_r in range(soc_matrixsize_1):
        all_soc[t][socvalue]=all_soc_real[t][soc_r]
        socvalue+=1
      soc_matrixsize_2=n_singlets*n_triplets*2+n_triplets*(n_triplets-1)
      soc_matrixsize_2=int(soc_matrixsize_2)
      for soc_i in range(soc_matrixsize_2):
        all_soc[t][socvalue]=all_soc_imag[t][soc_i]
        socvalue+=1
    # for dipole matrix: x values for each step, then y values for each step, then z values for each step separated in a block (e.g. step1:xxxxyyyyyzzzzz)
      dipole_range=int(dipole_range)
      for x in range(0,dipole_range):
        all_dipole[t][x]=all_dipole_x[t][x]
   #   for y in range((dipole_range,2*dipole_range):
        all_dipole[t][x+dipole_range]=all_dipole_y[t][x]
  #    for z in range(2*(((nmstates-1)*nmstates)/2+nmstates),3*(((nmstates-1)*nmstates)/2+nmstates)-1):
        all_dipole[t][x+2*dipole_range]=all_dipole_z[t][x]
    tensor_dimension = 1+all_energy.shape[1]+all_soc.shape[1]+all_dipole.shape[1]+all_grad.shape[1]+all_nac.shape[1]
    dict_properties["AllAtomTypes"]=all_atype
    dict_properties["AllGeometries"]=all_geom
    dict_properties["NumberOfAtoms"]=natoms
    dict_properties["Step"]=all_step
    dict_properties["Stepsize"]=stepsize
    dict_properties["Tensordimension"]=tensor_dimension
    dict_properties["Energy"]=all_energy
    dict_properties["SpinOrbitCoupling"]=all_soc
    dict_properties["Dipole"]=all_dipole
    dict_properties["NonAdiabaticCoupling"]=all_nac
    dict_properties["Gradient"]=all_grad
    dict_properties["NumberOfStates"]=read_dict["NumberOfStates"]
    dict_properties["Ezero"]=ezero
    dict_properties["n_Singlets"]=n_singlets
    dict_properties['n_Triplets']=n_triplets
    return dict_properties


def readfile(sharc_dat):
  #sleep when file is not readable - file might be appended from another process with QM data
  while True:
    try:
      f=open(sharc_dat)
      fcntl.flock(f,fcntl.LOCK_SH | fcntl.LOCK_NB)
      break
    except IOError as e:
      time.sleep(0.1)
  out=f.readlines()
  fcntl.flock(f,fcntl.LOCK_UN)
  f.close()
  return out

class output_dat:
  def __init__(self,sharc_dat):
    self.data=readfile(sharc_dat)
    self.sharc_dat = sharc_dat
    # get atom types
    # get number of atoms
    #get line numbers where new timesteps start
    self.startlines=[]
    self.startline=[]
    iline=-1
    jline=-1
    while True:
      iline+=1
      jline+=1
      if iline==len(self.data):
        break
      if jline==len(self.data):
        break
      if 'Step' in self.data[iline]:
        self.startlines.append(iline)
      if '! Elements' in self.data[jline]:
        self.startline.append(jline)
    self.current=0
    for line in self.data:
      if 'natom' in line:
        a=line.split()[1]
        break
    self.natoms= int(a)
    # get number of states
    for line in self.data:
      if 'nstates_m' in line:
        s=line.split()[1:]
        break
    self.states=[ int(i) for i in s ]
    self.all_atype=[]
    for r in range(self.natoms):
      index=1+r+self.startline[self.current]
      line=self.data[index]
      t=line.split()
      t = t[0]
      self.all_atype.append(str(t))
    self.current=0
    for line in self.data:
      if 'write_grad' in line:
        gradientindex=line.split()[1]
        self.gradientindex=int(gradientindex)
      if 'write_overlap' in line:
        overlapindex=line.split()[1]
        self.overlapindex=int(overlapindex)
      if 'write_property1d' in line:
        prop1dindex=line.split()[1]
        self.prop1dindex=int(prop1dindex)
      if 'write_property2d' in line:
        prop2dindex=line.split()[1]
        self.prop2dindex=int(prop2dindex)
      if 'write_nacdr' in line:
        nacindex=line.split()[1]
        self.nacindex=int(nacindex)
      if 'ezero' in line:
        t=line.split()[1]
        self.ezero = float(t)
      if 'n_property1d' in line:
        n_1dindex=line.split()[1]
        self.n_1dindex=int(n_1dindex)
        if self.prop1dindex==0:
          self.n_1dindex=int(0)
        else:
          pass
      if 'n_property2d' in line:
        n_2dindex=line.split()[1]
        self.n_2dindex=int(n_2dindex)
        if self.prop2dindex==0:
          self.n_2dindex=int(0)
        else:
          pass
        break
      nm=0
      for i,n in enumerate(self.states):
        nm+=n*(i+1)
      self.nmstates=nm

  def __iter__(self):
    return self

  def __next__(self):
    # returns time step, U matrix and diagonal state
    self.read_dict = {  "Hamiltonian_real"  : False,
                          "Hamiltonian_imag"  : False,
                          "Dipole_x"			: False,
                          "Dipole_y"			: False,
                          "Dipole_z"			: False,
                          "Gradient"			: False,
                          "NonAdiabaticCoupling"		: False }
    read_dict=self.read_dict
    current=self.current
    self.current+=1
    if current+1>len(self.startlines):
      raise StopIteration
    # get rid of blank lines and comment
    # generate lists
    # set counter to zero
    # parse data
    # based on size initialize all relevant arrays
    all_geom = [ [ 0 for i in range(3) ] for j in range(self.natoms) ]
    for iline in range(self.natoms):
      if self.overlapindex == 0:
        index=self.startlines[current]+18+7*self.nmstates+iline
      else:
        index=self.startlines[current]+19+8*self.nmstates+iline
      line=self.data[index]
      s=line.split()
      for j in range(3):
        all_geom[iline][j]=float(s[j])
    self.read_dict["Geometries"]=all_geom
    H_real=[ [ 0 for i in range(self.nmstates) ] for j in range(self.nmstates) ]
    H_imag=[ [ 0 for i in range(self.nmstates) ] for j in range(self.nmstates) ]
    for iline in range(self.nmstates):
      index=self.startlines[current]+3+iline
      line=self.data[index]
      s=line.split()
      for j in range(self.nmstates):
        H_real[iline][j]=float(s[2*j])
        H_imag[iline][j]=float(s[2*j+1])
    self.read_dict["Hamiltonian_real"]=H_real
    self.read_dict["Hamiltonian_imag"]=H_imag
    self.read_dict["NumberOfStates"]=len(H_real)
    Dipole_x=[ [ 0 for i in range(self.nmstates) ] for j in range(self.nmstates) ]
    for iline in range(self.nmstates):
      index=self.startlines[current]+5+2*self.nmstates+iline
      line=self.data[index]
      s=line.split()
      for j in range(self.nmstates):
        Dipole_x[iline][j]=float(s[2*j])
    self.read_dict["Dipole_x"]=Dipole_x
    Dipole_y=[ [ 0 for i in range(self.nmstates) ] for j in range(self.nmstates) ]
    for iline in range(self.nmstates):
      index=self.startlines[current]+6+3*self.nmstates+iline
      line=self.data[index]
      s=line.split()
      for j in range(self.nmstates):
        Dipole_y[iline][j]=float(s[2*j])
    self.read_dict["Dipole_y"]=Dipole_y
    Dipole_z=[ [ 0 for i in range(self.nmstates) ] for j in range(self.nmstates) ]
    for iline in range(self.nmstates):
      index=self.startlines[current]+7+4*self.nmstates+iline
      line=self.data[index]
      s=line.split()
      for j in range(self.nmstates):
        Dipole_z[iline][j]=float(s[2*j])
    self.read_dict["Dipole_z"]=Dipole_z
    Gradient=[ [0 for i in range(3) ] for j in range(self.natoms*self.nmstates) ]
    if self.gradientindex==1:
      for iline in range(self.nmstates):
        if self.overlapindex==0:
          index=self.startlines[current]+19+self.n_1dindex+self.n_2dindex+2*self.natoms+(7+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        else:
          index=self.startlines[current]+20+self.n_1dindex+self.n_2dindex+2*self.natoms+(8+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        for a in range(self.natoms):
          line=self.data[index+a]
          s=line.split()
          for j in range(3):
            l=self.natoms*iline+a
            Gradient[l][j]=float(s[j])
    self.read_dict["Gradient"]=Gradient
    Nac=[ [0 for i in range(3) ] for j in range(self.natoms*self.nmstates*self.nmstates) ]
    if self.nacindex == 1:
      for iline in range(self.nmstates*self.nmstates):
        if self.overlapindex==0 and self.gradientindex==0:
          index=self.startlines[current]+19+self.n_1dindex+self.n_2dindex+2*self.natoms+(7+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        elif self.overlapindex==1 and self.gradientindex==0:
          index=self.startlines[current]+20+self.n_1dindex+self.n_2dindex+2*self.natoms+(8+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        elif self.gradientindex==1 and self.overlapindex==0:
          index =self.startlines[current]+19+self.n_1dindex+self.n_2dindex+self.natoms*self.nmstates+2*self.natoms+(8+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        elif self.gradientindex==1 and self.overlapindex==1:
          index =self.startlines[current]+20+self.n_1dindex+self.n_2dindex+self.natoms*self.nmstates+2*self.natoms+(9+self.n_1dindex+self.n_2dindex)*self.nmstates+(iline)*(self.natoms+1)+1
        for a in range(self.natoms):
          line=self.data[index+a]
          s=line.split()
          for j in range(3):
            l=self.natoms*iline+a
            Nac[l][j]=float(s[j])
    self.read_dict["NonAdiabaticCoupling"]=Nac
    return current, read_dict
if __name__ == '__main__':
    sharc_dat = "output_all.dat"
    train_frac = 0.8
    rng = np.random.RandomState( 1234 )
    descriptors, dict_properties, input_dim, train_descriptors, valid_descriptors, all_indices, valid_indices, train_indices, n_molecules, descriptors_mean, descriptors_std = get_descriptors(sharc_dat, rng, train_frac,n_molecules)
    quantities = ["energy", "soc", "dipole", "nac", "grad"]
    for quantity in quantities:
        output_dim, current_quantity_mean, current_quantity_std = get_training(quantity, dict_properties, rng, n_molecules, train_indices, valid_indices, train_frac, controlfile=None )
