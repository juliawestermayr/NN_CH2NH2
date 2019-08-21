# -*- coding: utf-8 -*-

import timeit
import theano
import theano.tensor as T
import numpy as np
import pickle
import os
import copy
from sys import argv
from process_input_output import QM_in, QM_out
from class_SHARCversion2 import get_training, get_descriptors, get_descriptors_state
from compute_derivatives import compute_dNNdxyz, get_dD_dxyz, get_hessian, get_hoppingdirection,get_dD_dxyz_coulomb,compute_dNNdxyz_coulomb
from MLFF_NN import neural_network
#trial


#================================================================
#   run NN
#================================================================
def run_all( options, weightfile, NNnumber ):
    """run NN for a list of given quantities (options["quantities"])
    1.) weights are loaded from "all_weights.pkl", then checked if weights are available for every quantity
    2.)plot_training.gp pipeline.py predict_all_qm.py prelim_mlp.py process_QM_data.py class_SHARCversion2.py empty QM_out()- data object is created to store the results of the NN run
    3.) for every quantity in options "quantity":
            - QM_in data is read (needs respective network_architecture and scaling factors which depend on the train fraction used for the respective quantity)
            - NN (with the respective weights) is run
            - NN output is stored in QM_out()-object
    4.) from QM_out()-object "QM.out" is written (QM_out_data.write_file() also checks if a transformation method from NN-output to QM.out format is available for every quantity)    
    """
    #c=os.getcwd()
    # read dictionary with all weights from 'all_weights.pkl'
    #NNoutput dictionary will be used to store predicted values of quantities from NNs
    #NNouput_new will be used to store the mean of the predicted values of quantities from all NNs
    NNoutput={}
    NNoutput_new={}
    maxsteps=float(options['tmax'])/float(options['dt'])
    maxsteps=int(maxsteps)
    ZN=False
    if "ZhuNakamura" in options:
      if options['ZhuNakamura'] == 1:
        ZN = True
      else:
        ZN = False
    for i in range(1,NNnumber+1):
      name = "NN_%i" %(i)
      all_weights = read_weights( weightfile )
      rng = np.random.RandomState(options['NN%i_seed' %(i)])
      # create input matrix
      input  = T.matrix('X')
      # create empty QM_out-data object for storing NN output
      QM_out_data = QM_out()
      #check if there are weights available for every desired quantity
      for quantity in options[ "quantity" ]:
        if quantity == "overlap":
          pass
        if quantity == "energy_grad":
          try:
            all_weights[name]['energy']
          except KeyError:
            print( "KeyError - No weights available for Energy (Energy_grad)." )
            exit()
        if quantity == "nac_grad":
          try:
            all_weights[name]['nac']
          except KeyError:
            print( "KeyError - No weights available for Energy (Energy_grad)." )
            exit()
        if quantity == "dipole":
          try:
            all_weights[name][quantity]
          except KeyError:
            print( "KeyError - No weights available for %s." % quantity )    
            exit()   
        if quantity == "soc":
          try:
            all_weights[name][quantity]
          except KeyError:
            print( "KeyError - No weights available for %s." % quantity )    
            exit()   
        if quantity == "nac":
          try:
            all_weights[name][quantity]
          except KeyError:
            print( "KeyError - No weights available for %s." % quantity )    
            exit()
      NNoutput[name]={}
      # run NN for every quantity 
      for quantity in options[ "quantity" ]:
        if quantity == "overlap":
          data, xyz_coords, states, n_atoms, nstates, nmstates, istates, spin, n_singlets, n_triplets = read_QM_in( network_architecture, scaling_factors, options[ "QMin" ], maxsteps, options )
          QM_out_data.add_overlap( nmstates )
          pass
        if quantity == "energy_grad":
          NNoutput[name][quantity]={}
          # read weights
          scaling_factors, saved_params, network_architecture, zeroindex = all_weights[name]['energy']
          # read data from QM.in
          data, xyz_coords, states, n_atoms, nstates, nmstates, istates, spin, n_singlets, n_triplets = read_QM_in( network_architecture, scaling_factors, options[ "QMin" ], maxsteps, options )
          # Initialize neural network
          NN = neural_network( network_architecture, input, rng, options["NN%i_seed" %(i)])
          NN.load_weights( saved_params )
          prediction= theano.function( inputs=[input], outputs=NN.prediction )
          # Actual output of neural network for input data
          #data = descriptors, should be the same for every quantity
          nn_output = (prediction( data.get_value().reshape((1,-1)))*scaling_factors[3] + scaling_factors[2])[0]
          # add nn_output for current quantity to QM_out_data
          #nn_output = array of numbers, as many numbers as states  are given
          x = len(zeroindex)
          zeroindex=np.array(zeroindex)
          #prepare zeroindices matrix
          if len(zeroindex) > 0:
            for j in range(len(zeroindex)):
              nn_output[zeroindex[j]] = 0.0
          NNoutput[name][quantity]["energy"]=nn_output
          # Gradients
          NN_grad_inputs=T.jacobian( NN.prediction[0,:], input )
          get_NN_grad = theano.function( inputs=[input], outputs=NN_grad_inputs )
          # Descriptor derived wrt xyz; only one descriptor
          dD_dxyz = get_dD_dxyz( xyz_coords[0], data.get_value()[0:], n_atoms )
          # compute derivative of NN output with respect to descriptors
          dNNdD = get_NN_grad(data.get_value()[0:].reshape((1,-1)) ) #function by NN ?
          #get first derivative
          # compute derivative of NN output with respect to xyz coordinates
          #derivative of NN w.r.t. xyz; array indices: [number of states] [number of atoms] [xyz]
          all_dNNdxyz = compute_dNNdxyz( dNNdD, dD_dxyz, scaling_factors, n_atoms )
          #multiply all_dNNdxyz with factor 0.529177211 (right now H/a.u. - to get from a.u. to Bohr divide by 0.529177211, since it is in the enumerator, multiple with this factor)
          all_dNNdxyz = np.array(all_dNNdxyz)#*0.529177211
          grad_derived = np.zeros((n_singlets+n_triplets)*n_atoms*3)
          m=0
          n=0
          l=0
          for h in range(((n_singlets+n_triplets)*n_atoms*3)):
            grad_derived[h]=all_dNNdxyz[m][n][l]
            l+=1
            if l>=3:
              l=0
              n+=1
              if n >= n_atoms:
                n=0
                m+=1
          #QM_out_data.add( quantity , nnoutput_new, nmstates, n_atoms, istates, nstates, spin )
          NNoutput[name][quantity]["gradient"]=grad_derived

          #compute Hessian and get hopping direction for Zhu-Nakamura formulas
          if ZN==True:
            #calculate hessian for hopping direction with Zhu-Nakamura
            Hessian_allstates=[]
            Hessian_singlet=[]
            Hessian_triplet=[]
            for state in range(n_singlets):
              dNNdD_state = dNNdD[state]
              Hessian = T.jacobian(expression=(NN_grad_inputs[state].flatten()),wrt=input)
              get_NN_hess = theano.function(inputs=[input], outputs=Hessian)
              dim=int(n_atoms*(n_atoms-1)/2)
              dNN2dD2=get_NN_hess(data.get_value().reshape((1,-1)))
              dNN2dD2=dNN2dD2.reshape((dim,dim))
              hessian=get_hessian(nmstates,n_atoms,state,dNNdD_state,dNN2dD2,scaling_factors,xyz_coords,dim)
              Hessian_singlet.append(hessian)
            Hessian_singlet=np.array(Hessian_singlet)
            for state in range(n_singlets,n_singlets+n_triplets):
              dNNdD_state = dNNdD[state]
              Hessian = T.jacobian(expression=(NN_grad_inputs[state].flatten()),wrt=input)
              get_NN_hess = theano.function(inputs=[input], outputs=Hessian)
              dim=int(n_atoms*(n_atoms-1)/2)
              dNN2dD2=get_NN_hess(data.get_value().reshape((1,-1)))
              dNN2dD2=dNN2dD2.reshape((dim,dim))
              hessian=get_hessian(nmstates,n_atoms,state,dNNdD_state,dNN2dD2,scaling_factors,xyz_coords,dim)
              Hessian_triplet.append(hessian)
            if n_triplets==int(0):
              Hessian_allstates=np.array(Hessian_singlet)*(0.529177249**2)
            else:
              Hessian_triplet=np.array(Hessian_triplet)
              Hessian_allstates=np.concatenate((Hessian_singlet,Hessian_triplet,Hessian_triplet,Hessian_triplet))
              #convert from 1/A² to 1/Bohr²
              Hessian_allstates=np.array(Hessian_allstates)*(0.529177249**2)
            hopping_direction = get_hoppingdirection(Hessian_allstates,nmstates,n_atoms)
            NNoutput[name]['ZN']={}
            NNoutput[name]['ZN']['Hessian'] = Hessian_allstates
            NNoutput[name]['ZN']['hopping_direction'] = hopping_direction

        if quantity == "energy":
          NNoutput[name][quantity]={}
          #  read weights
          scaling_factors, saved_params, network_architecture, zeroindex = all_weights[name][quantity]
          # read data from QM.in
          data, xyz_coords, states, n_atoms, nstates, nmstates, istates, spin, n_singlets, n_triplets = read_QM_in( network_architecture, scaling_factors, options[ "QMin" ], maxsteps, options )
          # Initialize neural network
          NN = neural_network( network_architecture, input, rng, options["NN%i_seed" %(i)] )
          NN.load_weights( saved_params )
          prediction = theano.function( inputs=[input], outputs=NN.prediction )
          # Actual output of neural network for input data
          nn_output = (prediction( data.get_value().reshape((1,-1)))*scaling_factors[3] + scaling_factors[2])[0]
          NNoutput[name][quantity]=nn_output

        if quantity == "soc":
          NNoutput[name][quantity]={}
          #  read weights
          scaling_factors, saved_params, network_architecture, zeroindex = all_weights[name][quantity]
          # read data from QM.in
          data, xyz_coords, states, n_atoms, nstates, nmstates, istates, spin, n_singlets, n_triplets = read_QM_in( network_architecture, scaling_factors, options[ "QMin" ], maxsteps, options )
          # Initialize neural network
          NN = neural_network( network_architecture, input, rng, options["NN%i_seed" %(i)] )
          NN.load_weights( saved_params )
          prediction = theano.function( inputs=[input], outputs=NN.prediction )
          # Actual output of neural network for input data
          nn_output = (prediction( data.get_value().reshape((1,-1)))*scaling_factors[3] + scaling_factors[2])[0]
          #prepare zeroindices matrix
          x = len(zeroindex)
          zeroindex=np.array(zeroindex)
          if len(zeroindex) > 0:
            for j in range(len(zeroindex)):
              nn_output[zeroindex[j]] = 0.0
          NNoutput[name][quantity]=nn_output

        if quantity == "dipole":
          NNoutput[name][quantity]={}
          #  read weights
          scaling_factors, saved_params, network_architecture, zeroindex = all_weights[name][quantity]
          # read data from QM.in
          data, xyz_coords, states, n_atoms, nstates, nmstates, istates, spin, n_singlets, n_triplets = read_QM_in( network_architecture, scaling_factors, options[ "QMin" ], maxsteps, options )
          # Initialize neural network
          NN = neural_network( network_architecture, input, rng, options["NN%i_seed" %(i)] )
          NN.load_weights( saved_params )    
          prediction = theano.function( inputs=[input], outputs=NN.prediction )
          # Actual output of neural network for input data
          nn_output = (prediction( data.get_value().reshape((1,-1)))*scaling_factors[3] + scaling_factors[2])[0]
          #prepare zeroindices matrix
          x = len(zeroindex)
          zeroindex=np.array(zeroindex)
          if len(zeroindex) > 0:
            for j in range(len(zeroindex)):
              nn_output[zeroindex[j]] = 0.0
          NNoutput[name][quantity]=nn_output

        if quantity == "nac":
            NNoutput[name][quantity]={}
            #  read weights
            scaling_factors, saved_params, network_architecture, zeroindex = all_weights[name][quantity]
            #print zeroindex
            #print "Scaling factors", scaling_factors, "saved_params", saved_params, "network_architecture", network_architecture
            # read data from QM.in
            data, xyz_coords, states, n_atoms, nstates, nmstates, istates, spin, n_singlets, n_triplets = read_QM_in( network_architecture, scaling_factors, options[ "QMin" ], maxsteps, options )
            # Initialize neural network
            NN = neural_network( network_architecture, input, rng, options["NN%i_seed" %(i)] )
            NN.load_weights( saved_params )    
            prediction = theano.function( inputs=[input], outputs=NN.prediction )
            # Actual output of neural network for input data
            nn_output = (prediction( data.get_value().reshape((1,-1)))*scaling_factors[3] + scaling_factors[2])[0]
            #prepare zeroindices matrix
            x = len(zeroindex)
            zeroindex=np.array(zeroindex)
            if len(zeroindex) > 0:
              for j in range(len(zeroindex)):
                nn_output[zeroindex[j]] = 0.0
            NNoutput[name][quantity]=nn_output

    #there is a threshold given for each quantity: if this threshold is exceeded, the prediction will be stopped and SHARC has to generate new training data
    if NNnumber>1:
      stop,QMout=QM_out_data.add2( options, NNoutput, nmstates, n_atoms, istates, nstates, spin, n_singlets, n_triplets,ZN )
      QM_out_data._write_file("QM.out")
    else:
      stop,QMout=QM_out_data.add( options['quantity'], name, NNoutput, nmstates, n_atoms,  istates, nstates, spin, n_singlets, n_triplets,ZN )
      QM_out_data._write_file("QM.out")
      #print(QMout)
    if stop=='True':
      currentpath = os.getcwd()
      path = options['stoppath']
      os.chdir(path)
      #writes a script that will be called when the error of the NNs among each other is too big
      os.system('touch STOP')
      os.chdir(currentpath)
      return QMout

def read_zeroindices( quantity ):
    zero = open( "zeroindices_%s" %( quantity ), "r" )
    indices=zero.read().split()
    x = len(indices)
    zeros=[]
    j=0
    if x > 1:
      if x>9:
        x=x-1
        j=1
        zeros=np.zeros(x)
        zeros[0]=indices[j][:]
      else:
        zeros=np.zeros(x)
        zeros[0]=indices[0][1:]
      for i in range(1,x-1):
        zeros[i]=indices[j+i]
      zeros[x-1]=indices[x-1+j][:-1]
    return zeros

#================================
# Write weights and scaling factors for later use
#================================

def write_weights( all_weights, weightfile ):
    # write dictionary all_weights to 'all_weights.pkl' 
    # all_weights[quantity] = [scaling_factors, best_weights, network_architecture]
    try:
        pf=open(weightfile,'wb')
    except IOError:
        print("Could not write to 'all_weights.pkl'.")
        exit()
    pickle.dump( all_weights, pf )
    pf.close()
    print( "Stored best weights in %s" %( weightfile ) )

#================================
# Read weights and scaling factors
#================================

def read_weights( weightfile ):
    # read dictionary all_weights from 'all_weights.pkl' 
    # all_weights[quantity] = [scaling_factors, best_weights, network_architecture]
    try:
        pf=open(weightfile,'rb')
    except IOError:
        print( "Could not open %s" %( "all_weights.pkl" ) )
        exit()
    all_weights = pickle.load( pf )
    pf.close()
    return all_weights

#================================
# Read input for "run" from file
#================================

def read_inp_data( inp_data, network_architecture, scaling_factors, xyz_file ):
    # 1) Read in data
    try:
        raw_inp = open( inp_data ).readlines()
    except IOError:
        print( "Could not open %s" % inp_data )
        exit()

    #for options in parse_input:
    descriptors = []
    for line in raw_inp:
        data = line.split( "quantity" )[1:].split( "descr:" )
        desc = [ np.float( i ) for i in data[1].split() ]
        descriptors.append( desc )
        descriptors = np.array( descriptors )
    input_dim = descriptors.shape[1]
    descriptors = ( descriptors - scaling_factors[0] ) / scaling_factors[1]
    descriptors = theano.shared( np.asarray(descriptors,dtype=theano.config.floatX), borrow=True )

    for quantity in options[ "quantities" ]:
        quantity = []
    # Read descriptor from each line; energies only for testing purposes
        for line in raw_inp:
            data = line.split( "quantity" )[1:].split( "descr:" )
        quantity = [ np.float( i ) for i in data[0].split() ]
        quantity.append( quantity )
        quantity    = np.array( quantity )
        output_dim = quantity.shape[1]
        #print output_dim, "outputdim", input_dim, "input_dim"
        if input_dim != network_architecture[0][0]:
            print( 'Descriptor dimension of input and training data not matching.' )
            exit()
        if output_dim != network_architecture[0][-1]:
            print( 'Output size and training data not matching.' )
            exit()
        quantity = ( quantity - scaling_factors[2] ) / scaling_factors[3]
        quantity = theano.shared( np.asarray(quantity,dtype=theano.config.floatX), borrow=True )    
    # Applying scaling factors from training
        data = [ descriptors, quantity ]
        data = [ descriptors, quantity ]
        dummy, dummy, xyz_coords = read_xyz_file( xyz_file ) 

        return data, xyz_coords

#=================================================
# Read input for "run" directly from "QM.in"
#=================================================

def read_QM_in( network_architecture, scaling_factors, filename, stepnumber, options ):
    """reads data from <filename> ( "QM.in" by default )
    """    
    #get path of QM.in
    pathQMNN=options['QMoutpath']
    #create QM_in object (imported from QM_transformer) that holds all information from the QM.in-file   
    raw_data = QM_in( filename, stepnumber, pathQMNN )
    #get descriptors
    descriptors =  raw_data.descriptors#_coulomb
    descriptors = np.array(descriptors)
    states = raw_data.states
    nmstates = raw_data.nmstates
    n_atoms = raw_data.n_atoms
    nstates = raw_data.nstates
    spin = raw_data.spin
    #charges = raw_data.charges
    istates = raw_data.istates
    n_singlets = raw_data.n_singlets
    n_triplets = raw_data.n_triplets
    input_dim = len( descriptors )

    if input_dim != network_architecture[0][0]:
        print( 'Descriptor dimension of input and training data not matching.' )
        exit()

    #scale descriptors 
    descriptors = ( descriptors - scaling_factors[0] ) / scaling_factors[1]

    #load descriptors in theano shared variables
    descriptors = theano.shared( np.asarray(descriptors,dtype=theano.config.floatX), borrow=True ) 
    xyz_coords = []
    xyz_coords.append(raw_data.geoms)

    return descriptors, xyz_coords, states, n_atoms, nstates, nmstates, istates, spin, n_singlets, n_triplets



if __name__ == "__main__":
    try:
        name, input_file, mode = argv
    except ValueError:
        print( "Usage: script <input_file>" )
        exit()
    inputfile=argv[1]
    options=parse_input(inputfile, 'SH2NN.inp')
    NNnumber = options['NNnumber']
    weightfile=options['weightpath']
    mode=argv[2]
    options['mode']=mode
    run_all( options, weightfile, NNnumber )
