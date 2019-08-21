# -*- coding: utf-8 -*-

import timeit
import theano
import theano.tensor as T
import numpy as np
import pickle
import os 
import copy
from sys import argv
from class_SHARCversion2 import get_training, get_descriptors,get_descriptors_state
from compute_derivatives import compute_dNNdxyz, get_dD_dxyz, get_hessian, get_hoppingdirection,get_dD_dxyz_coulomb,compute_dNNdxyz_coulomb
from MLFF_NN import neural_network
from get_coulomb import get_descriptors_coulomb 
#trial


#================================
# SGD training          
#================================

def train_nn(name, mode, data, descriptors, network_architecture, rng, scaling_factors, quantity, dict_properties, valid_indices, train_indices, seed, learning_rate, hypergradient_learning_rate, decay_steps, decay_factor, L2_reg, n_epochs, batch_size, improvement_threshold, force_influence):
    # data from previous: data = [ ( train_descriptors, train_current_quantity ), ( valid_descriptors, valid_current_quantity ) ]
    train_set_x, train_set_y = data[0]
    valid_set_x, valid_set_y = data[1]
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size +1 
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size +1
    # build NN
    # allocate symbolic variables for the data
    #curr_learning_rate = learning_rate
    index = T.lscalar()  # index to a [mini]batch
    input  = T.matrix('X')    # the data is presented as rasterized images
    target = T.matrix('Y')    # the labels are presented as 1D vector of
    LR_var = theano.shared( name="l", value=learning_rate )
    # construct neural network
    print( "Building %s..." %(name))
    # seed RNG
    # CAUTION/PROBLEM: rng is seeded for every training quantity, but could be seeded for every layer of every quantity - What is suitable?
    print( "Seed RNG: ", seed )
    rng = np.random.RandomState( seed )
    NN = neural_network( network_architecture, input, rng, seed )
    # for quantity energy_grad: calculate gradients from NN-energies
    number_of_geoms = len(train_indices) + len(valid_indices)
    indices = np.concatenate((train_indices,valid_indices))
    natoms = dict_properties['NumberOfAtoms']
    nmstates = dict_properties['NumberOfStates']
    n_singlets = dict_properties['n_Singlets']
    n_triplets = dict_properties['n_Triplets']
    if quantity == "energy_grad":
      #steps = number of geometries in output_all.dat used for training
      for geom in range(number_of_geoms):
        dict_properties.update({ 'grad_%i' %(indices[geom]) : False})
        NN_grad_inputs = T.jacobian( NN.prediction[0,:], input )
        get_NN_grad = theano.function( inputs=[input], outputs=NN_grad_inputs )
        # Descriptor derived wrt xyz0
        #build array with values of the descriptor for the first geometry (=Step)
        #the size (natoms*natoms-natoms)/2 comes from the fact that the descriptors are the inverse distance matrix(natoms*natoms), with diagonal elements=0
        #geometries are given in Angström, same accounts for geoms taken for coulomb matrix
        if dict_properties['Coulomb']==False:
          geom_matrixsize=int(natoms*(natoms-1)/2)
          data = np.zeros((geom_matrixsize))
          for descriptor_value in range(geom_matrixsize):
            data[descriptor_value]=descriptors[geom][descriptor_value]
          dD_dxyz = get_dD_dxyz( dict_properties['AllGeometries'][geom], data, natoms ) 
          # compute derivative of NN output with respect to descriptors
          dNNdD = get_NN_grad(data[0:].reshape((1,-1)) ) #function by NN ?
          # compute derivative of NN output with respect to xyz coordinates
          #for state in range( 6 ):
          #derivative of NN w.r.t. xyz; array indices: [number of states] [number of atoms] [xyz]
          #scaling necessary, because it is easier for the NN to predict for example energies around 0 than energies around 2000
          all_dNNdxyz = compute_dNNdxyz( dNNdD, dD_dxyz, scaling_factors, natoms )
          #multiply all_dNNdxyz with factor 0.529177211 (right now H/a.u. - to get from a.u. to Bohr divide by 0.529177211, since it is in the enumerator, multiple with this factor)
          #values are given in H/Bohr in output.dat - needs to be the same unit for comparison
          all_dNNdxyz = np.array(all_dNNdxyz)#*0.529177211
          #write the derived gradients into an array of the size: number of states (nmstates) * number of atoms (natoms) * 3 (xyz)
          grad_derived = np.zeros((n_singlets+n_triplets)*natoms*3)
          statenumber=0
          atom=0
          xyz=0
          #we have a gradient matrix for each singlet and triplet once
          for ind in range((n_singlets+n_triplets)*natoms*3):
            grad_derived[ind]=all_dNNdxyz[statenumber][atom][xyz]
            xyz+=1
            if xyz>=3:
              xyz=0
              atom+=1
              if atom >= natoms:
                atom=0
                statenumber+=1
          dict_properties.update({'grad_%i' %(indices[geom]):grad_derived})
        else:
          geom_matrixsize=int(natoms*(natoms-1)/2+natoms)
          data = np.zeros((geom_matrixsize))
          for descriptor_value in range(geom_matrixsize):
            data[descriptor_value]=descriptors[geom][descriptor_value]
          dD_dxyz = get_dD_dxyz_coulomb( dict_properties['AllGeometries'][geom], data, natoms,dict_properties['Charges'] ) 
          # compute derivative of NN output with respect to descriptors
          dNNdD = get_NN_grad(data[0:].reshape((1,-1)) ) #function by NN ?
          # compute derivative of NN output with respect to xyz coordinates
          #for state in range( 6 ):
          #derivative of NN w.r.t. xyz; array indices: [number of states] [number of atoms] [xyz]
          #scaling necessary, because it is easier for the NN to predict for example energies around 0 than energies around 2000
          all_dNNdxyz = compute_dNNdxyz_coulomb( dNNdD, dD_dxyz, scaling_factors, n_atoms )
          #multiply all_dNNdxyz with factor 0.529177211 (right now H/a.u. - to get from a.u. to Bohr divide by 0.529177211, since it is in the enumerator, multiple with this factor)
          #values are given in H/Bohr in output.dat - needs to be the same unit for comparison
          all_dNNdxyz = np.array(all_dNNdxyz)#*0.529177211
          #write the derived gradients into an array of the size: number of states (nmstates) * number of atoms (natoms) * 3 (xyz)
          grad_derived = np.zeros((n_singlets+n_triplets)*natoms*3)
          statenumber=0
          atom=0
          xyz=0
          #we have a gradient matrix for each singlet and triplet once
          for ind in range((n_singlets+n_triplets)*natoms*3):
            grad_derived[ind]=all_dNNdxyz[statenumber][atom][xyz]
            xyz+=1
            if xyz>=3:
              xyz=0
              atom+=1
              if atom >= natoms:
                atom=0
                statenumber+=1
          dict_properties.update({'grad_%i' %(indices[geom]):grad_derived})
    else:
      pass
    #get cost function
    cost = ( NN.mse_error( target, quantity, dict_properties, indices, force_influence) + L2_reg * NN.L2_sqr )
    #Theano function: compiling graphs into callable objects
    #has inputs, outputs, givens (2 pairs with the same type, specific substitutions to make in the computation graph)
    #since givens is iterable: the function iterates with index over minibatches
    # compiling a Theano function that computes the mistakes that are made by the model on a minibatch
    dict_properties.update({"batchsize" : batch_size})
    #for energy_grad the error of the forces and energies have to be separated
    if quantity == "energy_grad":
      validate_model = theano.function(
          inputs=[index],
          outputs=T.stack(NN.mse_error_valid( target, quantity, dict_properties, valid_indices) ),
          givens={
              input: valid_set_x[index * batch_size:(index + 1) * batch_size],
              target: valid_set_y[index * batch_size:(index + 1) * batch_size]
          }
      )
    else:
      validate_model = theano.function(
          inputs=[index],
          outputs=NN.mse_error( target, quantity, dict_properties, valid_indices, force_influence),
          givens={
              input: valid_set_x[index * batch_size:(index + 1) * batch_size],
              target: valid_set_y[index * batch_size:(index + 1) * batch_size]
          }
      )

    # compute the gradient of cost with respect to params
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in NN.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    #updates = [ (param, param - learning_rate * gparam) for param, gparam in zip( NN.params, gparams ) ]
 #>>   updates = Adam( cost, NN.params, hypergradient_learning_rate, learning_rate )
    updates = Adam( cost, NN.params, hypergradient_learning_rate, LR_var )
    #print LR_var.get_value()

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            input: train_set_x[index * batch_size: (index + 1) * batch_size],
            target: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    print('... training')

    # early-stopping parameters
    patience = 10000.0  # look as this many examples regardless
    patience_increase = 2.0  # wait this much longer when a new best is
                           # found
    #the improvement threshold is given in the options_run.input file
    patience_train = 10000.0
    patience_increase_train = 2.0
    validation_frequency = min(n_train_batches, patience // 2)
    training_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = np.inf
    best_training_loss = np.inf
    best_iter = 0.0
    best_iter_train = 0.0

    epoch = 0.0
    done_looping = False

    start_time = timeit.default_timer()
    best_weights = NN.params
    trainfile = "trainfile_%s_training" %mode
    trainfile_validation = "trainfile_%s_validation" %mode
    file = open( trainfile, "w" )
    file2= open( trainfile_validation, "w")
    curr_learning_rate = learning_rate
    while (epoch < n_epochs) and (not done_looping):
      epoch = epoch + 1
      if epoch % decay_steps == 0:
        curr_learning_rate = curr_learning_rate * decay_factor
        LR_var.set_value( curr_learning_rate )
      for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        # iteration number
        iter = (epoch - 1) * n_train_batches + minibatch_index
        #the next step is one epoch and this is done every time divided by 0
        if (iter + 1) % (validation_frequency) == 0:
          # compute zero-one loss on validation set
          validation_losses = [validate_model(i) for i
                            in range(n_valid_batches)]
          this_validation_loss = np.mean(validation_losses)
          #if quantity == "energy_grad":
          file2.write( "%20.12f %20.12f \n" %( epoch, np.sqrt( this_validation_loss )*scaling_factors[3][0]) )
            # if we got the best validation score until now
          if this_validation_loss < best_validation_loss:
            #improve patience if loss improvement is good enough
            if ( this_validation_loss < best_validation_loss * improvement_threshold ):
                patience = max(patience, iter * patience_increase)
                #improvement_threshold=improvement_threshold
                #print(this_validation_loss,best_validation_loss, curr_learning_rate)
            best_validation_loss = this_validation_loss
            best_iter = iter
            # Save best weights
            best_weights = copy.deepcopy( NN.params ) 
        if patience <= iter:
          done_looping = True
          break

        if (iter + 1) % training_frequency == 0:
          training_losses = [train_model(i) for i 
                            in range(n_train_batches)]
          this_training_loss = np.mean(training_losses)
          file.write("%20.12f %20.12f\n " %(epoch,np.sqrt(this_training_loss)*scaling_factors[3][0])) 
          if this_training_loss < best_training_loss:
            if ( this_training_loss < best_training_loss * improvement_threshold ):
              patience_train = max(patience_train, iter * patience_increase_train)

            best_training_loss = this_training_loss
            best_iter_train = iter

        if patience_train <= iter:
          done_looping = True
          break

    file.write( "Validation score scaled: %20.12f \n" %(np.sqrt( best_validation_loss )*scaling_factors[3][0] ) )
    file.write( "Validation score unscaled: %20.12f \n" %(np.sqrt( best_validation_loss ) ) )
    file.write( "Training score scaled: %20.12f \n" %(np.sqrt( best_training_loss )*scaling_factors[3][0] ) )
    file.write( "Training score unscaled: %20.12f \n" %(np.sqrt( best_training_loss ) ) )
    file.close()
    file2.close()
    end_time = timeit.default_timer()
    # Rescaling validation score with energy standard deviation
    print( 'Validation score of %s: %12.9f a.u.' % (mode, np.sqrt( best_validation_loss )*scaling_factors[3][0] ) )
    print( 'Training score of %s: %12.9f a.u.' % (mode, np.sqrt( best_training_loss )*scaling_factors[3][0] ) )
    print( 'Runtime: %.1fs' % ( end_time - start_time ) )

    return best_weights


#================================
# Alternative ADAM update
#================================

# https://arxiv.org/abs/1412.6980 (ADAM paper)
# http://sebastianruder.com/optimizing-gradient-descent/ (gradient descent overview)
# https://arxiv.org/abs/1703.04782 (Hypergradient Descent: Learning rate adaption) 

#1-b1 corresponds to beta1, 1-b2 corresponds to beta2
#p corresponds to the parameters to be updated
#lr_t corresponds to the updated learning rate
def Adam(cost, params, h_lr, lr, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    n = theano.shared( np.cast[theano.config.floatX](0.0) )
    n_t = n + 1.
    fix1 = 1. - (1. - b1)**n_t
    fix2 = 1. - (1. - b2)**n_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - ((lr_t) * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((n, n_t))
    return updates


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
#   train NN
#=================================================

def train_all( options, weightfile, NNnumber ):
    """Train NN for a list of given quantities (options["quantities"]).
    1.) descriptors are obtained from "output.xyz"
    2.) for every quantity in options["quantity"]:
            - training data is obtained from "output.dat", if no method for obtaining the desired quantity is available, exception is thrown
            - NN is trained
            - best_weights, scaling_factors and network_architecture are appended to dictionary all_weights
    3.) dictionary with weights for all_quantities is written to "all_weights.pkl"
    """

    # create empty dictionary for storing weights for all quantities, 
    #   e.g. all_weights["energy"] = scaling_factors, best_weights, network_architecture for energy
    #Train all Neural Networks separately and safe it in all_weights.pkl
    all_weights = {}
    for i in range(1,NNnumber+1):
      name = "NN_%i" %(i)
      all_weights[name]={}
      n_molecules = options['nmolec']
      rng = np.random.RandomState(options['NN%i_seed' %(i)])
      for quantity in options["quantity"]:
        if quantity == "overlap":
          pass
        elif quantity == "energy_grad":
          q='energy'
          sharc_dat = options["datafile"]
          # get descriptors from 'output.dat'
          #descriptors, dict_indices, dict_properties, input_dim, train_descriptors, valid_descriptors, all_indices, valid_indices, train_indices, n_molecules, descriptors_mean, descriptors_std, zeroindex = get_descriptors_coulomb( sharc_dat, rng, options["NN%i_split" %(i)],n_molecules ) 
          descriptors, dict_indices, dict_properties, input_dim, train_descriptors, valid_descriptors, all_indices, valid_indices, train_indices, n_molecules, descriptors_mean, descriptors_std, zeroindex = get_descriptors( sharc_dat, rng, options["NN%i_split" %(i)],n_molecules ) 
          mode = quantity
          output_dim, current_quantity_mean, current_quantity_std, train_current_quantity, valid_current_quantity, zeroindex = get_training(quantity, dict_properties, rng, n_molecules, train_indices, valid_indices,
                                                                                                                  options["NN%i_split" %(i)], 
                                                                                                                  options["controlfile"] )
          # NN_size_new: 1 20 5 for example for H-H: one input dimension, 20 NN HL, 5 output_dim for 5 states      
          options["NN%i_size_new" %(i)] = [input_dim] + options["NN%i_size_%s" %(i,q)] + [output_dim]
          network_architecture = ( options["NN%i_size_new" %(i)], options["NN%i_act_%s" %(i,q)] )
          scaling_factors = (descriptors_mean, descriptors_std, current_quantity_mean, current_quantity_std)
          data = [ ( train_descriptors, train_current_quantity ), ( valid_descriptors, valid_current_quantity ) ]
          # train NN
          # changed seed to fixed value that only seed changes indices trained of training set
          #if inverse distance - this is only for derivation of NN(Energy):
          best_weights = train_nn(name, mode, data, descriptors, network_architecture, rng, scaling_factors, "energy_grad", dict_properties, valid_indices, train_indices,
                           seed          		= options['NN%i_seed'%i],
                           learning_rate 		= options["NN%i_learn_%s" %(i,q)],
                           hypergradient_learning_rate  = options["NN%i_learn2_%s" %(i,q)],
                           decay_steps 			= options['NN%i_decay_steps_%s' %(i,q)],
                           decay_factor 		= options['NN%i_decay_factor_%s' %(i,q)],
                           L2_reg       		= options["NN%i_L2_reg_%s" %(i,q)],
                           n_epochs     		= options["NN%i_n_epochs" %(i)],
                           batch_size   		= options["NN%i_batch" %(i)],
                           improvement_threshold	= options["improvement_threshold_%s" %q],
                           force_influence 		= options['force_influence'])
          # add weights for trained quantity to dictionary 'all_weights'
          all_weights[ name ][ q ] = [ scaling_factors, best_weights, network_architecture, zeroindex ]
          #copy trainingfiles in new folder
          #if you want trainingdata put the '#' of the next four lines away
          write_weights( all_weights, weightfile )
        else:
          sharc_dat = options["datafile"]
          # get descriptors from 'output.dat'
          descriptors, dict_indices, dict_properties, input_dim, train_descriptors, valid_descriptors, all_indices, valid_indices, train_indices, n_molecules, descriptors_mean, descriptors_std, zeroindex = get_descriptors( sharc_dat, rng, options["NN%i_split" %(i)], n_molecules ) 
          #train NN for every quantity in options[ "quantities" ]
          mode = quantity
          output_dim, current_quantity_mean, current_quantity_std, train_current_quantity, valid_current_quantity, zeroindex = get_training(quantity, dict_properties, rng, n_molecules, train_indices, valid_indices,
                                                                                                                  options["NN%i_split" %(i)], 
                                                                                                                  options["controlfile"] )
          # NN_size_new: 1 20 5 for example for H-H: one input dimension, 20 NN HL, 5 output_dim for 5 states      
          options["NN%i_size_new" %(i)] = [input_dim] + options["NN%i_size_%s" %(i,quantity)] + [output_dim]
          network_architecture = ( options["NN%i_size_new" %(i)], options["NN%i_act_%s" %(i,quantity)] )
          scaling_factors = (descriptors_mean, descriptors_std, current_quantity_mean, current_quantity_std)
          data = [ ( train_descriptors, train_current_quantity ), ( valid_descriptors, valid_current_quantity ) ]
          # train NN
          best_weights = train_nn(name, mode, data, descriptors, network_architecture, rng, scaling_factors, quantity, dict_properties, valid_indices, train_indices,
                           seed                        = options["NN%i_seed" %(i)],
                           learning_rate               = options["NN%i_learn_%s" %(i,quantity)],
                           hypergradient_learning_rate = options["NN%i_learn2_%s" %(i,quantity)],
                           decay_steps		       = options['NN%i_decay_steps_%s' %(i,quantity)],
                           decay_factor		       = options['NN%i_decay_factor_%s' %(i,quantity)],
                           L2_reg                      = options["NN%i_L2_reg_%s" %(i,quantity)],
                           n_epochs                    = options["NN%i_n_epochs" %(i)],
                           batch_size                  = options["NN%i_batch" %(i)],
                           improvement_threshold	= options["improvement_threshold_%s" %quantity],
                           force_influence 		= options['force_influence'] )
          # add weights for trained quantity to dictionary 'all_weights'
          all_weights[ name ][ quantity ] = [ scaling_factors, best_weights, network_architecture, zeroindex ]
   
          write_weights( all_weights, weightfile )

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
    train_all( options, weightfile, NNnumber )
