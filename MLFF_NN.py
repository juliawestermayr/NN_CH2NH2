# -*- coding: utf-8 -*-

import timeit
import theano
import theano.tensor as T
import numpy as np
import pickle
import os
import copy
from sys import argv
#trial


#================================
# Multi-Layer Perceptron
#================================

class neural_network(object):

    def __init__( self, network_architecture, input, rng, seed ):
        # Get architecture
        #self.layerdim = NN size new - specific for every quantity, self.activations = NN size - given size at the beginning
        self.layerdim, self.activations = network_architecture
        #for 1 HL, self.nlayers is 2
        self.nlayers = len( self.layerdim ) - 1
        # Setup containers for network parameters
        self.W = []
        self.b = []
        self.params = []
        linput = input
        # build neural network function
        for l in range( self.nlayers ):
            linput = self.layer( linput, self.layerdim[l], self.layerdim[l+1], self.activations[l], rng, seed, l)
            #n_in always the first layer, n_out the next layer, repeated until l=self.nlayers
        self.output = linput
        self.prediction = self.output
        # get L2 norm
        self.L2_sqr = ( self.W[0]**2 ).sum()
        for w in range( 1,self.nlayers ):
          self.L2_sqr = self.L2_sqr + ( self.W[w]**2 ).sum()

    def layer( self, input, n_in, n_out, activation, rng, seed, l_index):
        # Get theano variable for input
        self.input = input
        # Init weights of layer
        W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
        b_values = np.zeros((n_out,), dtype=theano.config.floatX)
        W = theano.shared(value=W_values, name='W%d' %l_index, borrow=True)
        b = theano.shared(value=b_values, name='b%d' %l_index, borrow=True)

        self.W += [ W ]
        self.b += [ b ]
        self.params += [ W, b ]

        if activation == "t":
            output = T.tanh( T.dot( input, W ) + b )
        elif activation == "n":
            output = T.log( T.dot(0.5, T.exp( T.dot( input, W ) + b)) + 0.5)
        else:
            output = T.dot( input, W ) + b


        return output

    def mse_error( self, target, quantity, dict_properties, index, force_influence ):
      # for the quantity "energy_grad", the error includes the error of the derived gradient from NN-energies with respect to the QM-gradients
      if quantity == "energy_grad":
	#print "hello", force_influence
	#index is an array containing indices of the geometries - the indices are shuffled and when the training set is evaluated, index contains the indices of the training set, whereas it contains
	#the indices of the validation set, when the validation set is evaluated
	#for the cost function, the length of index is the number of all geometries
        allgeoms = len(index)
        nmstates = dict_properties['NumberOfStates']
        natoms = dict_properties['NumberOfAtoms']#l...factor, that controls the influence of the force errors relative to the energy errors during training 
        #l ... factor used to control the influence of the force errors (mse_gradient) relative to the energy errors during training 
        l = force_influence
        mse_energy = T.mean ( (self.prediction - target )**2 )
        mse_gradient = 0.0
        for geom in range((allgeoms)):
          #by using index[geom], the value of the entry in index is used, which ensures that the correct gradients are used from the validation and trainingsset
          grad_derived = dict_properties['grad_%i' %(index[geom])]
          grad = dict_properties['Gradient'][index[geom]]
          mse_gradient += (1./(3.*natoms)*np.sum( (grad - grad_derived)**2))
        mse_gradient = mse_gradient / allgeoms
        mse = mse_energy + l * mse_gradient
        #print mse, "mse ready"
        return mse
      elif quantity == "nac_grad":
        allgeoms = len(index)
        nmstates = dict_properties['NumberOfStates']
        natoms = dict_properties['NumberOfAtoms']#l...factor, that controls the influence of the force errors relative to the energy errors during training 
        #l ... factor used to control the influence of the force errors (mse_gradient) relative to the energy errors during training 
        l = force_influence
        mse_nac = T.mean ( (self.prediction - target )**2 )
        mse_gradient = 0.0
        for geom in range((allgeoms)):
          #by using index[geom], the value of the entry in index is used, which ensures that the correct gradients are used from the validation and trainingsset
          grad_derived = dict_properties['grad_%i' %(index[geom])]
          grad = dict_properties['Gradient'][index[geom]]
          mse_gradient += (1./(3.*natoms)*np.sum( (grad - grad_derived)**2))
        mse_gradient = mse_gradient / allgeoms
        mse = mse_nac + l * mse_gradient
        return mse
      else:
        mse = T.mean( ( self.prediction - target )**2 )
        return mse

    #writes the error separately for the validation of the energy and the forces
    def mse_error_valid( self, target, quantity, dict_properties, index ):
      #index is an array containing indices of the geometries - the indices are shuffled and when the training set is evaluated, index contains the indices of the training set, whereas it contains
      #the indices of the validation set, when the validation set is evaluated
      #for the cost function, the length of index is the number of all geometries
      allgeoms = len(index)
      nmstates = dict_properties['NumberOfStates']
      natoms = dict_properties['NumberOfAtoms']#l...factor, that controls the influence of the force errors relative to the energy errors during training  
      mse_energy = T.mean ( (self.prediction - target )**2 )
      mse_gradient = 0.0
      for geom in range(allgeoms):
        #by using index[geom], the value of the entry in index is used, which ensures that the correct gradients are used from the validation and trainingsset
        grad_derived = dict_properties['grad_%i' %(index[geom])]
        grad = dict_properties['Gradient'][index[geom]]
        mse_gradient += (1./(3.*natoms)*np.sum( (grad - grad_derived)**2))
      mse_gradient = mse_gradient / allgeoms
      return mse_energy, mse_gradient

    def load_weights( self, saved_params ):
      n_params = len( saved_params )
      if len( self.params ) != n_params:
        print( "Mismatch in number of network parameters" )
        exit()
      for entry in range( n_params ):
        self.params[entry].set_value( saved_params[entry].get_value(), borrow=True )
        #print self.params, "SELF PARAMS"



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
    if options[ "mode" ] == "train":
        train_all( options, weightfile, NNnumber )
    elif options[ "mode" ] == "run":
        run_all( options, weightfile, NNnumber )
    else:
        print( 'Wrong mode. Only "train" or "run" allowed.' )
        exit()
