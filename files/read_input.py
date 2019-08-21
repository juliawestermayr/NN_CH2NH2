# -*- coding: utf-8 -*-

import numpy as np
import os
import copy
from sys import argv
#trial

def checkscratch(SCRATCHDIR):
    '''Checks whether SCRATCHDIR is a file or directory. If a file, it quits with exit code 1, if its a directory, it passes. If SCRATCHDIR does not exist, tries to create it.

    Arguments:
    1 string: path to SCRATCHDIR'''

    exist=os.path.exists(SCRATCHDIR)
    if exist:
        isfile=os.path.isfile(SCRATCHDIR)
        if isfile:
            print( '$SCRATCHDIR=%s exists and is a file!' % (SCRATCHDIR))
            sys.exit(16)
    else:
        try:
            os.makedirs(SCRATCHDIR)
        except OSError:
            print( 'Can not create SCRATCHDIR=%s\n' % (SCRATCHDIR))
            sys.exit(17)

#================================
# Read input file
#================================

def parse_input( input_file, SH2NN_file ):
    raw_inp = open( input_file, 'r' ).readlines()
    SH2NN_inp = open( SH2NN_file, 'r').readlines()
    options = { 'NNnumber'              : 1,
	        'superiorfolder_path'   : os.getcwd(),
                'QMin'                  : 'QM.in',
                'QMout'                 : 'QM.out',
                'weightpath'            : os.getcwd()}

    for line in raw_inp: 
      line = line.strip()
      if line.startswith( "NumberOfNNs" ):
        line = line.split()
        NNnumber = int(line[1])
        options["NNnumber"]=NNnumber
        NNnr=NNnumber+1
      elif line.startswith( "quantities" ):
        line = line.split()
        options["quantity"] = line[1:]
        options['threshold_meangrad2']='False'
        if "energy_grad" in options['quantity'] and "grad" in options['quantity']:
          options.update( {'threshold_meangrad2' : 'True',
         'threshold_meangrad'  :  False } )
      elif line.startswith( 'gradient_threshold' ):
        line = line.split()
        options['threshold_meangrad'] = float ( line[1] )
      elif line.startswith( "superiorfolder_path" ):
        line = line.split()
        options["superiorfolder_path"] = line[1]
      elif line.startswith( "controlfile" ):
        line = line.split()
        options["controlfile"] = line[1]
      elif line.startswith( "QMin" ):
        line = line.split()
        options["QMin"] = line[1]
      elif line.startswith( "QMout" ): 	
        line = line.split()
        options["QMout"] = line[1]
      elif line.startswith( "improvement_threshold_energy" ):
        line = line.split()
        options["improvement_threshold_energy"] = float(line[1])
      elif line.startswith( "improvement_threshold_soc" ):
        line = line.split()
        options["improvement_threshold_soc"] = float(line[1])
      elif line.startswith( "improvement_threshold_dipole" ):
        line = line.split()
        options["improvement_threshold_dipole"] = float(line[1])
      elif line.startswith( "improvement_threshold_nac" ):
        line = line.split()
        options["improvement_threshold_nac"] = float(line[1])
      elif line.startswith( "improvement_threshold_gradient" ):
        line = line.split()
        options["improvement_threshold_grad"] = float( line[1] )
      elif line.startswith( "force_influence" ):
        line = line.split()
        options["force_influence"] = float (line[1])
      elif line.startswith( "#" ):
        continue
    for i in range(1,NNnr):
      options.update( { 'NN%i_seed' %(i) : 1234,
      'NN%i_split' %(i)        : 0.9,
      'NN%i_n_epochs' %(i)     : False,
      'NN%i_batch' %(i)        : False} )
      for line in raw_inp:
          line = line.strip()
          if line.startswith( "NN%i_decay_steps_energy" %(i)):
            line = line.split()
            options['NN%i_decay_steps_energy' %(i)] = float(line[1])
          elif line.startswith( "NN%i_decay_factor_energy" %(i) ):
            line = line.split()
            options['NN%i_decay_factor_energy'%(i)] = float(line[1])
          elif line.startswith( "NN%i_decay_steps_soc" %(i)):
            line = line.split()
            options['NN%i_decay_steps_soc' %(i)] = float(line[1])
          elif line.startswith( "NN%i_decay_factor_soc" %(i) ): 
            line = line.split()
            options['NN%i_decay_factor_soc' %(i)] = float(line[1])
          elif line.startswith( "NN%i_decay_steps_dipole" %(i)):
            line = line.split()
            options['NN%i_decay_steps_dipole'%(i)] = float(line[1])
          elif line.startswith( "NN%i_decay_factor_dipole" %(i)):
            line = line.split()
            options['NN%i_decay_factor_dipole' %(i)] = float(line[1])
          elif line.startswith( "NN%i_decay_steps_nac" %(i) ):
            line = line.split()
            options['NN%i_decay_steps_nac' %(i)] = float(line[1])
          elif line.startswith( "NN%i_decay_factor_nac" %(i)):
            line = line.split()
            options['NN%i_decay_factor_nac' %(i)] = float(line[1])
          elif line.startswith("NN%i_seed" %(i)):
            line = line.split()
            options["NN%i_seed" %(i)] =  int( line[1] )
          elif line.startswith("NN%i_split" %(i) ):
            line = line.split()
            options["NN%i_split" %(i)] = float( line[1] )  
          elif line.startswith( "NN%i_epochs" %(i) ):
            line = line.split()
            options["NN%i_n_epochs" %(i)] = int( line[1] )
          elif line.startswith( "NN%i_batch" %(i)):
            line = line.split()
            options["NN%i_batch" %(i)] = int( line[1] )
          elif line.startswith("NN%i_size_energy" %(i) ):
            options.update( { "NN%i_size_energy" %(i)     : False,
             "NN%i_act_energy" %(i)	      : False,
             "NN%i_L2_reg_energy" %(i)	      : False,
             "NN%i_learn_energy" %(i)	      : False} )
            line = line.split()
            options["NN%i_size_energy" %(i)] = [ int(j) for j in  line[1:] ]
          elif line.startswith("NN%i_act_energy" %(i) ):
            line = line.split()
            options["NN%i_act_energy" %(i)] = [ j for j in  line[1:] ]
          elif line.startswith("NN%i_L2_reg_energy" %(i) ):
            line = line.split()
            options["NN%i_L2_reg_energy" %(i)] = float( line[1] )
          elif line.startswith( "NN%i_learn_energy" %(i) ):
            line = line.split()
            options["NN%i_learn_energy" %(i)] = float ( line[1] )
          elif line.startswith( 'NN%i_hypergradient_learningrate_energy' %(i) ):
            line = line.split()
            options['NN%i_learn2_energy' %(i)] = float ( line[1] )
          elif line.startswith("NN%i_size_soc" %(i) ):
            quantity='soc'
            options.update( { "NN%i_size_%s" %(i, quantity) : False,
                "NN%i_act_%s" %(i, quantity)                : False,
                "NN%i_L2_reg_%s" %(i, quantity)             : False,
                "NN%i_learn_%s" %(i, quantity)              : False,
                'NN%i_learn2_%s' %(i, quantity)		    : False} )
            line = line.split()
            options["NN%i_size_%s" %(i, quantity)] = [ int(j) for j in  line[1:] ]
          elif line.startswith("NN%i_act_soc" %(i) ):
              line = line.split()
              options["NN%i_act_soc" %(i)] = [ j for j in  line[1:] ]
          elif line.startswith("NN%i_L2_reg_soc" %(i) ):
            line = line.split()
            options["NN%i_L2_reg_soc" %(i)] = float( line[1] )
          elif line.startswith( "NN%i_learn_soc" %(i) ):
            line = line.split()
            options["NN%i_learn_soc" %(i)] = float ( line[1] )
          elif line.startswith( 'NN%i_hypergradient_learningrate_soc' %(i) ):
            line = line.split()
            options['NN%i_learn2_soc' %(i)] = float ( line[1] )
          elif line.startswith("NN%i_size_dipole" %(i) ):
            quantity='dipole'
            options.update( { "NN%i_size_%s" %(i, quantity) : False,
                "NN%i_act_%s" %(i, quantity)                : False,
                "NN%i_L2_reg_%s" %(i, quantity)             : False,
                "NN%i_learn_%s" %(i, quantity)              : False,
                'NN%i_learn2_%s' %(i, quantity)		    : False})
            line = line.split()
            options["NN%i_size_dipole" %(i)] = [ int(j) for j in  line[1:] ]
          elif line.startswith("NN%i_act_dipole" %(i) ):
              line = line.split()
              options["NN%i_act_dipole" %(i)] = [ j for j in  line[1:] ]
          elif line.startswith("NN%i_L2_reg_dipole" %(i) ):
            line = line.split()
            options["NN%i_L2_reg_dipole" %(i)] = float( line[1] )
          elif line.startswith( "NN%i_learn_dipole" %(i) ):
            line = line.split()
            options["NN%i_learn_dipole" %(i)] = float ( line[1] )
          elif line.startswith( 'NN%i_hypergradient_learningrate_dipole' %(i) ):
            line = line.split()
            options['NN%i_learn2_dipole' %(i)] = float ( line[1] )
          elif line.startswith("NN%i_size_nac" %(i) ):
            quantity='nac'
            options.update( { "NN%i_size_%s" %(i, quantity) : False,
                "NN%i_act_%s" %(i, quantity)                : False,
                "NN%i_L2_reg_%s" %(i, quantity)             : False,
                "NN%i_learn_%s" %(i, quantity)              : False,
                'NN%i_learn2_%s' %(i, quantity)		    : False} )
            line = line.split()
            options["NN%i_size_nac" %(i)] = [ int(j) for j in  line[1:] ]
          elif line.startswith("NN%i_act_nac" %(i) ):
              line = line.split()
              options["NN%i_act_nac" %(i)] = [ j for j in  line[1:] ]
          elif line.startswith("NN%i_L2_reg_nac" %(i) ):
            line = line.split()
            options["NN%i_L2_reg_nac" %(i)] = float( line[1] )
          elif line.startswith( "NN%i_learn_nac" %(i) ):
            line = line.split()
            options["NN%i_learn_nac" %(i)] = float ( line[1] ) 
          elif line.startswith( 'NN%i_hypergradient_learningrate_nac' %(i) ):
            line = line.split()
            options['NN%i_learn2_nac' %(i)] = float ( line[1] )
          elif line.startswith("NN%i_size_grad" %(i) ):
            quantity='grad'
            options.update( { "NN%i_size_%s" %(i, quantity) : False,
                "NN%i_act_%s" %(i, quantity)                : False,
                "NN%i_L2_reg_%s" %(i, quantity)             : False,
                "NN%i_learn_%s" %(i, quantity)              : False,
                'NN%i_learn2_%s' %(i, quantity)		    : False} )
            line = line.split()
            options["NN%i_size_grad" %(i)] = [ int(j) for j in  line[1:] ]
          elif line.startswith("NN%i_act_grad" %(i) ):
              line = line.split()
              options["NN%i_act_grad" %(i)] = [ j for j in  line[1:] ]
          elif line.startswith("NN%i_L2_reg_grad" %(i) ):
            line = line.split()
            options["NN%i_L2_reg_grad" %(i)] = float( line[1] )
          elif line.startswith( "NN%i_learn_grad" %(i) ):
            line = line.split()
            options["NN%i_learn_grad" %(i)] = float ( line[1] ) 
          elif line.startswith( 'NN%i_hypergradient_learningrate_grad' %(i) ):
            line = line.split()
            options['NN%i_learn2_grad' %(i)] = float ( line[1] )
          elif line.startswith( "#" ):
            continue
    for line in SH2NN_inp:
      line = line.strip()
      if line.startswith( 'mode' ):
        line = line.split()
        options['mode']=line[1]
      elif line.startswith( 'weightpath' ):
        line = line.split()
        options['weightpath'] = line[1]
      elif line.startswith( 'QMin' ):
        line = line.split()
        options['QMinpath'] = line[1]
      elif line.startswith( 'QMout' ):
        line = line.split()
        options['QMoutSHARC'] = line[1]
      elif line.startswith( 'pathQMout' ):
        line = line.split()
        options['QMoutpath'] = line[1] 
        options['QMin'] = '%s/QM.in' %options['QMoutpath']
      elif line.startswith( 'stoppath' ):
        line = line.split()
        options['stoppath'] = line[1]
      elif line.startswith( "datafile" ):
        line = line.split()
        options["datafile"] = line[1]
      elif line.startswith( "tmax" ):
        line = line.split()
        options["tmax"] = line[1]
      elif line.startswith( "dt" ):
        line = line.split()
        options["dt"] = line[1]
      elif line.startswith( 'numberofmolecules'):
        line = line.split()
        options['nmolec']=int(line[1])
      elif line.startswith( 'ZhuNakamura'):
        line = line.split()
        options['ZhuNakamura']=int(line[1])
      elif line.startswith( '#' ):
        continue

    if options['NNnumber']>1:
      for line in raw_inp:
        line = line.strip()
        if line.startswith('threshold_energy'):
          options.update( { 'threshold_energy'  : False })
          line = line.split()
          options['threshold_energy'] = float( line[1] )
        elif line.startswith('threshold_soc'):
          options.update( { 'threshold_soc'  : False })
          line = line.split()
          options['threshold_soc'] = float( line[1] )
        elif line.startswith('threshold_dipole'):
          options.update( { 'threshold_dipole'  : False })
          line = line.split()
          options['threshold_dipole'] = float( line[1] )
        elif line.startswith('threshold_nac'):
          options.update( { 'threshold_nac'  : False })
          line = line.split()
          options['threshold_nac'] = float( line[1] )
        elif line.startswith('threshold_grad'):
          options.update( { 'threshold_grad'  : False })
          line = line.split()
          options['threshold_grad'] = float( line[1] )
        elif line.startswith('#'):
          continue

    if options['mode'] == 'train':
      for option in options.keys():
        if not options[option] and option != "controlfile":
          print( "Missing entry for %s" %( option ) )
          exit()
    elif options['mode'] == 'train_analytical':
      for option in options.keys():
        if not options[option] and option != "controlfile":
          print( "Missing entry for %s" %( option ) )
          exit()
    elif options['mode'] == 'run_analytical':
      for option in options.keys():
        if not options[option] and option != "controlfile":
          print( "Missing entry for %s" %( option ) )
          exit()
    elif options['mode'] == 'optimization':
      for option in options.keys():
        if not options[option] and option != "controlfile":
          print( "Missing entry for %s" %(option) ) 
          exit()
    elif options['mode'] == 'optimization_analytical':
      for option in options.keys():
        if not options[option] and option != "controlfile":
          print( "Missing entry for %s" %(option) )
          exit()
    elif options['mode'] == 'run':
      nonreq_options = ["controlfile", "datafile", "split"]
      for option in options.keys():
        if not options[option] and option not in nonreq_options:
          print( "Missing entry for %s" %( option ) )
          exit()
    elif options['mode'] == 'run2' or options['mode'] == 'run3':
      nonreq_options = ['controlfile']
      for option in options.keys():
        if not options[option] and option not in nonreq_options:
          print( "Missing entry for %s" %( option ) )
          exit()
    else:
      print("Mode not valid.")
      exit()

    for quantity in options["quantity"]:
      if quantity == "overlap":
        pass
      elif quantity == "nac_grad":
        pass
      elif "energy_grad" in quantity or "energy" in quantity:
        if ( len( options["NN1_size_energy"] ) + 1 ) != len( options["NN1_act_energy"] ):
          print( "Mismatch in network dimensions and activation functions for Energy" )
          exit()
      else:
        if ( len( options["NN1_size_%s"%quantity] ) + 1 ) != len( options["NN1_act_%s"%quantity] ):
          print( "Mismatch in network dimensions and activation functions for %s" %quantity )
          exit()
    return options



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
