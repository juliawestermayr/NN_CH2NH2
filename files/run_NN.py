## -*- coding: utf-8 -*-
#do all communication between SHARC and NN
from sys import argv
from NNtraining import train_all
from read_input import parse_input
from NNprediction import run_all
from process_input_output import QM_out
if __name__ == "__main__":
    #get name of input_file (currently "template.inp")
    try:
      name, input_file = argv
    except ValueError:
      print( "Usage: script <input_file>")
    filename = argv[1]
    options = parse_input( input_file, 'SH2NN.inp' )
    sharc_dat = options["datafile"]
    weightfile = options["weightpath"]
    NNnumber = options["NNnumber"]
    QMoutpath = options['QMoutpath']
    #path, where touch STOP will be executed, when the error of more NNs among each other is higher than a given threshold in options_run.input
    stoppath = options['stoppath']
    path = options["superiorfolder_path"]
    if options[ "mode" ] == "train":
      train_all( options, weightfile, NNnumber )
      exit
    if options[ "mode" ] == "train_analytical":
      train_analytical( options, weightfile, NNnumber )
      exit
    if options[ "mode" ] == "run_analytical":
      run_analytical( options, weightfile, NNnumber )
      exit
    if options[ "mode" ] == "run":
      run_all( options, weightfile, NNnumber )
      print( 'QM.out predicted by %s NN written to %s/QM.out' %(options['NNnumber'], options['QMoutpath']))
    if options[ "mode" ] == "optimization":
      train_all( options, weightfile, NNnumber )
      exit
    if options[ "mode" ] == "optimization_analytical":
      train_analytical( options, weightfile, NNnumber )
      exit
