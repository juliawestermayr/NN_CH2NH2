# -*- coding: utf-8 -*-
from sys import argv
import numpy as np
import os
#======================================
# Class for processing "QM.in"
#======================================


class QM_in:
    """all necessary information for NN extracted from QM.in-file
       STRUCTURE:
        self.n_atoms <number of atoms of molecule>
        self.atypes  <np-array of atom types>
        self.geoms  <np-array of x,y,z coordinates for every atom>
        self.descriptors <list of necessary descriptors for the molecule, taken from upper triangular descriptor matrix>
        
        def parse_quin( infile ) <returns number of atoms, list of atom types as np-array, list of geometries as np-array>
        def transform_coordinates( n_atoms, atypes, geoms ) <returns full descriptor matrix of molecule>
        def compute_dist_mat( geom ) <returns full distance matrix of molecule>
        def get_descriptors( n_atoms, atypes, geoms ) <returns upper triangular descriptor matrix as simple list>
        def write_xyz( n_atoms, atypes, geoms, outfilename ) <writes xyz-file that can be processed by NN>
        def write_descriptor_file( descriptors, outfilename ) <writes descriptor file that can be processed by NN>

    """

    def __init__( self, infilename, stepnumber, pathQMNN):
        self.n_atoms, self.atypes, self.geoms, self.states, self.nstates, self.nmstates, self.istates, self.spin, self.n_singlets, self.n_triplets = self.parse_qin( infilename, stepnumber, pathQMNN )
        self.descriptors = self.get_descriptors( self.n_atoms, self.atypes, self.geoms )
        self.descriptors_state = self.get_descriptors_state( self.n_atoms, self.atypes, self.geoms )
        #print "DESCRIPTORS", self.descriptors

  
    def parse_qin( self, filename, stepnumber, pathQMNN ):
        """open SHARC QM.in-file 
        return number of atoms, list of atom types, list of geometries
        """
        try:
          raw_data = open ( filename, 'r' ).readlines()
        except IOError:
            print(( "Could not open %s." % filename )) 
            exit ( 1 )
        #get number of atoms
        self.n_atoms = np.int( raw_data[0])

        #create list of atom types and list of corresponding x,y,z coordinates
        self.geoms = []
        self.atypes = []
        self.states = []
        #take a look for the number of steps that have already been calculated - if the max. number of steps given in the input and SH2NN.inp file is exceeded, remove the "RUN" file to stop the training- and running-process of the SHARC/NN
        for line in raw_data:
          if "step" in line:
            line=line.split()
            stepQMin=int(line[1])
            if stepQMin >= int(stepnumber):
              os.system('rm %s/RUN' %pathQMNN)
            else:
              pass
            break
        angstrom='False'
        for line in raw_data:
          if "unit angstrom" or "unit Angstrom" or "Unit Angstrom" or "Unit angstrom" in line:
            angstrom='True'
            break
        for line in raw_data:
          if "states" in line:
            s=line.split()[1:]
            break
        self.states=[ int(i) for i in s ]
        nm=0
        self.n_singlets=0
        self.n_triplets=0
        for i,n in enumerate(self.states):
          nm+=n*(i+1)
          if int(i) == int(0):
            self.n_singlets=n
          if int(i)==int(2):
            self.n_triplets=n
        self.nmstates=nm
        self.n_singlets=self.states[0]
        # Create list of indices for each state, for example 2 singulets generates the list 1 1, one triplett generates: 1 1 1: self.istates
       # Create list of indices which mark each state: self.nstates
        self.nstates = []
        self.istates=[]
        self.spin=[]
        for singlet in range(self.n_singlets):
          self.istates.append(singlet+1)
          self.nstates.append(1)
          self.spin.append(0)
        ms=-1
        for spins in range(3):
          for triplet in range(self.n_triplets):
            self.istates.append(triplet+1)
            self.nstates.append(3)
            self.spin.append(ms)
          ms+=1
        for curr_atom in range(self.n_atoms):
            atom_data = raw_data[curr_atom+2].strip().split()
            curr_atype = atom_data[0]
            if angstrom=='False':
              curr_geom = [float(coordinate) for coordinate in atom_data[1:4]]
              for xyz in range(3):
                curr_geom[xyz]=curr_geom[xyz]*0.529177211
            else:
              curr_geom = [float(coordinate) for coordinate in atom_data[1:4]]
            self.atypes.append(curr_atype)
            self.geoms.append(curr_geom)
        return self.n_atoms,np.array(self.atypes),np.array(self.geoms), self.states, self.nstates, self.nmstates, self.istates, self.spin, self.n_singlets, self.n_triplets


    def transform_coordinates( self, n_atoms, atypes, geoms ):
        """adapted from transform_coords.py
        return full descriptor matrix 
        """
        self.descr_mat = self.compute_dist_mat( geoms )
        np.fill_diagonal( self.descr_mat, 1.0 )
        self.descr_mat = 1.0 / self.descr_mat
        np.fill_diagonal( self.descr_mat, 0.0 )
        return self.descr_mat


    def compute_dist_mat( self, geom ):
        """adapted from transform_coords.py
        Fast distance matrix computation in numpy
        """
        m, n = geom.shape
        G = np.dot( geom, geom.T )
        H = np.tile( np.diag( G ), (m,1) )
        return np.sqrt( H + H.T - 2*G )


    def get_descriptors( self, n_atoms, atypes, geoms ):
        #returns list of elements of upper diagonal descriptor matrix as expected by NN

        descr_mat = self.transform_coordinates( n_atoms, atypes, geoms )
        return np.array([j for j in self.descr_mat[np.triu_indices(self.n_atoms,1)]])

    def get_descriptors_state( self, n_atoms, atypes, geoms ):
      """returns list of elements of upper diagonal descriptor matrix as expected by NN
      """
      descr_mat = self.transform_coordinates(n_atoms,atypes,geoms)
      des=np.array([j for j in self.descr_mat[np.triu_indices(self.n_atoms,0)]])
      state2=np.zeros((len(des),1))
      state2[:][:]=2.0
      state3=np.zeros((len(des),1))
      state3[:][:]=3.0
      state2=state2.flatten()
      state3=state3.flatten()
      des2=np.multiply(des,state2)
      des3=np.multiply(des,state3)
      descriptor_state=np.concatenate((des,des2,des3))
      descriptor_state=descriptor_state.flatten()
      return descriptor_state


    def  get_descriptors_coulomb_state2( self, n_atoms,atypes,geoms,charges):
      "gets coulomb matrix"
      descr_mat = self.transform_coordinates_coulomb(n_atoms,atypes,geoms,charges)
      des=np.array([j for j in self.descr_mat[np.triu_indices(self.n_atoms,0)]])
      state2=np.zeros((len(des),1))
      state2[:][:]=2.0
      state3=np.zeros((len(des),1))
      state3[:][:]=3.0
      state2=state2.flatten()
      state3=state3.flatten()
      des2=np.multiply(des,state2)
      des3=np.multiply(des,state3)
      descriptor_state=np.concatenate((des,des2,des3))
      descriptor_state=descriptor_state.flatten()
      return descriptor_state
    #this one is bad
    def  get_descriptors_coulomb_state1( self,n_atoms, atypes,geoms,charges):
      "gets coulomb matrix"
      descr_mat = self.transform_coordinates_coulomb(n_atoms,atypes,geoms,charges)
      des=np.array([j for j in self.descr_mat[np.triu_indices(self.n_atoms,0)]])
      descriptor_state=np.zeros((len(des)+3,1))
      for i in range(len(des)):
        descriptor_state[i]=des[i]
      descriptor_state[len(des)+0]=1.0
      descriptor_state[len(des)+1]=2.0
      descriptor_state[len(des)+2]=3.
      descriptor_state=descriptor_state.flatten()
      return descriptor_state


    def  get_descriptors_coulomb( self,filename,n_atoms):
      "gets coulomb matrix"
      descr_mat = self.transform_coordinates_coulomb(n_atoms,atypes,geoms,charges)
      des=np.array([j for j in self.descr_mat[np.triu_indices(self.n_atoms,0)]])
      return des


    def write_xyz( self, n_atoms, atypes, geoms, outfilename ):
        """writes xyz file for NN
        FILE FORMAT:
        <n-atoms>
        <dummy as line must not be empty>
        <atom_symbol> <x-coord> <y-coord> <z-coord>
            ...    
        """
        outfile = open( outfilename, 'w')
        outfile.write( "%d \n" % n_atoms )
        outfile.write( "dummyenergy \n") #read_xyz_file from transform_coords expects an energy in that line
        for curr_atom in range(n_atoms):
            outfile.write( "%s %12.9f %12.9f %12.9f\n" % ( atypes[curr_atom], geoms[curr_atom][0], geoms[curr_atom][1], geoms[curr_atom][2] ) )

    def write_descriptor_file( self, descriptors, outfilename ):
        outfile = open ( outfilename, 'w')
        outfile.write( "descr: %s" % ( ' '.join(["%12.7f" %j for j in descriptors]) ) ) 


#======================================
# Class for generating "QM.out"
#======================================

class QM_out:
    """
    data structure for storing all results from NN needed for writing 'QM.out', writing 'QM.out' on request

     STRUCTURE:
        self.quantities <dict, keys = quantities, values = string created from NN output, ready to be printed to 'QM.out'; is empty at the beginning, data is added by NN>
        self.order <list of quantities in order in which they should appear in 'QM.out'>
        
        def ( quantity, data ) <quantity = quantity name, data = output from NN for respective quantity; checks if function for transforming NN-output to QM.out-string is available and then calls it>
        def write_file() <writes 'QM.out'-file from strings stored in quantities{}>"""
    #            def _write_h( data ) <transforms energy values obtained by NN into hamiltonian-string needed for 'QM.out'    

    def __init__( self ):

    # implemented quantities
        self.quantity = {"Energy"			: "",
                         "SpinOrbitCoupling"		: "",
                         "Dipole"			: "",
                         "Nac"				: "",
                         "Gradient"			: "",
                         "Overlap"                      : "" }
      #  self.quantity = {}
        self.order = ['Hamiltonian', 'Dipole', 'Gradient', 'Nac', "Overlap"]

    def add_overlap( self, nmstates):
      self._write_overlap( nmstates )

    def add( self, quantity, name, NNoutput, nmstates, natoms, istates, nstates, spin, n_singlets, n_triplets,ZN ):
        """ adds quantity calculated by NN to data structure
            1.) check if transformation of quantity to 'QM.out'-string is available for given quantity
            2.) if available, calls respective function which returns string suitable for direct writing to 'QM.out' and adds this string to self.quantities{}
        """
        Stop = 'False'
        QMout={}
        name = 'NN_1'
        for q in quantity:
          if q == 'energy':
            self.energy=write_energy( NNoutput[name][q],nmstates, n_singlets, n_triplets )
            QMout['energy']=self.energy
          elif q == 'energy_grad':
            self.energy=self._write_energy( NNoutput[name][q]['energy'], nmstates, n_singlets, n_triplets )
            self.grad=self._write_grad( NNoutput[name][q]['gradient'], nmstates, natoms, istates, nstates, spin, n_singlets, n_triplets )
            QMout['hamiltonian'] = self.energy
            QMout['grad'] = self.grad
          elif q == 'soc':
            self.soc=self._write_soc( NNoutput[name][q], nmstates, n_singlets, n_triplets )
            QMout.update({'hamiltonian' : self.soc})
          elif q == 'dipole':
            self.dipole_x,self.dipole_y,self.dipole_z = self._write_dipole( NNoutput[name][q], nmstates, n_singlets, n_triplets )
            QMout['dipole_x']=self.dipole_x
            QMout['dipole_y']=self.dipole_y
            QMout['dipole_z']=self.dipole_z
          elif q == 'nac':
            self.nac=self._write_nac( NNoutput[name][q], nmstates, natoms, istates, nstates, spin, n_singlets, n_triplets )
            QMout['nac'] = self.nac
          elif q == 'overlap':
            pass
          elif q == 'grad':
            self.grad = self._write_grad( NNoutput[name][q], nmstates, natoms, istates, nstates, spin, n_singlets, n_triplets ) 
            QMout['grad']  = self.grad
          else:
            print(('Creating QM.out not implemented yet for quantity %s.') %(q))
            exit()
          #for scaling of nacs wrt energygaps
          #if "nac" in quantity:
            #self._write_nac_scale(NNoutput['NN_1']['nac'],NNoutput['NN_1']['energy'],nmstates,natoms,istates,nstates,spin,n_singlets,n_triplets)
        if ZN == True:
          self.nac = self._write_nac_ZNapproximation(NNoutput[name]['ZN']['Hessian'],NNoutput[name]['ZN']['hopping_direction'],istates,nstates,spin,nmstates,natoms,n_singlets,n_triplets)
          QMout['nac'] = self.nac
        return Stop,QMout

    def add2( self, options, NNoutput, nmstates, natoms, istates, nstates, spin,n_singlets, n_triplets,ZN ):
        """ adds quantity calculated by NN to data structure
            1.) check if transformation of quantity to 'QM.out'-string is available for given quantity
            2.) if available, calls respective function which returns string suitable for direct writing to 'QM.out' and adds this string to self.quantities{}
        """
        QMout={}
        stop = 'False'
        quantity=options['quantity']
        NNnumber = options['NNnumber']
        NNoutput_grad={}
        for q in quantity:
          if q == 'energy':
            NNoutput_new, stop=self.compareNNs(options, stop, NNoutput, q)
            self.energy = self._write_energy( NNoutput_new[q], nmstates, n_singlets, n_triplets )
            QMout['energy'] = self.energy
          if q == "soc":
            NNoutput_new, stop=self.compareNNs(options, stop, NNoutput, q)
            self.soc = self._write_soc( NNoutput_new[q], nmstates, n_singlets, n_triplets )
            QMout.update({'hamiltonian' : self.soc} )
          if q == "dipole":
            NNoutput_new, stop=self.compareNNs(options, stop, NNoutput, q)
            self.dipole_x,self.dipole_y,self.dipole_z = self._write_dipole( NNoutput_new[q], nmstates, n_singlets, n_triplets )
            QMout['dipole_x'] = self.dipole_x
            QMout['dipole_y'] = self.dipole_y
            QMout['dipole_z'] = self.dipole_z
          if q == "grad":
            NNoutput_new, stop=self.compareNNs(options, stop, NNoutput, q)
            self.grad = self._write_grad( NNoutput_new[q], nmstates, natoms, istates, nstates, spin )
            QMout['grad'] = self.grad
          if q == "nac":
            NNoutput_new, stop=self.compareNNs(options, stop, NNoutput, q)
            self.nac = self._write_nac( NNoutput_new[q], nmstates, natoms, istates, nstates, spin, n_singlets, n_triplets )
            QMout['nac'] = self.nac
          if q == "energy_grad":
            NNoutput_new, stop=self.compareNNs(options, stop, NNoutput, q)
            self.energy = self._write_energy( NNoutput_new['energy'], nmstates, n_singlets, n_triplets )
            self.grad = self._write_grad( NNoutput_new['grad'], nmstates, natoms, istates, nstates, spin, n_singlets, n_triplets )
            QMout['hamiltonian'] = self.energy
            QMout['grad' ] = self.grad
          if q == "overlap":
            pass
        if ZN == True:
          NNoutput_new,stop = self.compareNNs(options,stop,NNoutput,'ZN')
          self.nac = self._write_nac_ZNapproximation(NNoutput_new['Hessian'],NNoutput_new['hopping_direction'],istates,nstates,spin,nmstates,natoms,n_singlets,n_triplets)
          QMout['nac'] = self.nac
 
        return stop,QMout

    def compareNNs( self, options,stop, NNoutput,q):
      #This function should compare all NNs (NNnumber = number of used NNs) and calculate their difference of predicted values
      NNoutput_new={}
      Error={}
      Error['Mean']={}
      Error['Stabw']={}
      quantity = options['quantity']
      NNnumber = options['NNnumber']
      if q == 'overlap':
        pass
      elif q == 'energy_grad':
        Error['Mean']['energy'] = {}
        Error['Stabw']['energy'] = {}
        threshold_energy = options['threshold_energy']
        Error['Mean']['grad'] = {}
        Error['Stabw']['grad'] = {}
        prediction_sum_energy = 0.0
        prediction_sum_grad = 0.0
        prediction_mean_energy = 0.0
        prediction_mean_grad = 0.0
        pre_stabw_energy = 0.0
        pre_stabw_grad = 0.0
        prediction_stabw_energy = 0.0
        predictions_stabw_grad = 0.0
        #build the mean of the predicted energies and gradients by different neural networks
        for i in range(1, NNnumber+1):
          name = 'NN_%i' %(i)
          nnoutput_energy = NNoutput[name][q]['energy']
          nnoutput_grad = NNoutput[name][q]['gradient']
          prediction_sum_energy += nnoutput_energy
          prediction_sum_grad += nnoutput_grad
        prediction_mean_energy = prediction_sum_energy / NNnumber
        prediction_mean_grad = prediction_sum_grad / NNnumber
        Error['Mean']['energy'] = prediction_mean_energy
        Error['Mean']['grad'] = prediction_mean_grad
        #build the standard deviation of predicted energies and gradients by different neural networks
        #reference value to calculate error is the mean of NNs 
        for i in range(1, NNnumber+1):
          name = 'NN_%i' %(i)
          nnoutput_energy = NNoutput[name][q]['energy']
          nnoutput_grad = NNoutput[name][q]['gradient']
          pre_stabw_energy += (nnoutput_energy - Error['Mean']['energy'] )**2
          pre_stabw_grad += (nnoutput_grad - Error['Mean']['grad'] )**2
        prediction_stabw_energy = np.sqrt(1/(float(NNnumber)-1.0)*pre_stabw_energy)
        prediction_stabw_grad = np.sqrt(1/(float(NNnumber)-1.0)*pre_stabw_grad)
        Error['Stabw']['energy'] = prediction_stabw_energy
        Error['Stabw']['grad'] = prediction_stabw_grad
        error_energy=0.
        error_grad=0.
        error=0.
        energylen = len(nnoutput_energy)
        gradlen = len(nnoutput_grad)
        for t in range(energylen):
          error_energy+=Error['Stabw']['energy'][t]
        for g in range(gradlen):
          error_grad+=Error['Stabw']['grad'][g]
        error_energy = error_energy/energylen
        error_grad = error_grad/gradlen
        error = error_energy+error_grad
        if error >= threshold_energy:
          print( 'The error for Energy is too big! New trainings data from QM simulations need to be generated and the NNs need to be trained again to continue dynamics!' )
          stop='True'
        else:
          pass
        NNoutput_new['energy'] = Error['Mean']['energy']
        NNoutput_new['grad'] = Error['Mean']['grad']
      elif q == "ZN" :
        hopping_direction = 0
        hessian = 0
        for i in range(1,NNnumber+1):
          name= 'NN_%i' %(i)
          hopping_direction += NNoutput[name]['ZN']['hopping_direction']
          hessian += NNoutput[name]['ZN']['Hessian']
        hopping_direction = hopping_direction / NNnumber
        hessian = hessian / NNnumber
        NNoutput_new['hopping_direction'] = np.array(hopping_direction)
        NNoutput_new['Hessian'] = np.array(hessian)
      else:
        Error['Mean'][q]={}
        Error['Stabw'][q]={}
        threshold = options['threshold_%s' %(q)]
        #for each state get the sum of each predicted value from each NN
        prediction_sum=0.0
        prediction_mean=0.0
        pre_stabw=0.0
        prediction_stabw=0.0
        for i in range(1,NNnumber+1):
          name='NN_%i' %(i)
          nnoutput=NNoutput[name][q]
          #the length of the nnoutput gives the number of entries for each quantity.
          prediction_sum+=nnoutput
        prediction_mean=prediction_sum/NNnumber
        Error['Mean'][q]=prediction_mean
        for i in range(1,NNnumber+1):
          name='NN_%s' %(i)
          nnoutput=NNoutput[name][q]
          #this gives the sum of all matrices of the NNs minus the mean of all NN-matrices
          pre_stabw+=(nnoutput-Error['Mean'][q])**2
        prediction_stabw=np.sqrt(1/(float(NNnumber)-1.0)*pre_stabw)
        Error['Stabw'][q]=prediction_stabw
        error = 0.
        for t in range(len(nnoutput)):
          error+=Error['Stabw'][q][t]
        error=error/len(nnoutput)
        if error>=threshold:
          print(('The error for %s is too big! New trainings data from QM simulations need to be generated and the NNs need to be trained again to continue dynamics!' %(q)))
          stop='True'
        else:
          pass
        NNoutput_new[q]=Error['Mean'][q]
      return NNoutput_new, stop


    def _write_energy( self, data, nmstates, n_singlets, n_triplets ):
        """ returns string of hamiltonian matrix from energy output of NN, suitable for direct writing to 'QM.out'            
        """
        ## get data dimensions, create empty hamiltonian
        dim = nmstates
        hamiltonian_energy = np.zeros( (dim, dim ),dtype=complex ).tolist()
        # fill diagonal of hamiltonian with energy values
        #dubletts are ignored
        #fill energy of singlets
        for singlet in range(n_singlets):
          hamiltonian_energy[singlet][singlet] = complex(data[singlet],0.000)
        #fill energy of triplets
        for triplet in range(n_singlets,n_singlets+n_triplets):
          #ms=-1
          hamiltonian_energy[triplet][triplet] = complex(data[triplet],0.000)
          #ms=0
          hamiltonian_energy[triplet+n_triplets][triplet+n_triplets] = complex(data[triplet],0.000)
          #ms=+1
          hamiltonian_energy[triplet+2*n_triplets][triplet+2*n_triplets] = complex(data[triplet],0.000)
        self.quantity["Energymatrix"] = hamiltonian_energy
        # Create string for Hamiltonian matrix - if SOCs are given too, the entry "Hamiltonian" in the dictionary self.quantity will be 
        # overwritten with the Hamiltonian containing the energy + SOCs
        h_string = "! 1 Hamiltonian Matrix (%dx%d), complex \n %d %d \n" % ( dim,dim,dim,dim )
        for line in hamiltonian_energy:
            curr_str = ''
            for element in line: 
                curr_str = curr_str + "%20.12E %20.12E " %( np.real(element),np.imag(element))
            h_string += curr_str + "\n"  
        # add an empty line to string
        h_string = h_string + "\n" 
        self.quantity["Hamiltonian"]=h_string
        return hamiltonian_energy

    def _write_soc( self, data, nmstates, n_singlets, n_triplets ):
        dim = nmstates
        hamiltonian_energy = self.quantity["Energymatrix"]
        hamiltonian_soc = np.zeros( (dim, dim ),dtype=complex ).tolist()
        index=-1
        print(data.shape)
        for soc_line in range(nmstates):
          for soc_step in range(soc_line+1,nmstates):
            index+=1
            hamiltonian_soc[soc_line][soc_step] = complex(data[index*2],data[index*2+1])
        hamiltonian_energy=np.array(hamiltonian_energy)
        hamiltonian_soc=np.array(hamiltonian_soc)
        hamiltonian = hamiltonian_energy+ hamiltonian_soc
        self.quantity["HamiltonianMATRIX"]=hamiltonian
        hamiltonian=hamiltonian.tolist()
        # create string for QM.out
        h_string = "! 1 Hamiltonian Matrix (%dx%d), complex \n %d %d \n" % ( dim,dim,dim,dim )
        for line in hamiltonian:
            curr_str = ''
            for element in line:
                curr_str = curr_str + "%20.12E %20.12E " %(np.real(element),np.imag(element))
            h_string += curr_str + "\n"
        # add an empty line to string
        h_string = h_string + "\n"
        self.quantity["Hamiltonian"]=h_string
        return hamiltonian

    def _write_soc_old( self, data, nmstates, n_singlets, n_triplets ):
        dim = nmstates
        hamiltonian_energy = self.quantity["Energymatrix"]
        hamiltonian_soc = np.zeros( (dim, 2*dim ) ).tolist()
        #there are only n_singlets*n_triplets + 0.5(n_triplets(n_triplets-1)) values for SOCs - a,b,c,d,e,f - whereas a and b are : a+ib for S with T, c is S with T(ms=0), d is T with T (ms-1/ms-1) and e+if is T with T (ms-1/ms0 or ms+1/ms0)
        #the soc matrix will get filled with those values
        #the data matrix consists of a vecotr containing all real values and then all imaginary values - each vector represents one geometry
        a=0
        #this is the number of real values a and e:
        b=n_singlets*n_triplets+n_triplets*(n_triplets-1)/2
        b=int(b)
        for singlet in range(n_singlets):
          for soc1 in range(n_singlets,n_singlets+n_triplets):
            #fill a (real value) and b and c (imaginary values)
            #data contains all a first - the number of a's is n_triplets*n_singlets
            #upper triangular part
            hamiltonian_soc[singlet][2*soc1]=data[a]
            hamiltonian_soc[singlet][2*soc1+4*n_triplets]=data[a]
            #lower triangular part
            hamiltonian_soc[soc1][2*singlet]=data[a]
            hamiltonian_soc[soc1+2*n_triplets][2*singlet]=data[a]
            #fill imaginary values
            #upper triangular part
            hamiltonian_soc[singlet][2*soc1+1]=data[b]
            hamiltonian_soc[singlet][2*soc1+1+2*n_triplets]=data[b+n_triplets]
            #the imaginary values between singlets and triplets with ms=+1 are negative
            hamiltonian_soc[singlet][2*soc1+1+4*n_triplets]=-data[b]
            #lower triangular part
            hamiltonian_soc[soc1][2*singlet+1]=-data[b]
            hamiltonian_soc[soc1+n_triplets][2*singlet+1]=-data[b+n_triplets]
            hamiltonian_soc[soc1+2*n_triplets][2*singlet+1]=data[b]
            a+=1
            b+=1
          b+=n_triplets
        e_1=a
        e_2=a
        f_1=b
        f_2=b
        d_1=b
        d_2=b
        #now the matrix gets filled with couplings arising only from Triplets with Triplets
        for triplet1 in range(n_singlets,n_singlets+n_triplets-1):
          for soc2 in range(triplet1+1, n_singlets+n_triplets):
            #first lines of Singlets (ms-1 with ms +1,0,-1) after singlets of SOC matrix (the id, e+if values are diagonal matrices)
            #contains in the first block i*d (imaginary values), in the second e+i*f
            #upper triangular matrix
            hamiltonian_soc[triplet1][soc2*2+1]=data[d_1]
            hamiltonian_soc[triplet1][soc2*2+1+n_triplets*2]=data[f_1+n_triplets-triplet1-1+n_singlets]
            hamiltonian_soc[triplet1][soc2*2+n_triplets*2]=data[e_1]
            #lower triangular matrix
            hamiltonian_soc[soc2][triplet1*2+1]=-data[d_1]
            hamiltonian_soc[soc2][triplet1*2+1+n_triplets*2]=-data[f_1+n_triplets-triplet1-1+n_singlets]
            hamiltonian_soc[soc2][triplet1*2+n_triplets*2]=-data[e_1]
            #lower block matrix (T(ms=0) with T (ms=-1)
            #upper triangular part
            hamiltonian_soc[triplet1+n_triplets][soc2*2+1]=data[f_1+n_triplets-triplet1-1+n_singlets]
            hamiltonian_soc[triplet1+n_triplets][soc2*2]=-data[e_1]
            #lower triangular part
            hamiltonian_soc[soc2+n_triplets][triplet1*2+1]=-data[f_1+n_triplets-triplet1-1+n_singlets]
            hamiltonian_soc[soc2+n_triplets][triplet1*2]=data[e_1]
            e_1+=1
            f_1+=1
            d_1+=1
          f_1+=n_triplets-triplet1-1+n_singlets
          d_1+=n_triplets-triplet1-1+n_singlets
        #filling couplings of T with ms=0 and T with ms=+1
        for triplet2 in range(n_singlets+n_triplets,n_singlets+2*n_triplets):
          for soc3 in range(triplet2+1,n_singlets+2*n_triplets):
            #the diagonals are 0 and the off-diagonals contain e+if
            #upper triangular parts
            hamiltonian_soc[triplet2][soc3*2+n_triplets*2]=data[e_2]
            hamiltonian_soc[triplet2][soc3*2+1+n_triplets*2]=data[f_2+2*n_triplets-triplet2-1+n_singlets]
            hamiltonian_soc[triplet2+n_triplets][soc3*2]=-data[e_2]
            hamiltonian_soc[triplet2+n_triplets][soc3*2+1]=data[f_2+2*n_triplets-triplet2-1+n_singlets]
            #lower triangular parts
            hamiltonian_soc[soc3][triplet2*2+n_triplets*2]=-data[e_2]
            hamiltonian_soc[soc3][triplet2*2+1+n_triplets*2]=-data[f_2+2*n_triplets-triplet2-1+n_singlets]
            hamiltonian_soc[soc3+n_triplets][triplet2*2]=data[e_2]
            hamiltonian_soc[soc3+n_triplets][triplet2*2+1]=-data[f_2+2*n_triplets-triplet2-1+n_singlets]
            e_2+=1
            f_2+=1
          f_2+=2*n_triplets-triplet2-1+n_singlets
        #filling couplings of T with both ms=+1
        for triplet3 in range(n_singlets+2*n_triplets,n_singlets+3*n_triplets):
          for soc4 in range(triplet3+1,n_singlets+3*n_triplets):
            #upper triangular part
            hamiltonian_soc[triplet3][soc4*2+1]=-data[d_2]
            #lower triangular part
            hamiltonian_soc[soc4][triplet3*2+1]=data[d_2]
            d_2+=1
          d_2+=3*n_triplets-triplet3-1+n_singlets
        """for k in range((dim-1)*dim):
          if h >= 2*dim:
            l+=1
            h=2*l+2  
          #upper triangular matrix
          hamiltonian_soc_triu[l][h]=data[k]
          h+=1
        for k in range(dim*2):
          if i >= dim:
            j+=1
            i=j+1
          #lower triangular matrix
          hamiltonian_soc_tril[i][2*j]=data[2*k]
          hamiltonian_soc_tril[i][2*j+1]=data[2*k+1]
          i+=1"""

        #multiplicate every imaginary number of lower triangular matrix with -1
        hamiltonian_energy=np.array(hamiltonian_energy)
        hamiltonian_soc=np.array(hamiltonian_soc)
        hamiltonian = hamiltonian_energy+ hamiltonian_soc
        self.quantity["HamiltonianMATRIX"]=hamiltonian
        hamiltonian=hamiltonian.tolist()
        # create string for QM.out
        h_string = "! 1 Hamiltonian Matrix (%dx%d), complex \n %d %d \n" % ( dim,dim,dim,dim )
        for line in hamiltonian:
            curr_str = ''
            for element in line:
                curr_str = curr_str + "%20.12E" % element
            h_string += curr_str + "\n"
        # add an empty line to string
        h_string = h_string + "\n"
        self.quantity["Hamiltonian"]=h_string
        return hamiltonian

    def _write_dipole( self, data, nmstates, n_singlets, n_triplets ):
        dim = nmstates
        dipole_x = np.zeros( (dim, 2*dim) )
        dipole_y = np.zeros( (dim, 2*dim ) )
        dipole_z = np.zeros( (dim, 2*dim ) )
        #for matrix shape
        dipole_value_2=((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)
        #fill dipole matrix
        NNvalues=0
        for singlet in range(n_singlets):
          for dipole_values in range(singlet,n_singlets):
            #fill upper triangular part
            dipole_value_2=int(dipole_value_2)
            dipole_x[singlet][dipole_values*2]=data[NNvalues]
            dipole_y[singlet][dipole_values*2]=data[NNvalues+dipole_value_2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)]
            dipole_z[singlet][dipole_values*2]=data[NNvalues+dipole_value_2*2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)*2]
            #fill lower triangular part
            dipole_x[dipole_values][singlet*2]=data[NNvalues]
            dipole_y[dipole_values][singlet*2]=data[NNvalues+dipole_value_2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)]
            dipole_z[dipole_values][singlet*2]=data[NNvalues+dipole_value_2*2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)*2]
            NNvalues+=1
        for triplet in range(n_singlets,n_singlets+n_triplets):
          for dipole_values in range(triplet,n_singlets+n_triplets):
            #fill upper triangular part
            #ms=-1
            dipole_value_2=((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets) 
            dipole_value_2=int(dipole_value_2)
            dipole_x[triplet][dipole_values*2]=data[NNvalues]
            dipole_y[triplet][dipole_values*2]=data[NNvalues+dipole_value_2]
            dipole_z[triplet][dipole_values*2]=data[NNvalues+dipole_value_2*2]
            #ms=0
            dipole_x[triplet+n_triplets][dipole_values*2+2*n_triplets]=data[NNvalues]
            dipole_y[triplet+n_triplets][dipole_values*2+2*n_triplets]=data[NNvalues+dipole_value_2]
            dipole_z[triplet+n_triplets][dipole_values*2+2*n_triplets]=data[NNvalues+dipole_value_2*2]
            #ms=-1
            dipole_x[triplet+2*n_triplets][dipole_values*2+4*n_triplets]=data[NNvalues]
            dipole_y[triplet+2*n_triplets][dipole_values*2+4*n_triplets]=data[NNvalues+dipole_value_2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)]
            dipole_z[triplet+2*n_triplets][dipole_values*2+4*n_triplets]=data[NNvalues+dipole_value_2*2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)*2]
            #fill lower triangular part
            #ms=-1
            dipole_x[dipole_values][triplet*2]=data[NNvalues]
            dipole_y[dipole_values][triplet*2]=data[NNvalues+dipole_value_2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)]
            dipole_z[dipole_values][triplet*2]=data[NNvalues+dipole_value_2*2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)*2]
            #ms=0
            dipole_x[dipole_values+n_triplets][triplet*2+2*n_triplets]=data[NNvalues]
            dipole_y[dipole_values+n_triplets][triplet*2+2*n_triplets]=data[NNvalues+dipole_value_2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)]
            dipole_z[dipole_values+n_triplets][triplet*2+2*n_triplets]=data[NNvalues+dipole_value_2*2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)*2]
            #ms=-1
            dipole_x[dipole_values+2*n_triplets][triplet*2+4*n_triplets]=data[NNvalues]
            dipole_y[dipole_values+2*n_triplets][triplet*2+4*n_triplets]=data[NNvalues+dipole_value_2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)]
            dipole_z[dipole_values+2*n_triplets][triplet*2+4*n_triplets]=data[NNvalues+dipole_value_2*2]#((n_singlets-1)*n_singlets/2+(n_triplets-1)*n_triplets/2+n_singlets+n_triplets)*2]
            NNvalues+=1
        self.quantity["Dipole_x"] = dipole_x.tolist()
        self.quantity["Dipole_y"] = dipole_y.tolist()
        self.quantity["Dipole_z"] = dipole_z.tolist()
        # create string for QM.out
        d_string = "! 2 Dipole Moment Matrix (3x%dx%d), complex \n %d %d \n" % ( dim,dim,dim,dim )
        for line in dipole_x:
            curr_str = ''
            for element in line: 
                curr_str = curr_str + "%20.12E" % element
            d_string += curr_str + "\n"
        # states states after each matrix as numbers given
        d_string = d_string + "%d %d \n" % ( dim,dim )
        for line in dipole_y:
            curr_str = ''
            for element in line:
                curr_str = curr_str + "%20.12E" % element
            d_string += curr_str+"\n"
        d_string = d_string + "%d %d \n" % ( dim,dim )
        for line in dipole_z:
            curr_str = ''
            for element in line:
                curr_str = curr_str + "%20.12E" % element
            d_string += curr_str + "\n"
        self.quantity["Dipole"]=d_string 
        return self.quantity['Dipole_x'], self.quantity['Dipole_y'], self.quantity['Dipole_z']

    def _write_grad( self, data, nmstates, natoms, istates, nstates, spin, n_singlets, n_triplets ):
        #The gradient matrix is natosx3(xyz) for each state
        #convert gradient to H/Bohr
        dim = nmstates
        gradient= np.zeros((dim*natoms, 3))
        xyz = 0
        atom_state = 0
        # fill singlets
        for singlet in range(n_singlets*natoms*3):
          if xyz >= 3:
            xyz = 0
            atom_state += 1 
          gradient[atom_state][xyz]=data[singlet]*0.529177249
          xyz+=1
        #fill triplets
        xyz=0
        atom_state+=1
        for triplet in range(n_singlets*natoms*3,(n_singlets+n_triplets)*natoms*3):
          if xyz >= 3:
            xyz = 0
            atom_state +=1
          #ms=-1
          gradient[atom_state][xyz]=data[triplet]*0.529177249
          #ms=0
          gradient[atom_state+(natoms*n_triplets)][xyz]=data[triplet]*0.529177249
          #ms=+1
          gradient[atom_state+(natoms*2*n_triplets)][xyz]=data[triplet]*0.529177249
          xyz+=1
        self.quantity["GradientMatrix"]=gradient
        gradient = gradient.tolist()
        g_string = "! 3 Gradient Vectors (%dx%dx3, real) \n" % (dim, natoms)
        j=natoms
        i=0
        for i in range(dim*natoms):
          if j >= natoms:
            g_string = g_string + "%d %d ! state %d \n" % (natoms, 3, i/natoms+1) #here: states(k)
            j=0
          curr_str = ''
          for element in gradient[i][:]:
            curr_str = curr_str + "%20.12E" % element
          g_string += curr_str + "\n"
          j+=1
        self.quantity["Gradient"]=g_string
        return gradient

    def _write_nac_ZNapproximation( self, Hessian,hopping_direction, istates,nstates,spin,nmstates, natoms, n_singlets, n_triplets ):
        dim = nmstates
        nac = np.zeros((nmstates*nmstates*natoms, 3))
        index1=-1
        for i in range(nmstates):
          for j in range(nmstates):
            for iatom in range(natoms):
              index1+=1
              for xyz in range(3):
                nac[index1][xyz] = hopping_direction[i][j][iatom*3+xyz]

        self.quantity["NACMatrix"]=nac
        n_string = "! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (dim, dim, natoms)
        n_string = "! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (dim, dim, natoms)
        j=natoms
        dim2=dim*natoms
        dim2=int(dim2)
        dim3=dim*dim*natoms
        dim3=int(dim3)
        i=0
        k=0
        for i in range(dim3):
          if j>=natoms:
            #convert i/dim2 into integer
            value_i_dim2=i/dim2
            value_i_dim2=int(value_i_dim2)
            n_string = n_string + "%d %d ! %d %d %d %d %d %d \n" % (natoms, 3, nstates[value_i_dim2], istates[value_i_dim2], spin[value_i_dim2], nstates[k], istates[k], spin[k])
            k+=1
            j=0
            if k>=dim:
              k=0
          curr_str=''
          for element in nac[i][:]:
            curr_str = curr_str+ "%20.12E" % element
          n_string += curr_str +"\n"
          j+=1
        self.quantity["Nac"]=n_string
        return nac

    def _write_nac( self, data, nmstates, natoms, istates, nstates, spin, n_singlets, n_triplets ):
        dim = nmstates
        nac = np.zeros((dim*dim*natoms, 3))
        singlet=0
        nacstate_singlet=0
        matrixcounter=0
        nacmatrix=1
        singletstate2=0
        #fill all nacs of singlets
        while True:
          if singlet < n_singlets:
            for singletstate in range((singlet+1)*natoms,n_singlets*natoms):
              if singletstate2>=natoms:
                singletstate2=0
              for xyz in range(3):
                nac[singletstate+singlet*dim*natoms][xyz]=data[nacstate_singlet]
                #for state 1-2 write also state 2-1
                nac[singletstate2+dim*natoms*singlet+nacmatrix*dim*natoms+singlet*natoms][xyz]=-data[nacstate_singlet]
                nacstate_singlet+=1
                matrixcounter+=1
              singletstate2+=1
              if matrixcounter>=natoms*3:
                nacmatrix+=1
                matrixcounter=0
            singlet+=1
            nacmatrix=1
          else:
            break
        #write the matrices for triplets
        nacstate_triplet=nacstate_singlet
        matrixcounter=0
        nacmatrix=1
        tripletstate2=n_singlets*dim*natoms+n_singlets*natoms
        triplet=0
        while True:
          if triplet < n_triplets: 
            triplet_value_1=(n_singlets+triplet+1)*natoms+n_singlets*dim*natoms
            triplet_value_1=int(triplet_value_1)
            triplet_value_2=n_singlets*dim*natoms+(n_singlets+n_triplets)*natoms
            triplet_value_2=int(triplet_value_2)
            for tripletstate in range(triplet_value_1,triplet_value_2):
              if tripletstate2>=natoms+n_singlets*dim*natoms+n_singlets*natoms:
                tripletstate2=n_singlets*dim*natoms+n_singlets*natoms
              for xyz in range(3):
                #fill ms=-1
                nac[tripletstate+triplet*dim*natoms][xyz]=data[nacstate_triplet]
                nac[tripletstate2+dim*natoms*triplet+nacmatrix*dim*natoms+triplet*natoms][xyz]=-data[nacstate_triplet]
                #fill ms=0
                nac[tripletstate+(triplet+n_triplets)*dim*natoms+n_triplets*natoms][xyz]=data[nacstate_triplet]
                nac[tripletstate2+dim*natoms*triplet+nacmatrix*dim*natoms+(triplet+n_triplets)*natoms+n_triplets*dim*natoms][xyz]=-data[nacstate_triplet]
                #fill ms=+1
                nac[tripletstate+(triplet+2*n_triplets)*dim*natoms+2*n_triplets*natoms][xyz]=data[nacstate_triplet]
                nac[tripletstate2+dim*natoms*triplet+nacmatrix*dim*natoms+(triplet+2*n_triplets)*natoms+2*n_triplets*dim*natoms][xyz]=-data[nacstate_triplet]
                nacstate_triplet+=1
                matrixcounter+=1
              tripletstate2+=1
              if matrixcounter>=natoms*3:
                nacmatrix+=1
                matrixcounter=0
            triplet+=1
            nacmatrix=1
          else:
            break
        self.quantity["NACMatrix"]=nac
        n_string = "! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (dim, dim, natoms)
        n_string = "! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (dim, dim, natoms)
        j=natoms
        dim2=dim*natoms
        dim2=int(dim2)
        dim3=dim*dim*natoms
        dim3=int(dim3)
        i=0
        k=0
        for i in range(dim3):
          if j>=natoms:
            #convert i/dim2 into integer
            value_i_dim2=i/dim2
            value_i_dim2=int(value_i_dim2)
            n_string = n_string + "%d %d ! %d %d %d %d %d %d \n" % (natoms, 3, nstates[value_i_dim2], istates[value_i_dim2], spin[value_i_dim2], nstates[k], istates[k], spin[k])
            k+=1
            j=0
            if k>=dim:
              k=0
          curr_str=''
          for element in nac[i][:]:
            curr_str = curr_str+ "%20.12E" % element
          n_string += curr_str +"\n"
          j+=1
        self.quantity["Nac"]=n_string
        return nac

    def _write_nac_scale( self, data,data_energy, nmstates, natoms, istates, nstates, spin, n_singlets, n_triplets ):
        dim = nmstates
        nac = np.zeros((dim*dim*natoms, 3))
        singlet=0
        nacstate_singlet=0
        matrixcounter=0
        nacmatrix=1
        singletstate2=0
        #fill all nacs of singlets
        while True:
          if singlet < n_singlets:
            for singletstate in range((singlet+1)*natoms,n_singlets*natoms):
              if singletstate2>=natoms:
                singletstate2=0
              for xyz in range(3):
                nac[singletstate+singlet*dim*natoms][xyz]=data[nacstate_singlet]
                #for state 1-2 write also state 2-1
                nac[singletstate2+dim*natoms*singlet+nacmatrix*dim*natoms+singlet*natoms][xyz]=-data[nacstate_singlet]
                nacstate_singlet+=1
                matrixcounter+=1
              singletstate2+=1
              if matrixcounter>=natoms*3:
                nacmatrix+=1
                matrixcounter=0
            singlet+=1
            nacmatrix=1
          else:
            break
        #write the matrices for triplets
        nacstate_triplet=nacstate_singlet
        matrixcounter=0
        nacmatrix=1
        tripletstate2=n_singlets*dim*natoms+n_singlets*natoms
        triplet=0
        while True:
          if triplet < n_triplets: 
            triplet_value_1=(n_singlets+triplet+1)*natoms+n_singlets*dim*natoms
            triplet_value_1=int(triplet_value_1)
            triplet_value_2=n_singlets*dim*natoms+(n_singlets+n_triplets)*natoms
            triplet_value_2=int(triplet_value_2)
            for tripletstate in range(triplet_value_1,triplet_value_2):
              if tripletstate2>=natoms+n_singlets*dim*natoms+n_singlets*natoms:
                tripletstate2=n_singlets*dim*natoms+n_singlets*natoms
              for xyz in range(3):
                #fill ms=-1
                nac[tripletstate+triplet*dim*natoms][xyz]=data[nacstate_triplet]
                nac[tripletstate2+dim*natoms*triplet+nacmatrix*dim*natoms+triplet*natoms][xyz]=-data[nacstate_triplet]
                #fill ms=0
                nac[tripletstate+(triplet+n_triplets)*dim*natoms+n_triplets*natoms][xyz]=data[nacstate_triplet]
                nac[tripletstate2+dim*natoms*triplet+nacmatrix*dim*natoms+(triplet+n_triplets)*natoms+n_triplets*dim*natoms][xyz]=-data[nacstate_triplet]
                #fill ms=+1
                nac[tripletstate+(triplet+2*n_triplets)*dim*natoms+2*n_triplets*natoms][xyz]=data[nacstate_triplet]
                nac[tripletstate2+dim*natoms*triplet+nacmatrix*dim*natoms+(triplet+2*n_triplets)*natoms+2*n_triplets*dim*natoms][xyz]=-data[nacstate_triplet]
                nacstate_triplet+=1
                matrixcounter+=1
              tripletstate2+=1
              if matrixcounter>=natoms*3:
                nacmatrix+=1
                matrixcounter=0
            triplet+=1
            nacmatrix=1
          else:
            break
        scaling_i=-1
        nac_value_count=-1
        for i in range(nmstates):
          scaling_i+=1
          scaling_j=-1
          for j in range(nmstates):
            scaling_j+=1
            E_i=float(data_energy[i])
            E_j=float(data_energy[j])
            delta_E=np.abs(E_i-E_j)
            if int(scaling_i)==int(scaling_j):
              delta_E=int(1)
            for k in range(natoms):
              nac_value_count+=1
              for l in range(3):
                nac[nac_value_count][l]=nac[nac_value_count][l]#/delta_E
        self.quantity["NACMatrix"]=nac
        n_string = "! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (dim, dim, natoms)
        n_string = "! 5 Non-adiabatic Couplings (ddr) (%dx%dx%dx3, real) \n" % (dim, dim, natoms)
        j=natoms
        dim2=dim*natoms
        dim2=int(dim2)
        dim3=dim*dim*natoms
        dim3=int(dim3)
        i=0
        k=0
        for i in range(dim3):
          if j>=natoms:
            #convert i/dim2 into integer
            value_i_dim2=i/dim2
            value_i_dim2=int(value_i_dim2)
            n_string = n_string + "%d %d ! %d %d %d %d %d %d \n" % (natoms, 3, nstates[value_i_dim2], istates[value_i_dim2], spin[value_i_dim2], nstates[k], istates[k], spin[k])
            k+=1
            j=0
            if k>=dim:
              k=0
          curr_str=''
          for element in nac[i][:]:
            curr_str = curr_str+ "%20.12E" % element
          n_string += curr_str +"\n"
          j+=1
        self.quantity["Nac"]=n_string
        return nac

    def _write_overlap( self, nmstates):
        "For Analytical Potential, write unit matrix as overlap matrix"
        dim = nmstates
        overlap_matrix = np.zeros( ( dim, 2*dim ) ).tolist()
        i=0
        for i in range(dim):
          overlap_matrix[i][2*i] = 1
          i+=1
        o_string = "! 6 Overlap matrix (%dx%d), complex \n %d %d \n" %( dim, dim, dim, dim )
        for line in overlap_matrix:
          curr_str = ''
          for element in line:
            curr_str = curr_str + "%20.12E" % element
          o_string += curr_str + "\n"
        o_string = o_string + "\n"
        self.quantity["Overlap"]=o_string


    def _write_file( self, filename ):
        """write all collected strings in right order to filename ('QM.out' by default)
        """
        outfile = open( filename , "w" )
        QM_outstring = ''
        for i in range(len(self.order)):
          QM_outstring += self.quantity[self.order[i]]
        outfile.write(QM_outstring)
        outfile.close()

if __name__ == "__main__":
    data = QM_out()
    data.add( "energy", [ -1,2,3,4,5] )
    data.add( "soc" )



