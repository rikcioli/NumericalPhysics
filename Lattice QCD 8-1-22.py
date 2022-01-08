# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 19:39:58 2021

@author: rikci
"""

import numpy as np
import copy
from scipy import linalg
import tqdm

class LatticeQCD:
    """Path integral Lattice QCD"""

    def __init__(self, N_lat = 8, N_config = 100, N_cor = 50, eps = 0.24, u0to4 = 0.5):
        """
        4-dimensional hypercubic lattice with SU(3) matrices as links on each site and PBC.

        Parameters
        ----------
        N_lat : int, optional
            Number of lattice sites by each side. The default is 8.
        N_config : int, optional
            Number of different configurations. The default is 20.
        N_cor : int, optional
            Number of configurations to discard in thermalization loops to reduce
            correlations between each configuration and the next. The default is 30.
        eps : float, optional
            Parameter which controls the magnitude of the shift in the metropolis algorithm.
            The default is 0.24.
        ----------
        rng : numpy Generator
            Random number generator used by the class.
        N_rnd_matrices : int
            Number of random SU(3) matrices generated at the beginning.
        random_matrices : numpy.ndarray
            Array of SU(3) matrices: the first index chooses the matrix, the other 2 line and column.
        ----------
        Lambda1 : numpy.ndarray
            Collection of all the links on each site, as a 7-indices array.
            The first four indices specify the lattice site. The fifth is for the link direction.
            The last two specify line and column of the link.
            On each site there are 4 directions. The links specified by a given site always point towards
            increasing values of the coordinates. To consider a link pointing towards
            decreasing values of the coordinates, the inverse matrix associated 
            to the neighbouring site is computed instead.
        arr_of_paths : list of numpy.ndarray
            List of configurations. Each configuration is a given Lambda1.
        ----------
        beta : float
            Term which factorizes in front of the action. Corresponds to 6/g^2
        u0to4 : float
            Tadpole correction to each plaquette operator. Has a trial value at the beginning,
            gets evaluated by the program as the mean value of the plaquette operator
            over the ground state.
        acceptance : float
            Fraction of successful hits of the metropolis algorithm. Initialized to 0.
        """

        #parameters inserted by creating the class object
        self.N_lat = N_lat              
        self.N_cf = N_config            
        self.N_cor = N_cor        
        self.eps = eps
        
        #initialize the random matrices to use for the metropolis algorithm
        self.rng = np.random.default_rng()
        self.N_rnd_matrices = 200       
        self.random_matrices = np.zeros([self.N_rnd_matrices, 3, 3], dtype=complex)
        
        #initialize the lattice
        self.Lambda1 = np.zeros([N_lat, N_lat, N_lat, N_lat, 4, 3, 3], dtype=complex)        
        self.arr_of_paths = []
        
        #parameters related to the action
        self.beta = 2.75
        self.u0to4 = u0to4
        self.acceptance = 0.
        
            
    def compute_random_matrices(self):
        """
        Computes a pool of N_rnd_matrices random SU(3) matrices for the metropolis algorithm.
        Any SU(3) matrix U can be obtained by exponentiating a linear combination of the
        generators of the su(3) algebra. Such linear combination is a traceless hermitian
        matrix H, whose components are random complex numbers with real and imaginary part
        between -eps and +eps.
        Both U and its inverse (hermitian conjugate) are stored in the self.random_matrices array
        """
        
        id_mat = np.zeros([3,3])            
        for i in range(3):
            id_mat[i][i]=1                  #useful identity matrix
            
        for k in range(0, self.N_rnd_matrices, 2):
            
            Re_M, Im_M = 2*self.rng.random((3,3)) - 1, 2*self.rng.random((3,3)) - 1
            M = Re_M + Im_M*1j
            H = (M + M.T.conjugate())/2                    #random hermitian matrix
            H = H - np.trace(H)*id_mat/3                #obtain traceless hermitian matrix   
                
            U = linalg.expm(self.eps*H*1j)             #unitary matrix, obtained by exponentiating i*eps*H         
            self.random_matrices[k] = U             #storing both the matrix and its hermitian conjugate
            self.random_matrices[k+1] = U.T.conjugate()


    def cold_start(self):
        """
        Initialize all links to identity matrices.

        Returns
        -------
        None.

        """
        for i in range(self.N_lat):          #indices on lattice points
            for j in range(self.N_lat):
                for k in range(self.N_lat):
                    for l in range(self.N_lat):
                        for mu in range(4):                 #index on direction
                            for p in range(3):              #SU(3) matrix index
                                self.Lambda1[i][j][k][l][mu][p][p] = 1
                                
                        
    def initialize(self):
        self.compute_random_matrices()
        self.cold_start()

    
    
    def link(self, path, lattice = None):
        """
        Given an array of ordered points on the lattice, each specified by 4 numbers and
        each being separated from the previous and the next one by no more 
        than a lattice spacing (they have to be neighbours), this method returns the matrix
        product between the links that connect such points, in the given order.
        
        Parameters
        ----------
        path : list of int
            List of points specifying a path through the lattice sites.
        lattice : numpy.ndarray, optional
            Configuration on which to evaluate the product. The default is None,
            meaning the product is evaluated using the current configuration stored
            in self.Lambda1

        Raises
        ------
        ImportError
            Raises an error when the inserted points do not form a path of
            neighbouring points.

        Returns
        -------
        numpy.ndarray
            Returns the requested matrix product as a 2d numpy array.

        """
        if lattice is None:
            lattice = self.Lambda1
        
        #initialize list of matrices to multiply
        matrices = []
        
        for link_N in range(len(path)-1):                #for each pair of points
            #evaluate displacement vector
            f_point = path[link_N+1]
            i_point = path[link_N]
            delta = f_point - i_point

            #check if points are neighbours
            sum_delta = delta[0] + delta[1] + delta[2] + delta[3]               
            if sum_delta!=1 and sum_delta!=-1:
                raise ImportError(
                    """Non valid sequence of points inserted; please insert ordered, neighbouring points"""
                    )

            #find which component of displacement vector is non zero
            direction = 0
            while delta[direction] == 0:
                direction += 1
                
            #if the displacement is towards increasing coordinates, return the link as it is
            #if it is in the opposite direction, return the hermitian conjugate of the link
            #on the arrival point
            if delta[direction] == 1:
                U_pos = np.append(i_point%self.N_lat, direction)
                matrices.append(lattice[tuple(U_pos)])
            else:
                U_pos = np.append(f_point%self.N_lat, direction)
                matrices.append(lattice[tuple(U_pos)].T.conjugate())
        
        #compute the matrix product
        return np.linalg.multi_dot(matrices)
        
        
    def ten_hits_metropolis_enhanced(self, i, j, k, l, mu):
        """
        Metropolis algorithm applied 10 times on each site, before moving on to the next.
        Divided in 2 parts. The first part computes the matrix product between the 
        3 links needed to close the plaquette, together with the link specified by
        i, j, k, l and mu. Since there are two plaquettes on each side of the link for
        each plane, there are six plaquettes in total. The sum of these matrix products
        is stored in Gamma.
        The second part is the proper metropolis algorithm, using standard Wilson action.
        

        Parameters
        ----------
        i, j, k, l : int
            Integers specifying the lattice site
        mu : int
            Specifies the direction, thus the link

        Returns
        -------
        None.

        """
        Gamma = np.zeros([3,3], dtype = complex)
        hand = [1, -1]
        i_pos = np.array([i,j,k,l])         #starting point  
        
        #choose a nu different from mu, i.e. a plane spanned by mu and nu
        nu = (mu+1)%4                      
        
        #the cycle ends when nu = mu, i.e. the whole procedure is repeated for each plane
        while nu != mu:                                                                                       
            for orientation in hand:
                
                i_pos[mu] += 1                       #move along mu direction
                path = [copy.copy(i_pos)]                #save point
                i_pos[nu] += orientation             
                path.append(copy.copy(i_pos))
                i_pos[mu] += -1                      
                path.append(copy.copy(i_pos))
                i_pos[nu] += -orientation
                path.append(copy.copy(i_pos))
                
                Gamma += self.link(path)          
                
            #next plane
            nu = (nu+1)%4                        
        
        #SECOND PART: METROPOLIS ALGORYTHM
        #save value of the current link
        U_old = self.Lambda1[i,j,k,l,mu]
    
        for hit in range(10):
            
            #compute old action involving the current link
            S_old = - (self.beta/self.u0to4)*(np.real(np.trace(np.dot(U_old, Gamma)))/3)
            
            #extract a random matrix and update the old link
            M = self.random_matrices[self.rng.integers(0, self.N_rnd_matrices)]
            U_new = np.dot(M, U_old)
            
            #compute shift in action
            dS = - (self.beta/self.u0to4)*(np.real(np.trace(np.dot(U_new, Gamma)))/3) - S_old
            if dS<0 or (dS>0 and self.rng.random() < np.exp(-dS)):
                U_old = U_new
                self.acceptance += 1/(10*4*(self.N_lat**4)*5*self.N_cor)
        
        #store the result at the end of 10 hits
        self.Lambda1[i,j,k,l,mu] = U_old
        
        return                

        

    def thermalize(self):
        """
        Thermalization of the lattice, starting from a cold configuration.
                                                
        Returns
        -------
        None.

        """                 
        self.acceptance = 0.
        for therm_index in tqdm.tqdm(range(5*self.N_cor), desc = "Thermalizing the lattice"):

            #metropolis algorithm is called for each link
            for i in range(self.N_lat):
                for j in range(self.N_lat):
                    for k in range(self.N_lat):
                        for l in range(self.N_lat):
                            for mu in range(4):
                                self.ten_hits_metropolis_enhanced(i, j, k, l, mu)
        print("\n Acceptance ratio %g" % self.acceptance, flush = True)
        
        return
    
        
    def compute_random_paths(self):
        """
        Computes random configurations weighted according to exp(-S).
        self.N_cor sweeps of the lattice are discarded between one valid configuration
        and the next to minimize statistical correlations.
        Configurations are stored in self.arr_of_path for later use.

        Returns
        -------
        None.

        """
        self.arr_of_paths = []
        
        for alpha in tqdm.tqdm(range(self.N_cf), desc = "Computing random paths"):          
            for therm_index in range(self.N_cor):               
                for i in range(self.N_lat):
                    for j in range(self.N_lat):
                        for k in range(self.N_lat):
                            for l in range(self.N_lat):
                                for mu in range(4):
                                    self.ten_hits_metropolis_enhanced(i, j, k, l, mu)
                                                    
            self.arr_of_paths.append(copy.copy(self.Lambda1))
            
        return
    
    
    def average_over_paths(self, f, f_parameters = None):
        """
        Given a function f that accepts as input:
            1) a numpy.ndarray (that is, the lattice);
            2) a list of parameters 'f_parameters';
        returns the average of said f over all configurations.
        When calling this method on a function of the lattice which needs parameters as input,
        the function and the parameters must be inserted separately in the method call.

        Parameters
        ----------
        f : float, array, numpy.ndarray
            The function can be arbitrary, as long as it is a function of the lattice.
            
        f_parameters : list of parameters, optional
            Parameters needed for the function to be evaluated. The default is None.

        Returns
        -------
        avg : float, array, numpy.ndarray
            Average of the input function over configurations. Type equal to the type
            of the input function.

        """
        if f_parameters is None:
            avg = f(self.Lambda1)*0                         #this ensures avg is of the
                                                            #same type as f
            for alpha in range(self.N_cf):            
                avg += f(self.arr_of_paths[alpha])/self.N_cf
        else:
            #the function is called with the parameters needed
            avg = f(self.Lambda1, f_parameters)*0
            for alpha in range(self.N_cf):            
                avg += f(self.arr_of_paths[alpha], f_parameters)/self.N_cf
                
        return avg
    
    
    def evaluate_on_paths(self, f, f_parameters = None):
        """
        Same as average_over_paths, except it returns a list containing evaluations
        of f over all configurations, without averaging them.
        

        Parameters
        ----------
        f : float, array, numpy.ndarray
            The function can be arbitrary, as long as it is a function of the lattice.
            
        f_parameters : list of parameters, optional
            Parameters needed for the function to be evaluated. The default is None.

        Returns
        -------
        arr_of_evaluations : list of float, array, numpy.ndarray
            Evaluation of the input function over configurations. Each element of
            the list has the same type as the input function.

        """
        arr_of_evaluations = []
        
        if f_parameters is None:
            for alpha in range(self.N_cf):
                arr_of_evaluations.append(f(self.arr_of_paths[alpha]))
        else:
            for alpha in range(self.N_cf):
                arr_of_evaluations.append(f(self.arr_of_paths[alpha], f_parameters))
            
        return arr_of_evaluations
    
    
    def wilson_rectangle(self, configuration, shape = [1,1]):
        """
        Returns the average of an arbitrary rectangle operator over a given configuration.
        Its size is specified by the 'shape' parameter.

        Parameters
        ----------
        configuration : numpy.ndarray
            7-indices array: the lattice.
        shape : list of two integers, optional
            Specifies the shape of the rectangle. The default is [1,1], called Plaquette operator.

        Returns
        -------
        avg_rect : float
            Average of the rectangle operator.

        """
        base, height = shape[0], shape[1]
        avg_rect = 0
        for i in range(self.N_lat):          #for each link
            for j in range(self.N_lat):
                for k in range(self.N_lat):
                    for l in range(self.N_lat):
                        node = np.array([i,j,k,l])
                        
                        #if the shape required is a square, follows a reduced loop
                        #computing only half the wilson_loops
                        if base == height:
                            for mu in range(3):
                                for nu in range(mu+1, 4):
                                    loop = [copy.copy(node)]
                                    for n_points in range(base):
                                        node[mu] += 1
                                        loop.append(copy.copy(node))
                                    for n_points in range(height):
                                        node[nu] += 1
                                        loop.append(copy.copy(node))
                                    for n_points in range(base):
                                        node[mu] -= 1
                                        loop.append(copy.copy(node))
                                    for n_points in range(height):
                                        node[nu] -= 1
                                        loop.append(copy.copy(node))
                                    avg_rect += (np.real(np.trace(self.link(loop, lattice = configuration)))/3)/(6*(self.N_lat)**4)
                        else:
                            for mu in range(4):
                                for nu in range(4):
                                    if nu != mu:
                                        loop = [copy.copy(node)]
                                        for n_points in range(base):
                                            node[mu] += 1
                                            loop.append(copy.copy(node))
                                        for n_points in range(height):
                                            node[nu] += 1
                                            loop.append(copy.copy(node))
                                        for n_points in range(base):
                                            node[mu] -= 1
                                            loop.append(copy.copy(node))
                                        for n_points in range(height):
                                            node[nu] -= 1
                                            loop.append(copy.copy(node))
                                        avg_rect += 0.5*(np.real(np.trace(self.link(loop, lattice = configuration)))/3)/(6*(self.N_lat)**4)
        return avg_rect
    
    
    def tadpole_improve(self):
        """
        Tadpole improvement of the lattice action, by dividing each link
        by the mean value of the plaquette operator raised to the power of 1/4.
        The improvement is done by guessing an initial value for self.u0to4,
        computing random configurations with this value, evaluating the mean value
        of the plaquette operator and the adjusting the action accordingly.
        The adjustment is realized by an algorithm in which the new input is the
        geometric mean between the initial value and the evaluated one, plus an
        additional arithmetic mean taming the oscillations when the succession is
        close to convergence.
        This procedure converges rapidly to a fixed value.

        Returns
        -------
        None.

        """
        eps = 0.01
        print("Tadpole improvement started with initial value", self.u0to4, flush = True)
        
        #to speed up the procedure, the iterations are done on a smaller lattice,
        #called playground, which has the same parameters as the original lattice
        #except for the number of lattice points
        playground = LatticeQCD(N_lat = 4, N_config = 20, N_cor = self.N_cor,
                                 eps = self.eps, u0to4 = self.u0to4)
        playground.initialize()
        playground.thermalize()
        playground.compute_random_paths()
        
        #parameters to enable the arithmetic mean when the succession starts oscillating
        positive = None
        long_search = True
        
        #loop until convergence is found
        i = 0
        while True:
            u0_old = playground.u0to4
            u0_new = playground.average_over_paths(playground.wilson_rectangle)
            
            #find the positivity of the first increment in u0, stored as a boolean
            if positive is None:
                positive = (u0_new - u0_old > 0)
            
            #stop search if convergence is reached
            if np.abs(u0_new - u0_old) < eps:
                self.u0to4 = u0_old
                print("\n Convergence reached to value", u0_old, ", tadpole improvement completed \n", flush = True)                
                break
            
            #otherwise, evaluate the geometric mean and compute again random paths
            #with the updated value for u0
            else:
                i+=1
                if i == 10:
                    print("Convergence of tadpole improvement failed after 10 attempts, tadpole improvement will be disabled \n", flush = True)
                    self.u0to4 = 1
                    break
                
                print("\n Update number", i, flush = True)
                
                #if the succession starts oscillating, the arithmetic mean is switched on
                if positive != (u0_new - u0_old > 0):
                    long_search = False
                if long_search == True:
                    playground.u0to4 = np.sqrt(u0_old*u0_new)
                else:                    
                    playground.u0to4 = (np.sqrt(u0_old*u0_new) + u0_old)/2
                    
                print("Old value:", u0_old, ", New value:", playground.u0to4, flush=True)                
                playground.thermalize()
                playground.compute_random_paths()

        
    def bootstrap_plaquettes(self, N_bootstrap = 100):
        """
        Application of the method of bootstrap copies to evaluate the values of
        different Wilson loops with associated uncertainties.
        The "a by a" plaquette and "2a by a" rectangle operator are considered.
        The estimate for the mean is given by the average of the operator over all
        the configurations stored in self.arr_of_paths.
        The estimate for the uncertainty is obtained by creating N_bootstrap 
        bootstrap copies of self.arr_of_paths and evaluating the average of the
        operator over each copy, thus obtaining N_bootstrap values which are stored
        in ensemble_of_plaquettes/ensemble_of_rects.
        The distribution that these values follow approximates the distribution
        that would have been obtained by repeating the full Monte Carlo procedure
        N_bootstrap times. Thus, it can be used to estimate the Monte Carlo uncertainty
        on the original mean by simply evaluating the standard deviation.       
        
        Variables
        -------
        N_bootstrap : int
            Number of bootstrap copies of the original self.arr_of_paths
        arr_of_plaquettes, arr_of_rects : list of float
            The plaquette/rectangle operator is evaluated on all the configurations stored
            in self.arr_of_paths. The results are stored in this arr_of_plaquettes/rects.
        ensemble_of_plaquettes, ensemble_of_rects : list of float
            Each element of the list is the MC average of the plaquette/rectangle operator
            over the configurations of a bootstrap copy of self.arr_of_paths.

        Returns
        -------
        mean_plaquette, mean_rect : float
            Mean value of the plaquette/rectangle operator over self.arr_of_paths
        err_plaquette, err_rect : float
            Statistical uncertainty associated to the mean value of the plaquette/
            rectangle, computed as the standard deviation of the bootstrap values.

        """
        
        #evaluate plaquette and rectangle operator over all paths
        arr_of_plaquettes = self.evaluate_on_paths(self.wilson_rectangle)
        arr_of_rects = self.evaluate_on_paths(self.wilson_rectangle, [2,1])
        
        #average to obtain MC estimates for the mean plaquette and rectangle operator
        mean_plaquette, mean_rect = 0, 0
        for alpha in range(self.N_cf):
            mean_plaquette += arr_of_plaquettes[alpha]/self.N_cf
            mean_rect += arr_of_rects[alpha]/self.N_cf
            
        #initialize the lists of MC average values of plaquette/rectangle operator
        ensemble_of_plaquettes = []
        ensemble_of_rects = []
        
        #loop on bootstrap copies
        for i in range(N_bootstrap):
            bootstrap_plaquette = 0
            bootstrap_rect = 0
            
            #evaluate mean value of plaquette and rectangle on the given bootstrap copy
            for alpha in range(self.N_cf):
                choice = self.rng.integers(0, self.N_cf)
                bootstrap_plaquette += arr_of_plaquettes[choice]/self.N_cf
                bootstrap_rect += arr_of_rects[choice]/self.N_cf
            
            #store the results in the ensembles
            ensemble_of_plaquettes.append(copy.copy(bootstrap_plaquette))
            ensemble_of_rects.append(copy.copy(bootstrap_rect))
        
        #evaluate mean of op and op squared over ensemble
        mean_ensemble_plaquette, mean_ensemble_rect = 0, 0
        mean_ensemble_plaquette_sq, mean_ensemble_rect_sq = 0, 0        
        for i in range(N_bootstrap):
            mean_ensemble_plaquette += ensemble_of_plaquettes[i]/N_bootstrap
            mean_ensemble_plaquette_sq += (ensemble_of_plaquettes[i]**2)/N_bootstrap
            mean_ensemble_rect += ensemble_of_rects[i]/N_bootstrap
            mean_ensemble_rect_sq += (ensemble_of_rects[i]**2)/N_bootstrap
            
        #evaluate spread of the ensemble as standard deviation
        #each bootstrap MC average should differ from the original MC average by an amount
        #of order the montecarlo error          
        err_plaquette = np.sqrt(np.abs(mean_ensemble_plaquette_sq - mean_ensemble_plaquette**2))
        err_rect = np.sqrt(np.abs(mean_ensemble_rect_sq - mean_ensemble_rect**2))
        
        print("\nWilson plaquette is", mean_plaquette, "+-", err_plaquette)
        print("Wilson rect is", mean_rect, "+-", err_rect)
        
        return mean_plaquette, err_plaquette, mean_rect, err_rect
    
        

test = LatticeQCD()
test.tadpole_improve()
test.initialize()
test.thermalize()
test.compute_random_paths()
wilson_plaquette, err_plaquette, wilson_rect, err_rect = test.bootstrap_plaquettes()

#wilson_square = test.average_over_paths(test.wilson_rectangle, [2,2])

