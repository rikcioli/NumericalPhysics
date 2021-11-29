# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:33:19 2021

@author: rikci
"""

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import copy



class MyClass:
    """Path integral Harmonic Oscillator"""

    def __init__(self, lat_spacing = 0.5, N_lat = 20, N_config = 1000, N_cor = 50, eps = 1.4):
        self.a = lat_spacing                      #lattice spacing
        self.N_lat = N_lat              #number of lattice points
        self.N_cf = N_config            #number of different paths/configurations
        self.N_cor = N_cor         #number of loops for thermalization
        self.eps = eps                  #boundary of the interval for the variation in position each step
        self.x = np.zeros(N_lat)        #array of positions
        self.arr_of_paths = []          #array (actually list) of paths
        self.t = np.linspace(0, N_lat*lat_spacing, N_lat, endpoint=False)
        self.G = np.zeros(N_lat)        #2 point function, as a function of the time interval na (just n)
        self.G_MC_error = np.zeros(N_lat)    #statistical error on the 2 point function

    def S(self, j, x):
        jp2 = (j+2)%self.N_lat
        jp = (j+1)%self.N_lat # next site
        jm = (j-1)%self.N_lat # previous site
        jm2 = (j-2)%self.N_lat
        return self.a*x[j]**2/2 + (x[jp] - x[j])**2/(2*self.a)
    

    def thermalize(self):                   #thermalization of the lattice, starting
                                            #from a cold configuration
        for i in range(10*self.N_cor):
            for j in range(self.N_lat):     #sweep cycle
                x_old = self.x[j]
                S_old = self.S(j, self.x)
                dx = rnd.uniform(-self.eps,self.eps)
                self.x[j] += dx
                dS = self.S(j, self.x) - S_old
                if dS > 0 and rnd.uniform(0,1) > np.exp(-dS):
                    self.x[j] = x_old




    def compute_random_paths(self):          
        acceptance = 0.
        for alpha in range(self.N_cf):                  #cycle on the N_cf configurations
            for i in range(self.N_cor):         #thermalization cycle
            
                for j in range(self.N_lat):     #sweep cycle for the metropolis algorithm
                    x_old = self.x[j]
                    S_old = self.S(j, self.x)
                    dx = rnd.uniform(-self.eps,self.eps)      #generate random variation
                    self.x[j] += dx
                    dS = self.S(j, self.x) - S_old            #difference in action corresponding to that variation
                    if dS > 0 and rnd.uniform(0,1) > np.exp(-dS):       #metropolis, accept if negative
                        self.x[j] = x_old
                    else:
                        acceptance += 1.0/(self.N_cf*self.N_cor*self.N_lat)
                        
            self.arr_of_paths.append(copy.deepcopy(self.x))          #save the average at the end of the bin
            if alpha%(self.N_cf/20) == 0:
                print("Working %d of %g" % (alpha, self.N_cf))
        print("Acceptance ratio %g" % acceptance)
        
        
    def show_paths(self):
        for path in self.arr_of_paths:
            plt.plot(path, self.t) 
        plt.xlabel("x(t)")
        plt.ylabel("t")
        plt.show()
    
    def execute(self):
        self.thermalize()
        self.compute_random_paths()
        self.show_paths()
        
    def average_over_paths(self, f):
        """
        Given a function of the positions x, returns the average of the function
        over all configurations

        Parameters
        ----------
        f : np.float64 or np.ndarray
            Must be a function that only has an array of positions x(t) as input.
            Can be either one or multi-dimensional.

        Raises
        ------
        ImportError
            DESCRIPTION.

        Returns
        -------
        avg : np.float64 or np.ndarray
            Average of f over configurations.

        """
        avg = copy.deepcopy(f(self.x))
        if type(avg) == np.float64:
            avg = 0
        elif type(avg) == np.ndarray:
            for count, item in enumerate(avg):
                avg[count] = 0 
        else:
            raise ImportError(
                """Please use a valid function
                """       
                )
        for alpha in range(self.N_cf):            
            avg += f(self.arr_of_paths[alpha])/self.N_cf
        return avg
    
    def evaluate_on_paths(self, f):
        """
        Given a function of the positions x, evaluates the function
        over all configurations. Returns the values in an array of evaluations.

        Parameters
        ----------
        f : numpy.float64 or numpy.ndarray
            Must be a function that only has an array of positions x(t) as input.
            Can be either one or multi-dimensional.

        Returns
        -------
        arr_of_evaluations : TYPE
            DESCRIPTION.

        """
        arr_of_evaluations = []
        for alpha in range(self.N_cf):
            arr_of_evaluations.append(f(self.arr_of_paths[alpha]))
        return arr_of_evaluations
    
    def evaluate_green_functions(self):
        """
        Returns the average of the 2-point function over all configurations,
        stored in a n-dim array.
        Plots the green function against time.
        """
        self.G = self.average_over_paths(self.two_point_function)        
        plt.plot(self.t, self.G, linestyle = 'none', color = "red", marker = 'o')
        plt.show()
        
    def delta_energy_vs_t(self):
        """
        Evaluates energy difference between ground state and first excited state,
        as a function of time.
        Each energy difference is computed starting from a bootstrap copy of the 
        ensemble of green functions. 
        With 50-100 bootstrap copies, this method evaluates the mean and the 
        statistical uncertainty of the collection of 50-100 energy differences.
        It then plots the results. A straight line corresponding to the theoretical
        asymptotical value is shown.
        
        -------
        arr_of_Gn : list of numpy.float64 or numpy.ndarray
            array containing the 2-point function values over all configurations.
        deltaE : numpy.ndarray
            array containing the energy difference over time.
        ensemble_of_energies : list of numpy.ndarray
            list of the various deltaE, each evaluated on a different bootstrap copy.
            

        """
        arr_of_Gn = self.evaluate_on_paths(self.two_point_function)     #collection of Gn
        deltaE = np.zeros(self.N_lat)
        ensemble_of_energies = []
        N_bootstrap_copies = 100
        for i in range(N_bootstrap_copies):           #loop on all bootstrap copies
            avg_bootstrap_G = np.zeros(self.N_lat)    #prepare to evaluate G(t) for the current copy
            for alpha in range(self.N_cf):            #create a bootstrap copy of the ensemble
                choice = int(rnd.uniform(0, self.N_cf))
                avg_bootstrap_G += arr_of_Gn[choice]/self.N_cf     
            for n in range(self.N_lat):             #evaluate deltaE(t) for the current copy
                Gn = avg_bootstrap_G[n]
                Gn1 = avg_bootstrap_G[(n+1)%self.N_lat]
                deltaE[n] = np.log(np.abs(Gn/Gn1))/self.a
            ensemble_of_energies.append(copy.deepcopy(deltaE))    #append the current deltaE(t)
        
        mean_deltaE = np.zeros(self.N_lat)
        for i in range(N_bootstrap_copies):
            mean_deltaE += ensemble_of_energies[i]/N_bootstrap_copies
        err_deltaE = np.zeros(self.N_lat)
        for i in range(N_bootstrap_copies):
            err_deltaE += (ensemble_of_energies[i]**2 - mean_deltaE**2)/N_bootstrap_copies
        err_deltaE = np.sqrt(err_deltaE)
        
        plt.errorbar(self.t, mean_deltaE, yerr = err_deltaE, linestyle = 'none', color = "green", marker = 'o')
        plt.plot(self.t, np.ones(self.N_lat))
        plt.xlabel("t")
        plt.ylabel("DeltaE(t)")
        plt.grid(True)       
        plt.xlim([-0.25,self.N_lat*self.a*0.375])
        plt.show()

        
    #List of useful class functions to average
    
    def two_point_function(self, x):
        vector_func = np.zeros(self.N_lat)
        for n in range(self.N_lat):
            for j in range(self.N_lat):
                vector_func[n] += x[(j+n)%self.N_lat]*x[j]/self.N_lat
        return vector_func
    
    def two_point_function_squared(self, x):
        vector_func = np.zeros(self.N_lat)
        for n in range(self.N_lat):
            for j in range(self.N_lat):
                vector_func[n] += x[(j+n)%self.N_lat]*x[j]/self.N_lat
        return vector_func**2      


x = np.zeros(20)
x[7] = 2
def f(x):
    x = x+1
    return x*(3+x[19])

def x_7(x):
    return x[7]

def x_vector(x):
    return x

HO = MyClass()
HO.thermalize()
HO.compute_random_paths()
#HO.evaluate_green_functions()
HO.delta_energy_vs_t()



