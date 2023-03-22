import numpy as np

import matplotlib.pyplot as plt 

 
plt.rcParams['figure.dpi'] = 300

"""
ind: 
    0 : id int 
    1 : sex
    2 : generation
    3 : mother int
    4 : father int
    5 : ancestry coefficient [0,1]
c = 100 means that anc = 1 indivudal is 100 times as likely to mate with anc=1
as with anc = 0. rn linear.
 
"""


class Population:
    
    arr_factor = 1.02
    
    def __init__(self, K, G):
        self.K = K
        self.G = G
        self.c = 500
        self.E_offspring = 2
        self.r = 1
        self.rows = 6
        
    def run(self):
        """
        execute simulation
        Returns
        -------
        None.
        """
        gen_arr1 = self.initialize()
        self.plot(gen_arr1)
        while self.g > 0:
            self.g -= 1
            gen_arr0 = gen_arr1
            gen_arr1 = self.gen_cycle(gen_arr0)
            self.pop_arr[self.n0:self.n1] = gen_arr1
            self.plot(gen_arr1)
        self.pop_arr = self.pop_arr[:self.n1]
        
    def initialize(self):
        """
        make a pedigree arr for the whole population and one for the founding
        generation
        """
        self.n0 = 0
        self.n1 = self.K
        self.N = self.K
        self.g = self.G
        self.E_length = int(self.K * (self.g + 1) * 1.02)
        self.pop_arr = np.zeros((self.E_length, 6), dtype = np.float32)
        gen_arr = np.zeros((self.K, 6), dtype = np.float32)
        gen_arr[:, 0] = np.arange(self.K)
        gen_arr[:, 1] = np.random.choice([0,1], size = self.K)
        gen_arr[:, 2] = self.G
        gen_arr[:, 3] = -1
        gen_arr[:, 4] = -1
        split = self.K // 2
        gen_arr[:split, 5] = 0
        gen_arr[split:, 5] = 1
        self.pop_arr[self.n0:self.n1] = gen_arr 
        return(gen_arr)
    
    def gen_cycle(self, gen_arr0):
        
        mating_arr = self.mate(gen_arr0)
        gen_arr1 = self.make_gen_arr(gen_arr0, mating_arr)
        return(gen_arr1)
        
    def mate(self, gen_arr0):
        """
        E_offspring_vec = 2 * np.exp(params.r * (1 - (N_vec / Kfac_vec)))
        """
        f_arr = self.get_sex(gen_arr0, 0)
        m_arr = self.get_sex(gen_arr0, 1)
        f_ancs = f_arr[:, 5]
        m_ancs = m_arr[:, 5]
        F = len(f_arr)
        E_offspring = self.E_offspring * np.exp((self.r * (1 - (self.N / self.K))))
        n_offspring = np.random.poisson(E_offspring, F)
        n_matings = int(np.sum(n_offspring, dtype = np.int64) * 1.1)
        mating_arr = np.zeros((n_matings, 2))
        
        l0 = 0
        for i in np.arange(F):
            if n_offspring[i] > 0:
                cd = self.get_mating_cd(m_ancs, f_ancs[i])
                X = np.random.uniform()
                select = np.searchsorted(cd, X) - 1
                l1 = l0 + n_offspring[i]
                mating_arr[l0:l1] = [i, select]
                l0 = l1
        mating_arr = mating_arr[:l0]
        return(mating_arr)
    
    def make_gen_arr(self, gen_arr0, mating_arr):
        
        self.N = len(mating_arr)
        self.n0 = self.n1
        self.n1 += self.N
        f_ids = mating_arr[:, 0].astype(np.int32)
        m_ids = mating_arr[:, 1].astype(np.int32)
        f_arr = self.get_sex(gen_arr0, 0)
        m_arr = self.get_sex(gen_arr0, 1)
        
        gen_arr1 = make_arr(self.N)
        gen_arr1[:, 0] = np.arange(self.n0, self.n1)
        gen_arr1[:, 1] = np.random.choice([0,1], size = self.N)
        gen_arr1[:, 2] = self.g
        gen_arr1[:, 3] = f_arr[f_ids, 0]
        gen_arr1[:, 4] = m_arr[m_ids, 0]
        ancs = np.vstack((f_arr[f_ids, 5], m_arr[m_ids, 5]))
        gen_arr1[:, 5] = np.mean(ancs, 0)
        return(gen_arr1)
 
    def make_arr(self, length):
        """Get a 2d array of dimension (length, self.rows)
        """
        arr = np.zeros((length, self.rows), dtype = np.float32)
        return(arr)
    
    def get_sex(self, arr, sex):
        """Get a subpopulation array of the only females or males
        """
        sub_arr = arr[arr[:, 1] == sex]
        return(sub_arr)
        
    def get_mating_cd(self, m_ancs, f_anc):
        #linear
        distance = np.abs(m_ancs - f_anc)
        cminus1 = self.c - 1
        pref = self.c - cminus1 * distance
        pref /= np.sum(pref)
        cd = np.cumsum(pref)
        return(cd)
    
    def plot(self, gen_arr):
        
        ancs = gen_arr[:,5]
        bins = np.linspace(0, 1, 101)
        fig = plt.figure(figsize=(6, 6))
        sub = fig.add_subplot(111)
        sub.hist(ancs, bins = bins, color = 'blue')
        sub.set_ylim(0, self.K)
        sub.set_xlim(0, 1)
        sub.set_title(str(gen_arr[0,2]))
        

def mate(gen_arr0):
    pop = Population(40,10)
    pop.N = 40
    
    f_arr = pop.get_sex(gen_arr0, 0)
    m_arr = pop.get_sex(gen_arr0, 1)
    f_ancs = f_arr[:, 5]
    m_ancs = m_arr[:, 5]
    F = len(f_arr)
    E_offspring = pop.E_offspring * np.exp((pop.r * (1 - (pop.N / pop.K))))
    n_offspring = np.random.poisson(E_offspring, F)
    n_matings = int(np.sum(n_offspring, dtype = np.int64) * 1.1)
    mating_arr = np.zeros((n_matings, 2))
    
    l0 = 0
    for i in np.arange(F):
        if n_offspring[i] > 0:
            cd = pop.get_mating_cd(m_ancs, f_ancs[i])
            X = np.random.uniform()
            select = np.searchsorted(cd, X)
            l1 = l0 + n_offspring[i]
            mating_arr[l0:l1] = [i, select]
            l0 = l1
    mating_arr = mating_arr[:l0]
    return(mating_arr)
        
def make_gen_arr(gen_arr0, mating_arr):
    pop = Population(40,10)
    pop.N = 40
    pop.n0 = 40
    pop.n1 = 40
    pop.g = 0
    
    pop.N = len(mating_arr)
    pop.n1 += pop.N
    f_ids = mating_arr[:, 0].astype(np.int32)
    m_ids = mating_arr[:, 1].astype(np.int32)
    f_arr = pop.get_sex(gen_arr0, 0)
    m_arr = pop.get_sex(gen_arr0, 1)
    
    gen_arr1 = make_arr(pop.N)
    gen_arr1[:, 0] = np.arange(pop.n0, pop.n1)
    gen_arr1[:, 1] = np.random.choice([0,1], size = pop.N)
    gen_arr1[:, 2] = pop.g
    gen_arr1[:, 3] = f_arr[f_ids, 0]
    gen_arr1[:, 4] = m_arr[m_ids, 0]
    ancs = np.vstack((f_arr[f_ids, 5], m_arr[m_ids, 5]))
    gen_arr1[:, 5] = np.mean(ancs, 0)
    
    pop.n0 += pop.N
    
    return(gen_arr1)
        
        
def make_arr(length):
    pop = Population(10,10)
    
    arr = np.zeros((length, pop.rows), dtype = np.float32)
    
    return(arr)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        