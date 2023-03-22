import numpy as np

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
        self.c = 4
        self.E_offspring = 2
        self.r = 1
        
    def run(self):
        """
        execute simulation

        Returns
        -------
        None.

        """
        self.compute_vars()
        gen_arr = self.initialize()
        while self.g > 0:
            gen_arr0 = gen_arr
            gen_arr = self.gen_cycle(gen_arr0)
        
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
        """
        
        """
        
        
    def mate(self, gen_arr0):
        """
        E_offspring_vec = 2 * np.exp(params.r * (1 - (N_vec / Kfac_vec)))
        """
        f_arr = self.get_sex(gen_arr0, 0)
        m_arr = self.get_sex(gen_arr0, 1)
        f_anc = f_arr[:, 5]
        m_anc = m_arr[:, 5]
        F = len(f_arr)
        E_offspring = self.E_offspring * (self.r * (1 - (self.N / self.K)))
        n_offspring = np.random.poisson(E_offspring, F)
        n_matings = np.sum(n_offspring, dtype = np.int64)
        mating_arr = np.zeros((n_matings, 2))
        for i in np.arange(F):
            if n_matings[i] > 0:
                anc = f_anc[i]
                distance = np.abs(m_anc - anc)
                reduction = 1 - distance
                scaled = self.c * reduction
                scaled /= np.sum(scaled)
                X = np.random.uniform()
                cd = np.cumsum(scaled)
                select = np.searchsorted(cd, X)
                mating_arr[i] = [i, select]
        return(mating_arr)
        
    
    def get_sex(arr, sex):
        """
        get a subpopulation array of the only females or males
        """
        sub_arr = arr[arr[:, sex] == 0]
        return(sub_arr)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        