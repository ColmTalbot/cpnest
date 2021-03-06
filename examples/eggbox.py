import unittest
import numpy as np
import cpnest.model

class EggboxModel(cpnest.model.Model):
    """
    Eggbox problem from https://arxiv.org/pdf/0809.3437v1.pdf
    """
    names=['1','2','3','4','5']
    bounds=[[0,10.0*np.pi],[0,10.0*np.pi],[0,10.0*np.pi],[0,10.0*np.pi],[0,10.0*np.pi]]
    data = None
    @staticmethod
    def log_likelihood(x):
        return log_eggbox(x)

    def force(self,x):
        f = np.zeros(1, dtype = {'names':x.names, 'formats':['f8' for _ in x.names]})
        return f

def log_eggbox(p):
    tmp = 1.0
    for n in p.names:
        tmp *= np.cos(p[n]/2.)
    return (tmp+2.0)**5.0

class EggboxTestCase(unittest.TestCase):
    """
    Test the eggox model
    """
    def setUp(self):
        self.work=cpnest.CPNest(EggboxModel(),verbose=1,nthreads=1,nlive=1000,maxmcmc=1000)

    def test_run(self):
        self.work.run()

def test_all():
    unittest.main(verbosity=2)

if __name__=='__main__':
        work=cpnest.CPNest(EggboxModel(),verbose=3,nthreads=1,nlive=1000,maxmcmc=1000,poolsize=1000)
        work.run()

