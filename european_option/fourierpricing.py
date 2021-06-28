import numpy as np
from scipy.integrate import quad

class FourierPrice():
    def __init__(self,S0,K,g,r,sgm,T):
        self.S0 = S0;
        self.K = K;
        self.g = g;
        self.r = r;
        self.sgm = sgm;
        self.T = T;
        self.r_disc = np.e**(-self.r*self.T);

    def phi(omega,mubar,sgmbar):
        phi=np.e**(1j*omega*mubar-.5*(omega**2)*sgmbar)
        return phi

    def fourierprice(self):

        def Phi(omega,mubar,sgmbar):
            phi=np.e**(1j*omega*mubar-.5*(omega**2)*sgmbar)
            return phi
        def P1(omega,K):
            p1=np.real((np.e**(-1j*omega*np.log(K))*Phi(omega-1j,mubar,sgmbar))/(1j*omega))
            return p1
        def P2(omega,K):
            p2=np.real((np.e**(-1j*omega*np.log(K))*Phi(omega,mubar,sgmbar))/(1j*omega))
            return p2

        mubar=np.log(self.S0)+self.r*self.T-.5*(self.sgm**2)*self.T
        sgmbar=(self.sgm**2)*self.T
        integral1=quad(P1,0,np.inf,args=(self.K))[0]
        integral2=quad(P2,0,np.inf,args=(self.K))[0]
        pi1=.5+(1/(np.pi*np.e**(self.r*self.T)*self.S0))*integral1
        pi2=.5+(1/np.pi)*integral2
        price=self.S0*pi1-self.K*self.r_disc*pi2
    
        return price
