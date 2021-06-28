import numpy as np
from math import factorial
from scipy.stats import norm
from European_Option import European_Option

class Jumps(European_Option):
    def __init__(self,S0,K,g,r,sgm,T,k,lam,delta,eps,callput):
        self.S0 = S0;
        self.K = K;
        self.g = g;
        self.r = r;
        self.sgm = sgm;
        self.T = T;
        self.k = k;
        self.lam = lam;
        self.delta = delta;
        self.eps = eps;
        self.callput = callput;
        self.r_disc = np.e**(-self.r*self.T);
        super().optd1();
        super().optd2();
        self.d1 = self.optd1();
        self.d2 = self.optd2();

    
    def jump_price(self):

        def bsprice(self,v2n,rn):
            d1 = (np.log(self.S0 / self.K) + (rn + (.5 * v2n)) * self.T) / (np.sqrt(self.T) * np.sqrt(v2n))
            d2 = d1 - (np.sqrt(self.T) * np.sqrt(v2n))
            if self.callput == 1:
                price=(self.S0*norm.cdf(d1))-(self.K*np.e**(-rn*self.T)*norm.cdf(d2))
            elif self.callput == -1:
                price=self.K*np.e**(-rn*self.T)*norm.cdf(-d2)-self.S0*norm.cdf(-d1)
            else:
                raise ValueError("")
            return price;

        lambdaprime=self.lam*(1+self.k)
        gamma=np.log(1+self.k)
        n=0
        i=0
        price=0
        P=(np.e**(-lambdaprime*self.T)*((lambdaprime*self.T)**n))/(factorial(n))
        rn=self.r-self.lam*self.k+((n*gamma)/self.T)
        v2n=self.sgm**2+((n*gamma)/self.T)
        BSprice=bsprice(self,v2n,rn)
        i+=(BSprice*P)
        while i > self.eps:
            n+=1
            price+=i
            P=(np.e**(-lambdaprime*self.T)*(lambdaprime*self.T)**n)/(factorial(n))
            rn=self.r-(self.lam*self.k)+((n*gamma)/self.T)
            v2n=self.sgm**2+((n*(self.delta**2))/self.T)
            BSprice=bsprice(self,v2n,rn)
            i=(BSprice*P)
        return price;