# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:11:03 2021

@author: Hunter Faulkner
"""

import numpy as np
from European_Option import European_Option
from scipy.stats import norm

class European_Hedge(European_Option):
    '''
    Monte Carlo Price and Dynamic Delta 
    Hedge of European Call/Put Options
    -----------------------------------
    S0      - Initial Underlying Price
    K       - Strike Price
    g       - Risk-Free Rate
    r       - Interest Rate
    sgm     - Volatility
    T       - Time until Maturity
    callput - Option Type
    '''
    def __init__(self,S0,K,g,r,T,callput,sgm=1000,price=None):
        self.S0=S0;
        self.K=K;
        self.g=g
        self.r=r;
        self.sgm=sgm;
        self.T=T;
        self.callput=callput;
        self.price=price
        # For Notational Convenience:
        self.g_disc=np.e**(-g*T);
        self.r_disc=np.e**(-r*T);
        # Inheritances
        super().optd1()
        super().optd2()
        super().pdf_norm()
        super().Price()
        # The following is called to calculate implied volatility
        if price != None:
            super().ImpVol()
            self.sgm=self.ImpVol();
        self.d1=self.optd1()
        self.d2=self.optd2()
        self.pdfnorm=self.pdf_norm()
        self.price=self.Price()
    
    def PriceMC(self,n,CI=False):
        W=np.random.standard_normal(n)
        S=self.S0*np.e**((self.r-self.g-.5*self.sgm**2)*self.T+
                    self.sgm*np.sqrt(self.T)*W)
        po=self.callput*(S-self.K)
        po=po*(po>0)
        mcPrice=self.r_disc*np.mean(po)
        mcstd=self.r_disc*np.std(po)/np.sqrt(n)
        if CI==True:
            upperCI=mcPrice+2*mcstd
            lowerCI=mcPrice-2*mcstd
            print('\n        95% Confidence Interval\n',
                  '       -----------------------\n',
                  f'{lowerCI}  -  {upperCI}\n')
        return mcPrice,mcstd
    
    def BSDeltaHedgeMC(self,mu,n,m,CI=False):
        # Simulate Stocks
        w=np.random.standard_normal((n,m))
        dt=self.T/m
        ST=(self.S0*np.exp(((mu-0.5*self.sgm**2)*dt
                            +self.sgm*np.sqrt(dt)*w).cumsum(axis=1)))
        
        # Current time period
        dtvec=np.linspace(0,self.T,m+1)[1:m]
        sq_dt=np.array([np.sqrt(self.T-dt) for dt in dtvec])
        d1=((np.log(ST[:,1:]/self.K)+(self.r+.5*self.sgm**2)*(self.T-dtvec))
             /(self.sgm*sq_dt))
        d2=d1-self.sgm*dtvec
        
        # Previous time period
        dtpvec=np.linspace(0,self.T,m+1)[:m-1]
        sq_dtp=np.array([np.sqrt(self.T-dt) for dt in dtpvec])
        d1p=((np.log(ST[:,:m-1]/self.K)+(self.r+.5*self.sgm**2)*(self.T-dtpvec))
             /(self.sgm*sq_dtp))
        d2p = d1p-self.sgm*dtpvec
    
        # Profit and Loss
        PnL=((ST[:,:m-1]*(norm.cdf(d1p)-norm.cdf(d1))
            -self.K*np.e**(-self.r*(self.T-(dtvec)))
            *(norm.cdf(d2p)-norm.cdf(d2)))*np.e**(self.r*(self.T-dtvec)))
        
        # Final Period's Profit and Loss
        PnLfinal = (ST[:,-1]*(norm.cdf(d1p[:,-1]))-self.K*(norm.cdf(d2p[:,-1]))
                    -((ST[:,-1]-self.K)*((ST[:,-1]-self.K)>0)))
        
        # Vectors and sum for 1D vector of PnLs
        PnLtot=(PnL.sum(axis=1)+(PnLfinal))
        
        # Outputs
        MCPnL=PnLtot.mean()
        MCSTD=PnLtot.std()/np.sqrt(n)
        if CI==True:
            lowerCI=MCPnL-2*MCSTD
            upperCI=MCPnL+2*MCSTD
            print('\n        95% Confidence Interval\n',
                  '       -----------------------\n',
                  f'{lowerCI}  -  {upperCI}\n')
        
        return MCPnL, MCSTD       
