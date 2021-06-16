# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:03:26 2021

@author: Hunter Faulkner
"""

import numpy as np
from scipy.stats import norm
from scipy import optimize

class European_Option:
    '''
    Description:
    -----------------------------------------
    Pricing and Greeks of Vanilla
    European Call/Put Options
    
    Note: Input price for implied volatility
    and tune sigma to the upper bracket of
    the root search algorithm. If giving a
    sigma, leave price undefined.
    
    Attributes:
    -----------------------------------------
    S0 : float
        Initial Underlying Price
        
    K : float
        Strike Price
        
    g : float
        Dividend Yield
        
    r : float
        Interest Rate
        
    sgm : float
        Volatility (Given or Implied by Price)
        
    T : float
        Time to Maturity
        
    callput : 1 for Call | -1 for Put
        Option Type
    
    Methods (Returns):
    -----------------------------------------
    Implied Volatility
    optd1
    optd2
    Price
    
    Traditional Greeks
    ------------------
    Delta   - Price Sensitivity to Underlying
    Gamma   - Sensitivity of Delta to Underlying
    Lambda  - Percentage Price Sensitivity to Percent Changes in Underlying
    Theta   - Price Sensitivity to Time until Maturity
    Vega    - Price Sensitivity to Volatility
    Rho     - Price Sensitivity to Interest Rate
    Epsilon - Price Sensitivity to Dividend Yield
    
    Higher Order Greeks
    -------------------
    Charm - Sensitivity of Delta to Time until Maturity
    Vanna - Sensitivity of Delta to Volatility
    Veta  - Sensitivity of Vega to Time until Maturity
    Vomma - Sensitivity of Vega to Volatility
    Vera  - Sensitivity of Rho to Volatility
    Speed - Sensitivity of Gamma to Underlying
    Color - Sensitivity of Gamma to Time until Maturity
    Zomma - Sensitivity of Gamma to Volatility
    Utima - Sensitivity of Vomma to Volatility
    '''
    def __init__(self,S0,K,g,r,T,callput,sgm=1000,price=None):
        self.S0=S0;
        self.K=K;
        self.g=g;
        self.r=r;
        self.sgm=sgm;
        self.T=T;
        self.callput=callput;
        self.price=price;
        # For Notational Convenience:
        self.g_disc=np.e**(-self.g*self.T);
        self.r_disc=np.e**(-self.r*self.T);
        self.d1=((np.log(self.S0/self.K)+(self.r+.5*self.sgm**2)*(self.T))/
                 (self.sgm*np.sqrt(self.T)));
        self.d2=self.d1-self.sgm*np.sqrt(self.T);
        self.pdfnorm=(1/np.sqrt(2*np.pi))*np.e**((-(self.d1)**2)/2);
        # The following is called to calculate implied volatility
        if price != None:
            self.sgm=self.ImpVol();
            self.d1=((np.log(self.S0/self.K)+(self.r+.5*self.sgm**2)*
                      (self.T))/(self.sgm*np.sqrt(self.T)));
            self.d2=self.d1-self.sgm*np.sqrt(self.T);
            self.pdfnorm=(1/np.sqrt(2*np.pi))*np.e**((-(self.d1)**2)/2);

    def ImpVol(self):
        
        def BS_Zero(sgm,self):
            self.d1=((np.log(self.S0/self.K)+(self.r+.5*sgm**2)*(self.T))/
                 (sgm*np.sqrt(self.T)));
            self.d2=self.d1-sgm*np.sqrt(self.T);
            if self.callput==1:
                Zero_P=((self.S0*norm.cdf(self.d1)
                     -self.K*self.r_disc*norm.cdf(self.d2))-self.price)
            elif self.callput==-1:
                Zero_P=((self.K*self.r_disc*norm.cdf(-self.d2)
                      -self.S0*norm.cdf(-self.d1))-self.price)
            else:
                raise ValueError('Invalid argument for callput. \
                                 Try -1 for put or 1 for call.')
            return Zero_P;

        self.IV=optimize.root_scalar(
                    BS_Zero,bracket=[-1000,1000], 
                    args=(self)
                    ).root
        return self.IV
   
    def optd1(self):
        return self.d1;
        
    def optd2(self):
        return self.d2;
        
    def Price(self):        
        if self.callput==1:
            BSP=(self.S0*norm.cdf(self.d1)
                   -self.K*self.r_disc*norm.cdf(self.d2))
        elif self.callput==-1:
            BSP=(self.K*self.r_disc*norm.cdf(-self.d2)
                 -self.S0*norm.cdf(-self.d1))
        else:
            raise ValueError('Invalid argument for callput. \
                             Try -1 for put or 1 for call.')
        return BSP;
    '''
    Traditional Option Greeks
    -------------------------
    '''
    def Delta(self):
        if self.callput==1:
            delta=self.g_disc*norm.cdf(self.d1)
        elif self.callput==-1:
            delta=self.g_disc*(-norm.cdf(-self.d1))
        else:
            raise ValueError('Invalid argument for callput. \
                             Try -1 for put or 1 for call.')
        return delta;
    
    def Gamma(self):
        gamma=((self.g_disc/(self.S0*self.sgm*np.sqrt(self.T)))*self.pdfnorm)
        return gamma;
    
    def Lambda(self):
        lam=(1/100)*((self.S0*self.g_disc*norm.cdf(self.d1))/
               (self.S0*self.g_disc*norm.cdf(self.d1)-
                self.K*self.r_disc*norm.cdf(self.d2)))
        return lam;
        
    def Theta(self):
        term1=(self.S0*self.sgm*(self.pdfnorm))
        if self.callput==1:
            term2=self.r*self.K*self.r_disc*norm.cdf(self.d2)
            theta=(1/100)*(term1+term2)
        elif self.callput==-1:
            term2=-self.r*self.K*self.r_disc*norm.cdf(-self.d2)
            theta=(1/100)*(term1+term2)   
        else:
            raise ValueError('Invalid argument for callput. \
                             Try -1 for put or 1 for call.')
        return theta;
    
    def Vega(self):
        vega=(1/100)*(self.S0*np.sqrt(self.T)*self.g_disc*(self.pdfnorm))
        return vega;
    
    def Rho(self):
        if self.callput == 1:
            rho=(1/100)*self.K*self.T*self.r_disc*norm.cdf(self.d2)
        elif self.callput == -1:
            rho=(-(1/100)*self.K*self.T*self.r_disc*norm.cdf(-self.d2))
        else:
            raise ValueError('Invalid argument for callput. \
                             Try -1 for put or 1 for call.')
        return rho;
   
    def Epsilon(self):
        if self.callput==1:
            epsilon=((1/100)*(-self.S0*(self.T)*self.g_disc)*norm.cdf(self.d1))
        elif self.callput==-1:
            epsilon=((1/100)*(self.S0*(self.T)*self.g_disc)*norm.cdf(-self.d1))
        else:
            raise ValueError('Invalid argument for callput. \
                             Try -1 for put or 1 for call.')
        return epsilon;
    '''
    Higher Order Greeks
    -------------------
    '''
    def Charm(self):
        charm=(((2*self.r*self.T-self.d2*self.sgm*np.sqrt(self.T))/
               (2*self.sgm*self.T*np.sqrt(self.T)))*self.pdfnorm)
        return charm;
    
    def Vanna(self):
        vanna=((-self.g_disc*self.d2)/self.sgm)*(self.pdfnorm)
        return vanna;
    
    def Veta(self):
        veta=(-self.S0*self.g_disc*np.sqrt(self.T)*norm.cdf(self.d1)*
             (self.g+(((self.r-self.g)*self.d1)/(np.sqrt(self.sgm**2*self.T)))
              -((1+(self.d1*self.d2))/(2*self.T)))*(1/100))
        return veta;
    
    def Vomma(self):
        vomma=((1/100)*((self.S0*self.g_disc*np.sqrt(self.T)*self.d1*self.d2)/
               (self.sgm))*self.pdfnorm)
        return vomma;
    
    def Speed(self):
        gamma=((self.g_disc/(self.sgm*self.S0*np.sqrt(self.T)))*self.pdfnorm)
        speed=(-((self.d1+np.sqrt(self.sgm**2*self.T))/(self.S0))*gamma)
        #(-((self.d1+self.sgm*np.sqrt(self.T))/((self.S0**2)*
         #       self.sgm*np.sqrt(self.T)))*self.g_disc*self.pdfnorm)
        return speed;
    
    def Color(self):
        color=(-((self.sgm+(np.log(self.S0/self.K)+(self.r+((self.sgm**2)/2))*
                self.T)*self.d1)/(2*(self.sgm**2)*(self.T**2)*self.S0))*
                self.pdfnorm)
        return color;
    
    def Zomma(self):
        gamma=((self.g_disc/(self.sgm*self.S0*np.sqrt(self.T)))*self.pdfnorm)
        zomma=((self.d1*self.d2-1)/(self.sgm))*gamma
        return zomma;
    
    def Ultima(self):
        vega=(self.S0*np.sqrt(self.T)*self.g_disc*(self.pdfnorm))
        ultima=(((-vega)/(self.sgm**2))*(self.d1*self.d2*(1-self.d1*self.d2)+
                self.d1**2 + self.d2**2)*(1/100))
        return ultima;
