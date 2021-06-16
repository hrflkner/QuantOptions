# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 19:40:20 2021

@author: Hunter Faulkner
"""

import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class stockstats:
    def __init__(self,ticker,start,end):
        self.ticker=ticker;
        self.start=start;
        self.end=end;
        self.S=yf.download(self.ticker,
                               self.start,
                               self.end);
        self.o=self.S.Open
        self.h=self.S.High
        self.l=self.S.Low
        self.c=self.S.Close
        
    def Stock_Price(self):
        return self.S;
    
    def HV(self):
        hv=pd.DataFrame()
        for s in self.ticker:
            hv[s]=[np.log(self.c[s]/self.c[s].shift(1)).std()];
        return hv;
    
    def RSHV(self):
        rshv=pd.DataFrame()
        for s in self.ticker:
            rshv[s]=[(np.sqrt((sum(
                np.log(self.h[s]/self.c[s])*np.log(self.h[s]/self.o[s])+
                np.log(self.l[s]/self.c[s])*np.log(self.l[s]/self.o[s]))/
                len(self.c[s]))))]
        return rshv
    
    def YZHV(self):
        yzhv=pd.DataFrame()
        for s in self.ticker:
            rshv=(np.sqrt((sum(
                np.log(self.h[s]/self.c[s])*np.log(self.h[s]/self.o[s])+
                np.log(self.l[s]/self.c[s])*np.log(self.l[s]/self.o[s]))/
                len(self.c[s]))))     
            k=(.34)/(1.34 + ((len(self.c[s])+1)/(len(self.c[s])-1)))
            on=(sum((np.log(self.o[s]/self.c[s].shift(1)).dropna()-
                     np.log(self.o[s]/self.c[s].shift(1)).dropna().mean())**2)*
                    (1/(len(self.c[s])-1)))
            otoc=((1/(len(self.c[s])-1))*sum((np.log(self.c[s]/self.o[s])-
                np.log(self.c[s]/self.o[s]).mean())**2))
            yzhv[s]=[np.sqrt(on**2+k*otoc**2+(1-k)*rshv**2)]
        return yzhv
    
    def Correlation(self,graph=False):
        correlation=(self.c).corr()
        if graph==True:
            plt.figure(figsize=(12,8))
            hm=sns.heatmap(correlation,vmin=-1,vmax=1,
                           annot=True,cmap='mako')
            d_range=f'{self.start} : {self.end}'
            hm.set_title(f'Asset Correlation [{d_range}]',
                         fontsize=16,fontweight='bold')
            plt.xticks(fontweight='bold')
            plt.yticks(fontweight='bold')
            plt.show()
        return correlation