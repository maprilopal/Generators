import cmath
import matplotlib
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import normaltest

class NormalGen:
    def __init__(self):
        pomoc.self=self

    def silnia(n):
        if n>1:
            return n*silnia(n-1)
        else:
            return 1

    def f_bl(x,n):
        i=0
        y=0
        d=2/(np.sqrt(np.pi))
        while i<n:
            y+=d*(((-1)**i)*(x**(2*i+1)))/((2*i+1)*NormalGen.silnia(i))
            i+=1
        return y

    def erf_inv(z):
        w=(np.sqrt(np.pi)/2)*(z+((np.pi*(z**3))/12)+((7*(np.pi**2)*(z**5))/480)+((127*(np.pi**3)*(z**7))/40320)+((4369*(np.pi**4)*(z**9))/5806080))
        return w

    def fzig(x):
        return np.e**((-x**2)/2)


    def CDF_inversion():
        x=0.001
        res=[]
        while x<1.0:
            res.append(norm.ppf(x))
            x+=0.001
        return(res)

    def CDF_raw(n):
        x=0
        res=[]
        while x<1:
            res.append(np.sqrt(2)*NormalGen.erf_inv(2*x-1))
            x+=1/n
        return(res)


    def CLT(n):
        res=[]
        j=0
        while j<n:
            v=[np.random.uniform(-0.5,0.5,1) for i in range(20)]
            z=sum(v)
            res.append(z[0])
            j+=1
        return(res)

    def BoxMuller(n):
        u1=np.random.uniform(0.0,1.0,n)
        u2=np.random.uniform(0.0,1.0,n)
        a=np.sqrt((-2.0)*np.log(u1))
        b=(2.0)*math.pi*u2
        z1=a*np.sin(b)
        z2=a*np.cos(b)
        return(z1,z2)


    def PolarReject(n):
        x=2.0*np.random.uniform(-1,1,n) #v1
        y=2.0*np.random.uniform(-1,1,n) #v2
        d=x**2+y**2
        i=0
        fx,fy=[],[]
        while(i<n):
            if(d[i]>0 and d[i]<1):
                fx.append(x[i]*np.sqrt((-2.0*np.log(d[i]))/d[i]))
                fy.append(y[i]*np.sqrt((-2.0*np.log(d[i]))/d[i]))
            i+=1
        return(fx,fy)



    #wymaga zmian
    def Marsaglia_Bray(n):
        s=np.random.uniform(0.0,1.0,n)
        u=[0 for y in range(n)]
        for i in range(n):
            u[i]=np.random.uniform(0,1,7)
        #print(u)
        #plt.hist(u)
        #plt.show()
        a0=0.8638
        a1=0.1107
        a2=0.0228002039
        a3=1-a0-a1-a2
        res=[]
        for i in range(n):
            g0=2*(u[i][0]+u[i][1]+u[i][2]-1.5)
            g1=1.5*(u[i][3]+u[i][4]-1)
            x=6*u[i][5]-3
            y=0.358*u[i][6]
            g2=norm.cdf(x)-(a0*g0+a1*g1)/(a0+a1+a2)
            if s[i]<a0:
                res.append(g0)
            elif s[i]<a0+a1:
                res.append(g1)
            elif s[i]<a0+a1+a2:
                k=0
                while y>g2 and k<2:
                    x=6*u[i][5]-3
                    y=0.358*u[i][6]
                    g2=norm.cdf(x)-(a0*g0+a1*g1)/(a0+a1+a2)
                    k+=1
                res.append(x)
        print(res)
        plt.figure()
        plt.hist(res,1000)
        plt.show()


    def Ratio_of_Uniforms(n):
        u=np.random.uniform(0.0,1.0,n)
        v=np.random.uniform(0.0,1.0,n)
        i=0
        res=[]
        while i<n:
            x=(v[i]*np.sqrt(2/np.e))/u[i]
            if x**2<=5-4**(1/4)*u[i]:
                res.append(x)
            elif x**2<(4*np.e**(-1.35))/u[i]+1.4:
                if v[i]**2<-4*u[i]**2*np.log(u[i]):
                    res.append(x)
            i+=1
        return(res)


        #działa
    def Leva_Ratio(n):
        u=np.random.uniform(0.0,1.0,n)
        v=np.sqrt(2/np.e)*np.random.uniform(0.0,1.0,n)
        s=0.449871
        t=-0.386595
        r1=0.27597
        r2=0.27846
        a=0.196
        b=0.25472
        i=0
        res=[]
        x=[0 for j in range(n)]
        y=[0 for j in range(n)]
        while i<n:
            x[i]=u[i]-s
            y[i]=np.abs(v[i])-t
            Q=x[i]**2+a*y[i]**2-b*x[i]*y[i]
            if Q<r1:
                res.append(v[i]/u[i])
            elif Q<r2:
                if v[i]**2<-4*u[i]**2*np.log(u[i]):
                    res.append(v[i]/u[i])
            i+=1
        return(res)

    def Grand():
        i=0
        x=-1
        while x<0:
            x=np.random.uniform(0.0,1.0,1)
        print(x)
        while x<0.5:
            x=2*x
            i+=1
        if x<[1]:
            print("mniejsze")
        a=[norm.ppf(1-2**(-j-1)) for j in range(6)]
        G=[1/2*(x**2-a[j]**2) for j in range(6)]
        u=(a[i+1]-a[i])*G[1]
        v=u*(u/2+a[i])
        while i>-1:
            u=(a[i+1]-a[i])*G[1]
            v=u*(u/2+a[i])
            while v>G[5]:
                if v<G[2]:
                    if G[3]<0.5:
                        print(a[i]+u)
                        return
                    else:
                        print(-a[i]-u)
                        return
                else:
                    v=G[4]



    def Ziggurat(m):
        n=256
        x=[0 for i in range(256)]
        x[255]=3.6541528853610088 #r
        v=0.00492867323399
        for j in range(254,0,-1):
            x[j]=np.sqrt(-2*np.log(v/x[j+1]+NormalGen.fzig(x[j+1])))
        j=0
        res=[]
        while j<m:
            n=np.random.randint(0,255,1)
            u1=np.random.uniform(0.0,1.0,1)
            u2=np.random.uniform(0.0,1.0,1)
            i=int(1+np.floor(n*u1)[0])
            z=x[i]*u2
            if np.abs(z)<x[i-1]:
                res.append(z[0])
            elif i!=n:
                y=(np.e**((-x[i-1]**2)/2)-np.e**((-x[i]**2)/2))*np.random.uniform(0.0,1.0,1)
                if y<np.e**((-z**2)/2)-np.e**((-x[i]**2)/2):
                    res.append(z[0])
            j+=1
        res1=[]
        for i in range(len(res)):
            res1.append(-res[i])
        res=res+res1
        return(res)

    def show(method, den):
        if len(method)>2:
            plt.figure()
            plt.hist(method,den,density=True)
            plt.show()
        elif len(method)==2:
            plt.figure()
            plt.subplot(211)
            plt.hist(method[0],den,density=True)
            plt.subplot(212)
            plt.hist(method[1],den,density=True)
            plt.show()

    def check_mean_std(method):
        res=method
        if len(method)>2:
            print('mean=%.3f stdv=%.3f' % (np.mean(res), np.std(res)))
        elif len(method)==2:
            print('1) mean=%.3f stdv=%.3f' % (np.mean(res[0]), np.std(res[0])))
            print('2) mean=%.3f stdv=%.3f' % (np.mean(res[1]), np.std(res[1])))

    def check_shapiro(method):
        if len(method)>2:
            stat, p = normaltest(method)
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            # interpret
            alpha = 0.05
            if p > alpha:
                print('Sample looks Gaussian (fail to reject H0)')
            else:
                print('Sample does not look Gaussian (reject H0)')

        elif len(method)==2:
            stat, p = shapiro(method[0])
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            # interpret
            alpha = 0.05
            if p > alpha:
                print('Sample looks Gaussian (fail to reject H0)')
            else:
                print('Sample does not look Gaussian (reject H0)')
            stat, p = shapiro(method[1])
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            # interpret
            alpha = 0.05
            if p > alpha:
                print('Sample looks Gaussian (fail to reject H0)')
            else:
                print('Sample does not look Gaussian (reject H0)')

    def check_normtest(method):
        if len(method)>2:
            stat, p = normaltest(method)
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            # interpret
            alpha = 0.05
            if p > alpha:
                print('Sample looks Gaussian (fail to reject H0)')
            else:
                print('Sample does not look Gaussian (reject H0)')

        elif len(method)==2:
            stat, p = normaltest(method[0])
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            # interpret
            alpha = 0.05
            if p > alpha:
                print('Sample looks Gaussian (fail to reject H0)')
            else:
                print('Sample does not look Gaussian (reject H0)')
            stat, p = normaltest(method[1])
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            # interpret
            alpha = 0.05
            if p > alpha:
                print('Sample looks Gaussian (fail to reject H0)')
            else:
                print('Sample does not look Gaussian (reject H0)')


