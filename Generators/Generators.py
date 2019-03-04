import cmath
import matplotlib
import math
import random
import numpy as nmp
import matplotlib.pyplot as plot
from scipy.stats import norm

def CDF_inversion():
    x=0.001
    res=[]
    while x<1.0:
        res.append(norm.ppf(x))
        x+=0.001
    plot.figure()
    plot.hist(res)
    plot.show()


def CLT1():
    v=[nmp.random.uniform(0.0,1.0,1) for i in range(12)]
    z=sum(v)-6
    print(z)
    plot.figure()
    plot.hist(v)
    plot.show()

def BoxMuller():
    u1=nmp.random.uniform(0.0,1.0,1000000)
    u2=nmp.random.uniform(0.0,1.0,1000000)
    print("log=", nmp.log(u1))
    a=nmp.sqrt((-2.0)*nmp.log(u1))
    b=(2.0)*math.pi*u2
    z1=a*nmp.sin(b)
    z2=a*nmp.cos(b)
    plot.figure()

    plot.subplot(221)
    plot.hist(u1)
    plot.subplot(222)
    plot.hist(u2)
    plot.subplot(223)
    plot.hist(z1)
    plot.subplot(224)
    plot.hist(z2)

    plot.show()


def fc(x,y,a,b):
    dx=x-a
    x1=a-dx
    dy=b-y
    y1=b+dy
    return (x1,y1)

def MontyPython1():
    Gausspdf=nmp.random.uniform(0.0,1.0,1000)
    print("Gausspdf:\n",Gausspdf)
    #plot.figure()
    #bins to pkt (101 pkt)
    #count to wartości funkcji w pkt (100pkt)
    count,bins,ignored=plot.hist(Gausspdf,1000)
    #print("bins:\n",bins,len(bins))
    #print("count:\n",count,len(count))
    #plot.show()
    u1=Gausspdf #all numbers
    s=2.0*nmp.floor(2.0*u1)+1 #+1 is choosing side
    print("len of u1 ",len(u1)," len of s ", len(s))
    u2=bins #horizontal component of the uniform 2D random sample
    b=nmp.sqrt(2/nmp.pi)
    a=norm.ppf(1/2*b)
    #print("aaaaaaaaaaaaaaaaaa",a)
    x=b*u2
    print("u2",len(u2)," x", len(x))
    u3=count #vertical
    print("len(count):",len(u3))
    res=[]
    y=[0 for i in range(len(u3))]
    i=0
    while i<len(u3):
        #print("CDF z x",norm.cdf(x[i]))
        #print("for A area")
        y[i]=u3[i]/(2.0*b)
        if x[i]<a:#a[i]
            res.append(s[i]*x[i])
        if y[i]<norm.cdf(x[i]):
            #print("for B area")
            res.append(s[i]*x[i])
        if fc(x[i],y[i],a,b)[1]<norm.cdf(x[i]):
            res.append(s[i]*fc(x[i],y[i],a,b)[0])
            #print("s=",s[i],"   s[i]*fc(x[i],y[i],a,b)[0]=",s[i]*fc(x[i],y[i],a,b)[0])
        i+=1
    print("len res", len(res))
    print(res)
    plot.figure()
    plot.hist(res)
    plot.show()


def PolarReject():
    x=2.0*nmp.random.uniform(-1,1,1000000) #v1
    y=2.0*nmp.random.uniform(-1,1,1000000) #v2
    d=x**2+y**2
    print("x=",x,"\ny=",y,"\nd=",d)
    i=0
    fx,fy=[],[]
    while(i<1000):
        if(d[i]>0 and d[i]<1):
            fx.append(x[i]*nmp.sqrt((-2.0*nmp.log(d[i]))/d[i]))
            fy.append(y[i]*nmp.sqrt((-2.0*nmp.log(d[i]))/d[i]))
        i+=1
    print("fx:",fx,"\nfy:",fy)
    plot.figure()
    plot.subplot(221)
    plot.hist(fx)
    plot.subplot(222)
    plot.hist(fy)
    plot.show()


#nie działa
def Marsaglia_Bray():
    s=nmp.random.uniform
    u=nmp.random.uniform(0.0,1.0,7)
    #plot.hist(u)
    #plot.show()
    a0=0.8638
    a1=0.1107
    a2=0.0228002039
    a3=1-a0-a1-a2
    s=u
    res=[]
    g0=2*(u[0]+u[1]+u[2]-1.5)
    g1=1.5*(u[3]+u[4]-1)
    x=6*u[5]-3
    y=0.358*u[6]
    g2=norm.cdf(x)-(a0*g0+a1*g1)/(a0+a1+a2)
    if s[0]<a0:
        res.append(g0)
    elif s[0]<a0+a1:
        res.append(g1)
    elif s[0]<a0+a1+a2:
        while y>g2:
            x=6*u[5]-3
            y=0.358*u[6]
        res.append(x)
    print(res)
    plot.figure()
    plot.hist(res)
    plot.show()


def Ratio_of_Uniforms():
    u=nmp.random.uniform(0.0,1.0,100000)
    v=nmp.random.uniform(0.0,1.0,100000)
    i=0
    res=[]
    while i<len(u):
        x=(v[i]*nmp.sqrt(2/nmp.e))/u[i]
        if x**2<=5-4**(1/4)*u[i]:
            res.append(x)
        elif x**2<(4*nmp.e**(-1.35))/u[i]+1.4:
            if v[i]**2<-4*u[i]**2*nmp.log(u[i]):
                res.append(x)
        i+=1
    print(res)
    plot.figure()
    plot.subplot(221)
    plot.hist(u)
    plot.subplot(222)
    plot.hist(v)
    plot.subplot(223)
    plot.hist(res)
    plot.show()

def Leva_Ratio():
    u=nmp.random.uniform(0.0,1.0,1000000)
    v=nmp.sqrt(2/nmp.e)*nmp.random.uniform(0.0,1.0,1000000)
    s=0.449871
    t=-0.386595
    r1=0.27597
    r2=0.27846
    a=0.196
    b=0.25472
    i=0
    res=[]
    x=[0 for j in range(len(u))]
    y=[0 for j in range(len(u))]
    while i<len(u):
        x[i]=u[i]-s
        y[i]=nmp.abs(v[i])-t
        Q=x[i]**2+a*y[i]**2-b*x[i]*y[i]
        if Q<r1:
            res.append(v[i]/u[i])
        elif Q<r2:
            if v[i]**2<-4*u[i]**2*nmp.log(u[i]):
                res.append(v[i]/u[i])
        i+=1
    print(res)
    plot.figure()
    plot.subplot(221)
    plot.hist(u)
    plot.subplot(222)
    plot.hist(v)
    plot.subplot(223)
    plot.hist(res)
    plot.show()

def Grand():
    i=0
    x=-1
    while x<0:
        x=nmp.random.uniform(0.0,1.0,1)
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

def fzig(x):
    return nmp.e**((-x**2)/2)


def Ziggurat():
    n=256
    #nmp.exp
    x=[0 for i in range(256)]
    x[255]=3.6541528853610088 #r
    v=0.00492867323399
    for j in range(254,0,-1):
        x[j]=nmp.sqrt(-2*nmp.log(v/x[j+1]+fzig(x[j+1])))
    #print(x)
    j=0
    res=[]
    while j<1000:
        n=nmp.random.randint(0,255,1)
        u1=nmp.random.uniform(0.0,1.0,1)
        u2=nmp.random.uniform(0.0,1.0,1)
        i=int(1+nmp.floor(n*u1)[0])
        #print(i)
        z=x[i]*u2
        if nmp.abs(z)<x[i-1]:
            res.append(z[0])
        elif i!=n:
            y=(fzig(x[i-1])-fzig(x[i]))*nmp.random.uniform(0.0,1.0,1)
            if y<fzig(z)-fzig(x[i]):
                res.append(z[0])
        j+=1
    print(res)
    plot.figure()
    plot.hist(res)
    plot.show()




#print(fc(9,2,5,3))