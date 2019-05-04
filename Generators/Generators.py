import cmath
import matplotlib
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import kstest
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import ttest_ind
from scipy.stats import f_oneway


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
        return (1/np.sqrt(2*np.pi))*np.e**((-x**2)/2)

    ZigTab=[0, 0.21524189591313286, 0.286174591747172, 0.3357375191803919, 0.37512133285041016, 0.40838913458795284, 0.4375184021866202, 0.46363433677172483, 0.4874439661217192, 0.5094233295859006, 0.5299097206464645, 0.5491517023129978, 0.5673382570404457, 0.5846167660936962, 0.6011046177439154, 0.6168969899962115, 0.6320722363750014, 0.6466957148843666, 0.6608225742341842, 0.6744998228274154, 0.6877678927862372, 0.7006618410975646, 0.7132122851820035, 0.725446140901285, 0.7373872114258206, 0.7490566620095641, 0.7604734064220663, 0.7716544242167227, 0.7826150232995724, 0.7933690588331366, 0.8039291169826491, 0.8143066701280488, 0.8245122087452734, 0.8345553540795038, 0.844444954902409, 0.854189171001546, 0.8637955455468125, 0.8732710680824803, 0.8826222295788962, 0.8918550707267785, 0.9009752244551607, 0.9099879534907552, 0.9188981836437213, 0.9277105333962213, 0.9364293402808836, 0.945058684462558, 0.9536024098755598, 0.9620641432175926, 0.9704473110588522, 0.9787551552889255, 0.9869907470938342, 0.9951569996299312, 1.0032566795395796, 1.011292417434967, 1.0192667174605181, 1.0271819660307404, 1.0350404398285944, 1.0428443131393599, 1.0505956645861965, 1.0582964833259958, 1.065948674757496, 1.073554065787861, 1.0811144096988787, 1.0886313906495035, 1.0961066278475926, 1.1035416794202573, 1.1109380460092386, 1.1182971741150523, 1.125620459211284, 1.1329092486483265, 1.1401648443639854, 1.1473885054167237, 1.154581450355842, 1.1617448594415647, 1.1688798767268245, 1.1759876120114805, 1.1830691426787516, 1.1901255154227925, 1.1971577478755768, 1.204166830140551, 1.2111537262399024, 1.2181193754817177, 1.2250646937527991, 1.231990574742436, 1.2388978911020185, 1.245787495544989, 1.2526602218912892, 1.2595168860601358, 1.2663582870146801, 1.2731852076618355, 1.2799984157103252, 1.2867986644897784, 1.2935866937335097, 1.3003632303274257, 1.3071289890273459, 1.3138846731468612, 1.3206309752177223, 1.3273685776246171, 1.334098153216076, 1.3408203658931444, 1.3475358711773515, 1.3542453167594228, 1.3609493430300943, 1.3676485835943104, 1.3743436657700232, 1.3810352110727346, 1.3877238356868773, 1.3944101509250642, 1.4010947636761955, 1.4077782768433644, 1.4144612897724578, 1.4211443986723162, 1.4278281970272835, 1.4345132760029398, 1.4412002248457914, 1.447889631277663, 1.4545820818855166, 1.4612781625074012, 1.467978458615224, 1.4746835556950189, 1.481394039625369, 1.4881104970546475, 1.4948335157777115, 1.5015636851126994, 1.5083015962785649, 1.5150478427739853, 1.521803020758286, 1.5285677294350175, 1.5353425714388362, 1.542128153226341, 1.5489250854715289, 1.5557339834665482, 1.562555467528433, 1.569390163412528, 1.5762387027333267, 1.5831017233934654, 1.5899798700216428, 1.5968737944202556, 1.6037841560235773, 1.6107116223673281, 1.617656869570529, 1.624620582830563, 1.6316034569324167, 1.6386061967731056, 1.6456295179023548, 1.652674147080643, 1.659740822855784, 1.6668302961592811, 1.6739433309237548, 1.6810807047228178, 1.6882432094348532, 1.6954316519322332, 1.7026468547976086, 1.7098896570690008, 1.7171609150155354, 1.7244615029457706, 1.731792314050702, 1.739154261283664, 1.7465482782794879, 1.7539753203154502, 1.761436365316702, 1.7689324149090733, 1.7764644955223405, 1.7840336595472719, 1.7916409865500058, 1.7992875845475762, 1.8069745913486894, 1.8147031759641636, 1.8224745400917792, 1.8302899196806626, 1.8381505865807246, 1.8460578502831158, 1.8540130597581441, 1.8620176053976285, 1.8700729210692333, 1.8781804862909743, 1.886341828534773, 1.894558525668707, 1.9028322085484433, 1.9111645637692791, 1.9195573365912257, 1.928012334050715, 1.9365314282737556, 1.9451165600067508, 1.953769742382731, 1.9624930649424588, 1.9712886979317665, 1.9801588968985953, 1.9891060076155682, 1.9981324713565611, 2.0072408305586817, 2.0164337349043686, 2.0257139478620303, 2.0350843537278065, 2.0445479652157306, 2.0541079316488626, 2.063767547809954, 2.0735302635169766, 2.0833996939965496, 2.093379631137048, 2.103474055713145, 2.113687150684932, 2.124023315687814, 2.134487182844319, 2.1450836340462023, 2.1558178198750624, 2.166695180352645, 2.177721467738641, 2.1889027716247202, 2.2002455466096476, 2.211756642882544, 2.2234433400909053, 2.235313384928327, 2.247375032945807, 2.259637095172217, 2.272108990226823, 2.284800802722945, 2.2977233489013287, 2.310888250599849, 2.324308018869622, 2.3379961487950305, 2.351967227377659, 2.3662370567158177, 2.380822795170625, 2.3957431197804797, 2.411018413899685, 2.4266709849357255, 2.4427253181989563, 2.459208374333311, 2.476149939669143, 2.4935830412696807, 2.5115444416253423, 2.530075232158517, 2.5492215503234608, 2.5690354526805366, 2.589575986706995, 2.6109105184875485, 2.6331163936303246, 2.6562830375755024, 2.680514643284522, 2.705933656121858, 2.732685359042827, 2.7609440052788226, 2.7909211740007858, 2.822877396825325, 2.8571387308721325, 2.894121053612348, 2.9343668672078542, 2.978603279880845, 3.0278377917686354, 3.083526132001233, 3.14788928951715, 3.224575052047029, 3.320244733839166, 3.4492782985609645, 3.654152885361009]


    def CDF_inversion():
        x=0.0001
        res=[]
        while x<1.0:
            res.append(norm.ppf(x))
            x+=0.0001
        return(res)

    def CDF_raw(n):
        x=1/n
        res=[]
        while x<1.0:
            y=2*x-1
            res.append(np.sqrt(2)*NormalGen.erf_inv(y))
            x+=1/n
        return(res)


    def CLT(n,m):
        res=[]
        j=0
        while j<n:
            v=[np.random.uniform(-0.5,0.5,1) for i in range(m)]
            z=sum(v)
            res.append(z[0])
            j+=1
        return(res)


    def CLT1(n):
        res=[]
        j=0
        while j<n:
            v=[np.random.uniform(0,1,1) for i in range(50)]
            z=sum(v)
            res.append(z[0]-25)
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
        #x=[0 for i in range(256)]
        #x[255]=3.6541528853610088 #r
        #v=0.00492867323399
        #for j in range(254,0,-1):
            #x[j]=np.sqrt(-2*np.log(v/x[j+1]+NormalGen.fzig(x[j+1])))
        x=NormalGen.ZigTab
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
                y=(1/np.sqrt(2*np.pi))*(np.e**((-x[i-1]**2)/2)-np.e**((-x[i]**2)/2))*np.random.uniform(0.0,1.0,1)
                if y<(1/np.sqrt(2*np.pi))*(np.e**((-z**2)/2)-np.e**((-x[i]**2)/2)):
                    res.append(z[0])
            elif np.abs(z)>x[255]:
                res.append(z[0])
            j+=1
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
        if len(res)>2:
            print('mean=%.3f stdv=%.3f' % (np.mean(res), np.std(res)))
        elif len(res)==2:
            print('1) mean=%.3f stdv=%.3f' % (np.mean(res[0]), np.std(res[0])))
            print('2) mean=%.3f stdv=%.3f' % (np.mean(res[1]), np.std(res[1])))

    def check_shapiro(method):
        if len(method)>2:
            stat, p = shapiro(method)
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


    def check_kstest(method):
        if len(method)>2:
            test_stat, pvalue = kstest(method, 'norm', args=(0, 1), N=len(method))
            print("method vs. N(0, 1): KS=%.4f with p-value = %.4f." % (test_stat, pvalue))

        elif len(method)==2:
            test_stat, pvalue = kstest(method[0], 'norm', args=(0, 1), N=len(method[0]))
            print("method vs. N(0, 1): KS=%.4f with p-value = %.4f." % (test_stat, pvalue))
            test_stat, pvalue = kstest(method[1], 'norm', args=(0, 1), N=len(method[1]))
            print("method vs. N(0, 1): KS=%.4f with p-value = %.4f." % (test_stat, pvalue))

    def check_chi2(method,n):
        Gausspdf=np.random.uniform(0.0,1.0,len(method))
        count1,bins1,ignored1=plt.hist(Gausspdf,n,density=True)
        count2,bins2,ignored2=plt.hist(method,n,density=True)
        stat, p, dof, expected = chi2_contingency([count1,count2])
        #print('dof=%d' % dof)
        #print(expected)
        # interpret test-statistic
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
        if abs(stat) >= critical:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
        # interpret p-value
        alpha = 1.0 - prob
        print('significance=%.3f, p=%.3f' % (alpha, p))
        if p <= alpha:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
        
def independent_ttest(data1, data2, alpha):
	# calculate means
	mean1, mean2 = mean(data1), mean(data2)
	# calculate standard errors
	se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
	sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
	t_stat = (mean1 - mean2) / sed
	# degrees of freedom
	df = len(data1) + len(data2) - 2
	# calculate the critical value
	cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
	p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
	return t_stat, df, cv, p


    def check_time(method,n):
        start_time = time.time()
        method(n)
        print("%s seconds" % (time.time() - start_time))


x=NormalGen.BoxMuller(10000)
NormalGen.check_chi2(x[0],10)
print("******************************** \n")
NormalGen.check_chi2(x[1],10)
print("******************************** \n")
NormalGen.check_tstudent(x[0],10)
print("******************************** \n")
NormalGen.check_tstudent(x[1],10)
print("******************************** \n")
NormalGen.check_kstest(x)
print("********************************** \n")
NormalGen.check_normtest(x)
print("********************************** \n")
NormalGen.check_shapiro(x)