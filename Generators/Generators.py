import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import kstest
from scipy.stats import skew
from scipy.stats import kurtosis

class NormalGenerators:

    def erf_inv(self, z, pi):
        w = ((pi**0.5)/2)*(z+(pi*(z**3)/12)+(7*(pi**2)*(z**5)/480) + (127*(pi**3)*(z**7)/40320) +
                           (4369*(pi**4)*(z**9)/5806080) + (34807*(pi**5)*(z**11))/182476800)+(z**17)/2
        return float(w)

    def phi(self, x, pi, e):
        return (1/np.sqrt(2*pi))*e**((-x**2)/2)

    # Cumulative distribution function inversion with error function and Maclaurin series
    def CDF_B(self, n):
        res = []
        j = 0
        while j < n:
            x = 2*np.random.uniform(0, 1)-1
            res.append(np.sqrt(2)*self.erf_inv(x, np.pi))
            j += 1
        return res

    # Tocher cumulative distribution function inversion
    def CDF_T(self, n):
        j = 0
        res = []
        while j < n:
            x = np.random.uniform(0, 1)
            t = np.sqrt(np.pi/8)
            res.append(t*np.log(x/(1-x)))
            j += 1
        return res

    # Aludaat and Alodat cumulative distribution function inversion
    def CDF_A(self, n):
        j = 0
        res = []
        a = np.sqrt(8/np.pi)
        while j < int(n/2):
            u = (2*np.random.uniform(0, 1)-1)**2
            x = np.log(1-u)
            res.append(np.sqrt(-a*x))
            j += 1
        while j >= int(n/2) and j < n:
            u = (2*np.random.uniform(0, 1)-1)**2
            x = np.log(1-u)
            res.append(-np.sqrt(-a*x))
            j += 1
        return res

    # Eidous and Al-Salman cumulative distribution function inversion
    def CDF_E(self, n):
        j = 0
        res = []
        while j < int(n/2):
            x = np.random.uniform(0, 1)
            res.append(np.sqrt((8/5)*np.log(1/(1-(2*x-1)**2))))
            j += 1
        while j >= int(n/2) and j < n:
            x = np.random.uniform(0, 1)
            res.append(-np.sqrt((8/5)*np.log(1/(1-(2*x-1)**2))))
            j += 1
        return res

    # Central Limit Theorem
    def CLT(self, n):
        res = []
        j = 0
        while j < n:
            v = sum(np.random.uniform(-0.5, 0.5, 1) for i in range(12))
            res.append(v[0])
            j += 1
        return res

    # Central Limit Theorem, but with range
    def CLT1(self, n, m, a, b):
        res = []
        j = 0
        mu = 0.5*(a+b)
        sig = np.sqrt((1/12)*((b-a)**2))
        while j < n:
            s = sum(np.random.uniform(a, b) for i in range(m))
            res.append((s-m*mu)/(np.sqrt(m)*sig))
            j += 1
        return res

    def BoxMuller(self, n):
        u1 = np.random.uniform(0.0, 1.0, n)
        u2 = np.random.uniform(0.0, 1.0, n)
        a = np.sqrt((-2.0)*np.log(u1))
        b = 2.0*np.pi*u2
        z1 = list(a*np.sin(b))
        z2 = list(a*np.cos(b))
        return [z1, z2]

    def PolarRejection(self, n):
        j = 0
        x, y = [], []
        while j < n:
            u1 = np.random.uniform(-1, 1)
            u2 = np.random.uniform(-1, 1)
            d = u1**2+u2**2
            if d < 1:
                f = np.sqrt((-2.0*np.log(d))/d)
                x.append(float(f*u1))
                y.append(float(f*u2))
                j += 1
        return[x, y]

    def Ziggurat(self, m):
        n = 256
        x = [0, 0.21524189591313286, 0.286174591747172, 0.3357375191803919, 0.37512133285041016, 0.40838913458795284,
             0.4375184021866202, 0.46363433677172483, 0.4874439661217192, 0.5094233295859006, 0.5299097206464645,
             0.5491517023129978, 0.5673382570404457, 0.5846167660936962, 0.6011046177439154, 0.6168969899962115,
             0.6320722363750014, 0.6466957148843666, 0.6608225742341842, 0.6744998228274154, 0.6877678927862372,
             0.7006618410975646, 0.7132122851820035, 0.725446140901285, 0.7373872114258206, 0.7490566620095641,
             0.7604734064220663, 0.7716544242167227, 0.7826150232995724, 0.7933690588331366, 0.8039291169826491,
             0.8143066701280488, 0.8245122087452734, 0.8345553540795038, 0.844444954902409, 0.854189171001546,
             0.8637955455468125, 0.8732710680824803, 0.8826222295788962, 0.8918550707267785, 0.9009752244551607,
             0.9099879534907552, 0.9188981836437213, 0.9277105333962213, 0.9364293402808836, 0.945058684462558,
             0.9536024098755598, 0.9620641432175926, 0.9704473110588522, 0.9787551552889255, 0.9869907470938342,
             0.9951569996299312, 1.0032566795395796, 1.011292417434967, 1.0192667174605181, 1.0271819660307404,
             1.0350404398285944, 1.0428443131393599, 1.0505956645861965, 1.0582964833259958, 1.065948674757496,
             1.073554065787861, 1.0811144096988787, 1.0886313906495035, 1.0961066278475926, 1.1035416794202573,
             1.1109380460092386, 1.1182971741150523, 1.125620459211284, 1.1329092486483265, 1.1401648443639854,
             1.1473885054167237, 1.154581450355842, 1.1617448594415647, 1.1688798767268245, 1.1759876120114805,
             1.1830691426787516, 1.1901255154227925, 1.1971577478755768, 1.204166830140551, 1.2111537262399024,
             1.2181193754817177, 1.2250646937527991, 1.231990574742436, 1.2388978911020185, 1.245787495544989,
             1.2526602218912892, 1.2595168860601358, 1.2663582870146801, 1.2731852076618355, 1.2799984157103252,
             1.2867986644897784, 1.2935866937335097, 1.3003632303274257, 1.3071289890273459, 1.3138846731468612,
             1.3206309752177223, 1.3273685776246171, 1.334098153216076, 1.3408203658931444, 1.3475358711773515,
             1.3542453167594228, 1.3609493430300943, 1.3676485835943104, 1.3743436657700232, 1.3810352110727346,
             1.3877238356868773, 1.3944101509250642, 1.4010947636761955, 1.4077782768433644, 1.4144612897724578,
             1.4211443986723162, 1.4278281970272835, 1.4345132760029398, 1.4412002248457914, 1.447889631277663,
             1.4545820818855166, 1.4612781625074012, 1.467978458615224, 1.4746835556950189, 1.481394039625369,
             1.4881104970546475, 1.4948335157777115, 1.5015636851126994, 1.5083015962785649, 1.5150478427739853,
             1.521803020758286, 1.5285677294350175, 1.5353425714388362, 1.542128153226341, 1.5489250854715289,
             1.5557339834665482, 1.562555467528433, 1.569390163412528, 1.5762387027333267, 1.5831017233934654,
             1.5899798700216428, 1.5968737944202556, 1.6037841560235773, 1.6107116223673281, 1.617656869570529,
             1.624620582830563, 1.6316034569324167, 1.6386061967731056, 1.6456295179023548, 1.652674147080643,
             1.659740822855784, 1.6668302961592811, 1.6739433309237548, 1.6810807047228178, 1.6882432094348532,
             1.6954316519322332, 1.7026468547976086, 1.7098896570690008, 1.7171609150155354, 1.7244615029457706,
             1.731792314050702, 1.739154261283664, 1.7465482782794879, 1.7539753203154502, 1.761436365316702,
             1.7689324149090733, 1.7764644955223405, 1.7840336595472719, 1.7916409865500058, 1.7992875845475762,
             1.8069745913486894, 1.8147031759641636, 1.8224745400917792, 1.8302899196806626, 1.8381505865807246,
             1.8460578502831158, 1.8540130597581441, 1.8620176053976285, 1.8700729210692333, 1.8781804862909743,
             1.886341828534773, 1.894558525668707, 1.9028322085484433, 1.9111645637692791, 1.9195573365912257,
             1.928012334050715, 1.9365314282737556, 1.9451165600067508, 1.953769742382731, 1.9624930649424588,
             1.9712886979317665, 1.9801588968985953, 1.9891060076155682, 1.9981324713565611, 2.0072408305586817,
             2.0164337349043686, 2.0257139478620303, 2.0350843537278065, 2.0445479652157306, 2.0541079316488626,
             2.063767547809954, 2.0735302635169766, 2.0833996939965496, 2.093379631137048, 2.103474055713145,
             2.113687150684932, 2.124023315687814, 2.134487182844319, 2.1450836340462023, 2.1558178198750624,
             2.166695180352645, 2.177721467738641, 2.1889027716247202, 2.2002455466096476, 2.211756642882544,
             2.2234433400909053, 2.235313384928327, 2.247375032945807, 2.259637095172217, 2.272108990226823,
             2.284800802722945, 2.2977233489013287, 2.310888250599849, 2.324308018869622, 2.3379961487950305,
             2.351967227377659, 2.3662370567158177, 2.380822795170625, 2.3957431197804797, 2.411018413899685,
             2.4266709849357255, 2.4427253181989563, 2.459208374333311, 2.476149939669143, 2.4935830412696807,
             2.5115444416253423, 2.530075232158517, 2.5492215503234608, 2.5690354526805366, 2.589575986706995,
             2.6109105184875485, 2.6331163936303246, 2.6562830375755024, 2.680514643284522, 2.705933656121858,
             2.732685359042827, 2.7609440052788226, 2.7909211740007858, 2.822877396825325, 2.8571387308721325,
             2.894121053612348, 2.9343668672078542, 2.978603279880845, 3.0278377917686354, 3.083526132001233,
             3.14788928951715, 3.224575052047029, 3.320244733839166, 3.4492782985609645, 3.654152885361009]
        j = 0
        res = []
        while j < m:
            u1 = np.random.uniform(-1, 1)
            u2 = np.random.uniform(-1, 1)
            i = int(1+np.floor((n-1)*u1))
            z = x[i]*u2
            if z < x[i-1]:
                res.append(z)
                j += 1
            elif i != 0:
                y = (self.phi(x[i], np.pi, np.e)-self.phi(x[i-1], np.pi, np.e))*np.random.uniform(-1, 1)
                if y < self.phi(z, np.pi, np.e)-self.phi(x[i], np.pi, np.e):
                    res.append(z)
                    j += 1
            elif z > x[255]:
                res.append(z)
                j += 1
        return res

    def show(self, method, bins):
        if len(method) > 2:
            plt.figure()
            plt.hist(method, bins, density=True)
            x = np.linspace(-5, 5, 100)
            plt.plot(x, norm.pdf(x, 0, 1), 'k', label = r'$\phi=\frac{1}{\sqrt{2 \pi}}{\rm e}^{\frac{x^{2}}{2}}$')
            plt.ylabel("Frequency")
            plt.xlabel("Values")
            plt.ylim(0.0, 0.45)
            plt.xlim(-5, 5)
            plt.legend(loc='best', frameon=False)
            plt.show()
        elif len(method) == 2:
            plt.figure()
            x = np.linspace(-5, 5, 100)

            plt.subplot(211)
            plt.hist(method[0], bins, density=True)
            plt.plot(x, norm.pdf(x, 0, 1), 'k', label=r'$\phi=\frac{1}{\sqrt{2 \pi}}{\rm e}^{\frac{x^{2}}{2}}$')
            plt.ylabel("Frequency")
            plt.xlabel("Values")
            plt.ylim(0.0, 0.45)
            plt.xlim(-5, 5)
            plt.legend(loc='best', frameon=False)
            plt.subplot(212)
            plt.hist(method[1], bins, density=True)
            plt.plot(x, norm.pdf(x, 0, 1), 'k', label= r'$\phi=\frac{1}{\sqrt{2 \pi}}{\rm e}^{\frac{x^{2}}{2}}$')
            plt.ylabel("Frequency")
            plt.xlabel("Values")
            plt.ylim(0.0, 0.45)
            plt.xlim(-5, 5)
            plt.legend(loc='best', frameon=False)
            plt.show()

    def check_mean_std(self, method):
        res = method
        if len(res) > 2:
            print('expected value=%.3f standard deviation=%.3f' % (np.mean(res), np.std(res)))
        elif len(res) == 2:
            print('1) expected value=%.3f standard deviation=%.3f' % (np.mean(res[0]), np.std(res[0])))
            print('2) expected value=%.3f standard deviation=%.3f' % (np.mean(res[1]), np.std(res[1])))

    def check_skew_kurt(self, method):
        if len(method) > 2:
            print('skewness=%.3f kurtosis=%.3f' % (skew(method), kurtosis(method)))
        elif len(method) == 2:
            print('1) skewness=%.3f kurtosis=%.3f' % (skew(method[0]), kurtosis(method[0])))
            print('2) skewness=%.3f kurtosis=%.3f' % (skew(method[1]), kurtosis(method[1])))

    def check_shapiro(self, method):
        if len(method) > 2:
            stat, p = shapiro(method)
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            alpha = 0.05
            if p > alpha:
                print('H0 cannot be rejected')
            else:
                print('H0 should be rejected')

        elif len(method)==2:
            stat, p = shapiro(method[0])
            print("Sample 1:")
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            alpha = 0.05
            if p > alpha:
                print('H0 cannot be rejected')
            else:
                print('H0 should be rejected')
            stat, p = shapiro(method[1])
            print("Sample 2:")
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            alpha = 0.05
            if p > alpha:
                print('H0 cannot be rejected')
            else:
                print('H0 should be rejected')

    def check_normtest(self, method):
        if len(method) > 2:
            stat, p = normaltest(method)
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            alpha = 0.05
            if p > alpha:
                print('H0 cannot be rejected')
            else:
                print('H0 should be rejected')

        elif len(method)==2:
            stat, p = normaltest(method[0])
            print("Sample 1:")
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            alpha = 0.05
            if p > alpha:
                print('H0 cannot be rejected')
            else:
                print('H0 should be rejected')
            stat, p = normaltest(method[1])
            print("Sample 2:")
            print('Statistics=%.3f, p=%.3f' % (stat, p))
            alpha = 0.05
            if p > alpha:
                print('H0 cannot be rejected')
            else:
                print('H0 should be rejected')

    def check_kstest(self, method):
        if len(method)>2:
            test_stat, pvalue = kstest(method, 'norm', args=(0, 1), N=len(method))
            print("method vs. N(0, 1): KS=%.4f, p = %.4f." % (test_stat, pvalue))

        elif len(method)==2:
            test_stat, pvalue = kstest(method[0], 'norm', args=(0, 1), N=len(method[0]))
            print("method vs. N(0, 1): KS=%.4f, p = %.4f." % (test_stat, pvalue))
            test_stat, pvalue = kstest(method[1], 'norm', args=(0, 1), N=len(method[1]))
            print("method vs. N(0, 1): KS=%.4f, p = %.4f." % (test_stat, pvalue))

    def check_time(self, method,n):
        start_time = time.time()
        method(n)
        print("%s s" % (time.time() - start_time))
