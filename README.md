# Normal (Gaussian) Random Number Generators

## The CDF Inversion Method
The numbers are generated by inversion of cumulative distribution function (quantile function).
#### Algorithm
1. Generate random number *u* from [0,1]
2. Find quantile function *F*
3. Calculate 
<img src="https://latex.codecogs.com/gif.latex?x&space;=&space;F^{-1}(u)" title="x = F^{-1}(u)" />


Four functions has been inversed:
1. Tocher:
<img src="https://latex.codecogs.com/gif.latex?\Omega^{-1}(y)&space;\approx&space;\sqrt{\frac{\pi}{8}}&space;ln&space;\frac{y}{y-1}" title="\Omega(x) \approx \frac{e^{2kx}}{1+e^{2kx}}, \hspace{0.2 cm} k=\sqrt{\frac{2}{\pi}} \hspace{1 cm}\rightarrow \hspace{1 cm} \Omega^{-1}(y) \approx \sqrt{\frac{\pi}{8}} ln \frac{y}{y-1}" />
2. Aluudat and Alodat:
<img src="https://latex.codecogs.com/gif.latex?\Omega^{-1}(y)&space;\approx&space;\sqrt{\sqrt{\frac{\pi}{8}}&space;ln&space;\frac{1}{1-(2y-1)^{2}}}" title="\Omega^{-1}(y) \approx \sqrt{\sqrt{\frac{\pi}{8}} ln \frac{1}{1-(2y-1)^{2}}}" />
3. Eidous and Al-Salman
<img src="https://latex.codecogs.com/gif.latex?\Omega^{-1}(y)&space;\approx&space;\sqrt{-\frac{8}{5}&space;ln&space;\frac{1}{1-(2y-1)^{2}}}" title="\Omega^{-1}(y) \approx \sqrt{-\frac{8}{5} ln \frac{1}{1-(2y-1)^{2}}}" />
4. Error function and Maclaurin series
<img src="https://latex.codecogs.com/gif.latex?\Omega^{-1}(y)&space;=&space;\sqrt{2}&space;erf^{-1}(2y-1)" title="\Omega^{-1}(y) = \sqrt{2} erf^{-1}(2y-1)" />
