from scipy.stats import f
from scipy.stats import t
import numpy as np
import pandas as pd






class MRLS:
 
  def __init__(self,Intercepto = True):
    self.beta0=False
    self.beta1=False
    self.matrizCov=False
    self.intercepto = Intercepto

  def Treino(self, x, y):

    ybarra = y.mean()  
    xbarra = x.mean() 
    MRLS.Mbeta1(self,x,xbarra,y,ybarra)
    MRLS.Mbeta0(self,xbarra,ybarra)


  def Mbeta1(self,x,xbarra,y,ybarra):

    if(self.intercepto):
        #modelo com intercepto
        sxy = MRLS.MSxy(self,y,x,ybarra)       
        sxx = MRLS.MSxx(self,x,xbarra)

        beta_1 = (sxy)/(sxx)
    else:
        beta_1 = (x@y)/(x@x)

    self.beta1 = beta_1

  
  def Mbeta0(self,xbarra,ybarra):
    if(self.intercepto):
        beta_0 = (ybarra)-(self.beta1)*(xbarra)
    else:
        beta_0 = 0
    self.beta0 = beta_0

 

  #teste de H0:beta1=0
  def Teste_B1(self,y,y_predict):
    ybarra = y.mean()
    n = np.shape(y)[0]

    sqreg = MRLS.MSqreg(self,y_predict,ybarra)

    qmres = MRLS.MQmres(self,y,y_predict)

    
    F = sqreg/qmres
    pvalor = 1-f.cdf(F, 1, n-2)
    if(self.pvalor < 0.05):
        return pvalor, False#rejeito H0
    else:
        return pvalor,True#aceito H0

 
 
  #teste de H0:beta0=0
  #Ja precisa ter treinado o modelo para se usar
  def Teste_B0(self,x,y,y_predict):
    n = np.shape(x)[0]
    xbarra = x.mean()

    sxx = MRLS.MSxx(self,x,xbarra)

  
    sqres = MRLS.MSqres(self,y,y_predict)
    Qmres = sqres/(n-2)

    t0 = self.beta0/(Qmres*((1/n)+(xbarra**2)/(sxx)))**(1/2)
    aux = [t.cdf(t0, n-2),1-t.cdf(t0, n-2)]

    pvalor = 2*(np.min(aux))
    if(pvalor < 0.05):
        return pvalor, False#rejeito H0
    else:
        return pvalor, True#aceito H0

 
 
  def MSqreg(self,y_predict,ybarra):
    if(self.intercepto):
        #modelo com intercepto
        sqreg = (np.linalg.norm(y_predict-ybarra, ord=2)**2)
    else:
        sqreg = (np.linalg.norm(y_predict, ord=2)**2)
    
    return sqreg
    
  
 
  def MSqt(self,y,ybarra):

    if(self.intercepto):
      # modelo com intercepto
      sqt=(np.linalg.norm(y-ybarra, ord=2))**2
    else:
      sqt=(np.linalg.norm(y, ord=2))**2
    
    return sqt
 

  def MQmres(self,y,y_predict):
    n = np.shape(y)[0]

    sqres = MRLS.MSqres(self,y,y_predict)

    if(self.intercepto):
      qmres = sqres/(n-2)
    else:
      qmres = sqres/(n-1)
    
    return qmres
 
 
  def MSqres(self,y,y_predict):
    self.Sqres=(np.linalg.norm(y-y_predict,ord=2))**2
    return self.Sqres


  def MSxx(self,x,xbarra):
    sxx = 0
    n = np.shape(x)[0]
    for i in range(n):
        sxx+=(x[i]-xbarra)**2
    return sxx


  def MSxy(self,y,x,ybarra):
    sxy = 0
    n = np.shape(x)[0]
    for i in range(n):
        sxy += (y[i]-ybarra)*x[i]
    return sxy
 
        
  def Predizer(self,x):
    y_predict = (self.beta0)+(self.beta1)*x
    y_predict = y_predict.reshape(-1)
    return y_predict


  def CoefDet(self,x,y):
    ybarra = y.mean()
    y_predict = MRLS.Predizer(self,x)

    Sqreg = MRLS.MSqreg(self,y_predict,ybarra)

    Sqt = MRLS.MSqt(self,y,ybarra)
    return Sqreg/Sqt
 
 
  def MatrizCov(self,x,y,y_predict):
    xbarra = x.mean()
    n = np.shape(x)[0]

    qmres = MRLS.MQmres(self,y,y_predict)

    sxx = MRLS.MSxx(self,x,xbarra)

    if(self.intercepto):
      Cov=np.zeros((2,2))
      Cov[0,0] = qmres*(1/n+(xbarra**2)/sxx)
      Cov[1,0] = Cov[0,1] = (-qmres*xbarra)/sxx
      Cov[1,1] = qmres/sxx
    else:
      Cov = qmres/((x)@(x))
    
    return Cov
    
 
  def Anova(self,y,y_predict):
    ybarra = y.mean()
    n = np.shape(y)[0]

    sqreg = MRLS.MSqreg(self,y_predict,ybarra)

    sqt = MRLS.MSqt(self,y,ybarra)

    sqres = MRLS.MSqres(self,y,y_predict)

    E=np.zeros((3, 3))
    #GL
    E[0,0] = 1
    E[1,0] = n-2
    E[2,0] = n-1
    #SQ
    E[0,1] = sqreg
    E[1,1] = sqres
    E[2,1] = sqt
    #QM
    E[0,2] = sqreg       
    E[1,2] = sqres/(n-2)
    E[2,2] = sqt/(n-1)
    coluna = 'GL SQ QM'.split()
    linha = 'Regressao Residuo Total'.split()
    anova = pd.DataFrame(data=E, index=linha, columns=coluna)
    return anova