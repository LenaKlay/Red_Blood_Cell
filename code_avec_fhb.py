#premier essai pour le stage

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


# FONCTION 

def func(y,t,Ht,PhiMaxNa,PLNa,PLK,PGNa,PGK,PGA,F,R,T,kCo,kHA,d,QHb,QMg,QX,KB) :

	QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmHb,CmB,CmY,fHb = y

	
	# Flux


	# P : pompe Na+
	
	FluxPNa = -PhiMaxNa*(((QNa/Vw)/((QNa/Vw)+0.2*(1+(QK/(Vw*8.3)))))**3) * ((CmK/(CmK + 0.1 * (1 + (CmK/18))))**2)
	FluxPK  = -FluxPNa/1.5
	FluxLNa = -PLNa * ((QNa/Vw) - CmNa)
	FluxLK  = -PLK  * ((QK/Vw)  - CmK)


	# G : électro-diffusion ( hypothèse : champ électrique constant )

	E = - R*T/F * np.log(( PGNa*QNa/Vw + PGK*QK/Vw + PGA*CmA ) / ( PGNa*CmNa + PGK*CmK + PGA*QA/Vw ))

	FluxGNa = -PGNa * FsurRT * E * (QNa/Vw - CmNa * np.exp(- FsurRT * E))/(1 - np.exp(- FsurRT * E))
	FluxGK  = -PGK  * FsurRT * E * (QK/Vw  - CmK  * np.exp(- FsurRT * E))/(1 - np.exp(- FsurRT * E))
	FluxGA  = +PGA  * FsurRT * E * (QA/Vw  - CmA  * np.exp(+ FsurRT * E))/(1 - np.exp(+ FsurRT * E))

	
	# CO : cotransporteur Na:K:2A

	FluxCo  = -kCo  * (((QA/Vw)**2) * (QNa/Vw) * (QK/Vw) - d * (CmA**2) * CmNa * CmK)
	

	# HA : cotransporteur H:A

	CmH     = KB * CmHb / ( CmB - CmHb )

	FluxHA  = -kHA  * (((QA * QH)/(Vw**2))- CmA * CmH)

	
	# Flux des différents ions (dQ)
	
	dQNadt  = FluxPNa + FluxLNa + FluxGNa + (0.5*FluxCo)
	dQKdt   = FluxPK  + FluxLK  + FluxGK  + (0.5*FluxCo)
	dQAdt   = FluxGA  + FluxHA  + FluxCo
	dQHdt   = FluxHA


	
	# Dérivée du Volume

	# dVwdt = ((dQNadt + dQKdt + dQAdt)/(CmNa + CmK + CmA + CmB + CmY))
	
	dVwdt =  ( (dQKdt + dQAdt) * ( (CmNa + CmK + CmA + CmB + CmY) + ( (fHb*QHb + QNa + QK + QA + QMg + QX) * (Ht/1-Ht) ) ) ) / ( (CmNa + CmK + CmA + CmB + CmY) * ( (CmNa + CmK + CmA + CmB + CmY) + ( (fHb*QHb + QNa + QK + QA + QMg + QX) * (Ht/1-Ht) ) + (b*(QHb**2)/(Vw**2) ) + (2*c*(QHb**3)/(Vw**3) ) ) )
						

	# Coefficient osmotique de l'hémoglobine 

	dfHbdt = - dVwdt * ( (b*QHb/(Vw**2) ) + (2*c*(QHb**2)/(Vw**3) ) )

	
	# Dérivées des concentrations

	dCmNadt = (Ht/(1 - Ht)) * (dVwdt*CmNa - dQNadt)
	dCmKdt  = (Ht/(1 - Ht)) * (dVwdt*CmK  - dQKdt)
	dCmAdt  = (Ht/(1 - Ht)) * (dVwdt*CmA  - dQAdt)

	dCmHbdt = (Ht/(1 - Ht)) * (dVwdt*CmHb - dQHdt)
	dCmBdt  = (Ht/(1 - Ht)) * (dVwdt*CmB)
	#dCmHdt  =  KB * ( (dCmBdt*(CmB-CmHb)) - (CmB*(dCmBdt-dCmHbdt)) ) / ((CmB-CmHb)**2)						
	dCmYdt  = (Ht/(1 - Ht)) * (dVwdt*CmY)
	

	dydt    = [dQNadt,dQKdt,dQAdt,dQHdt,dVwdt,dCmNadt,dCmKdt,dCmAdt,dCmHbdt,dCmBdt,dCmYdt,dfHbdt ]		

	return dydt


	

	dydt    = [dQNadt,dQKdt,dQAdt,dQHdt,dVwdt,dCmNadt,dCmKdt,dCmAdt,dCmHbdt,dCmBdt,dCmYdt,dfHbdt]		

	return dydt



# COEFFICIENTS 


Ht      = 0.1		# 1
PhiMaxNa= 8.99		# mmol/(l*h)

F       = 96485
E       = -0.0086	# V
R       = 8.314
T       = 310		# K
FsurRT  = F/(R*T)       # 1/V

d       = 1.05		# 1
b 	= 0.0645	# 1
c	= 0.0258	# 1 

#fHb     = 2.78		# 1
QHb     = 5		# mmol/l
QMg     = 2.5		# mmol/l
QX      = 19.2		# mmol/l
KB      = 10**-4.55	# mmol/l
Vw0     = 0.7		# 1


# Constante qui varie

PGA = 2.0		# 1/h	# 0.2 a 200


# Changements d'un cas à l'autre

# Cas 1 : avec fG = 0.1 et mode 'off'
#PLNa    = 0.0180	# 1/h
#PLK     = 0.0116	# 1/h
#PGNa    = 0.0017	# 1/h
#PGK     = 0.0015	# 1/h
#kCo     = 10**-9	# 1
#kHA     = 1		# 1

# Cas 2 : avec fG = 0.9 et mode 'on'
PLNa    = 0.0020	# 1/h
PLK     = 0.0013	# 1/h
PGNa    = 0.0151	# 1/h
PGK     = 0.0138*10**4	# 1/h
kCo     = 10**-6	# 1
kHA     = 10**9		# 1

# INITIALISATION 
										
y0 = np.array([Vw0*10, Vw0*140, Vw0*95, Vw0*10**(-7.26)*1000, Vw0, 140, 5, 131, 5.86, 10, 10, 2.78])	   	  # cmH enlevé = 10**(-7.4)*1000
       #         QNa,       QK,    QA,            QH,          Vw, CmNa,CmK,CmA, CmHb, CmB,CmY, fHb


t0   = 0
tmax = 1
Npts = 1001
t = np.linspace(t0, tmax, Npts)

sol = odeint(func, y0, t, args=(Ht,PhiMaxNa,PLNa,PLK,PGNa,PGK,PGA,F,R,T,kCo,kHA,d,QHb,QMg,QX,KB))


# PARAMETRE DES GRAPHIQUES

pHc = - np.log10(sol[:,3]/(1000*sol[:,4]))			# pHc = -log10(QH/(1000*Vw))					# 
pHm = - np.log10(KB * sol[:,8] / ( sol[:,9] - sol[:,8] ))	# pHm = -log10(CmH/1000) et CmH = KB*CmHb/(CmB-CmHb)
	
E = - R*T/F * np.log(( PGNa*sol[:,0]/sol[:,4] + PGK*sol[:,1]/sol[:,4] + PGA*sol[:,7] ) / ( PGNa*sol[:,5] + PGK*sol[:,6] + PGA*sol[:,2]/sol[:,4] ))
# E = - R*T/F * np.log(( PGNa*QNa/Vw + PGK*QK/Vw + PGA*CmA ) / ( PGNa*CmNa + PGK*CmK + PGA*QA/Vw ))


# GRAPHIQUE 


plt.figure()					       	# vecteur y = QNa,QK,QA,QH,Vw,CmNa,CmK,CmA,CmHb,CmB,CmY,fHb
							#              0,  1, 2, 3, 4, 5,   6,  7,  8,   9,  10, 11

plt.subplot(2, 3, 1)
plt.plot(t, sol[:,1] , 'green', label='QK')
plt.plot(t, sol[:,2] , 'blue', label='QA')
plt.legend(loc='best')
plt.grid()

plt.subplot(2, 3, 2)
plt.plot(t, E, label='E') 	
plt.legend(loc='best')
plt.grid()


plt.subplot(2, 3, 3) 		
plt.plot(t, pHm, label='pHm')
#plt.plot(t, pHm, label='pHm2')
plt.plot(t, pHc, label='pHc')
plt.legend(loc='best')
plt.grid()

plt.subplot(2, 3, 4)
plt.plot(t, sol[:,4]/Vw0, label='Vw/Vw0') 		
plt.legend(loc='best')
plt.grid()

plt.subplot(2, 3, 5)
plt.plot(t, sol[:,11], label='fHb') 		
plt.legend(loc='best')
plt.grid()
plt.show()
