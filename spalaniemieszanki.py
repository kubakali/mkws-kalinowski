import cantera as ct
import numpy as np
# koniecznosc pobrania matplotlib - do wykresów
import matplotlib.pyplot as plt
import math

# jednostki używane: ciśnienie - Pa, temperatura - K, wymiar - m, objętość - m3, prędkość obrotowa - 1 / s

# poniżej - dane wejściowe

air = 'o2:1, n2:3.76' # skład powietrza
fuel = 'c6h12:1' # skład paliwa które poddajemy spalaniu

# poniżej - warunki powietrza wlot / wylot / otoczenie
Tpow = 300. # temperatura wlotowa
ppow = 1.3e5 # ciśnienie wlotowe
pow_wejsciowe = air # skład powietrza na wlocie
pwyl = 1e5 # ciśnienie wylotowe
Tot = 300 # temperatura otoczenia
pot = 1.013e5 # ciśnienie otoczenia
pow_otoczenia = air # skład powietrza w otoczeniu

# poniżej - warunki paliwa
Tpal = 300. # teomperatura paliwa
ppal = 17e6 # ciśnienie paliwa
pal_wejsciowe = fuel # skład paliwa na wlocie

# poniżej - warunki tłoka
d = 0.1 # średnica tłoka silnikowego
k = 18 # współczynnik kompresji
n = 12. # prędkość obrotowa tłoka
V = 25e-4 # objętość wypchnięta przez tłok
V_s = V / (k - 1) # objętośc skuteczna
A = (1 / 4) * math.pi * d ** 2 # pole powierzhni tłoka
suw = V/A #  suw tłoka

# poniżej - kąty otwarcia / zamknięcia zaworów i wtryskiwacza w radianach
# 1 - moment otwarcia, 2 - moment zamknięcia
wlot_1= -18 / 180 * math.pi# otwarcie 3 stopni po GMP
wlot_2 = 198 / 180 * math.pi # zamknięcie 34 stopnie za DMP
wlot_ni = 1.e-6
wylot_1 = 522 / 180 * math.pi # otwarcie 48 stopni przed DMP
wylot_2 = 18 / 180 * math.pi # zamknięcie 5 stopni przed GMP
wylot_ni = 1.e-6
wtryskiwacz_1 = 170 / 180 * math.pi # wtrysk 90 stopni za GMP
wtryskiwacz_2 = 185 / 180 * math.pi # zamknięcie wtrysku 110 stopni za GMP

m_pal = 1e-5 # masa wtryskiwanego paliwa

sim_n_revolutions = 2
delta_T_max = 20.
rtol = 1.e-12
atol = 1.e-16

def kat_obrotu(t):
    
    return np.remainder(2 * np.pi * n * t, 8 * np.pi)


def predkosc_tloka(t):
    
    return - suw / 2 * 2 * np.pi * n * np.sin(kat_obrotu(t))


gas = ct.Solution('nDodecane_Reitz.yaml', 'nDodecane_IG')
gas.TPX = Tpow, ppow, pow_wejsciowe

cyl = ct.IdealGasReactor(gas)
cyl.volume = V_s

gas.TPX = Tpow, ppow, pow_wejsciowe
wlot = ct.Reservoir(gas)
wlot = ct.Valve(wlot, cyl)
wlot_delta = np.mod(wlot_2 - wlot_1, 4*np.pi)
wlot.valve_coeff = wlot_ni
wlot.set_time_function(lambda t: np.mod(kat_obrotu(t) - wlot_1, 4*np.pi) < wlot_delta)

gas.TPX = Tpal, ppal, pal_wejsciowe
wtryskiwacz = ct.Reservoir(gas)

wtryskiwacz_mfc = ct.MassFlowController(wtryskiwacz, cyl)
wtryskiwacz_delta  = np.mod(wtryskiwacz_2 - wtryskiwacz_1, 4* np.pi)
wtryskiwacz_time = (wtryskiwacz_2 - wtryskiwacz_1)/ 2. / np.pi / n
wtryskiwacz_mfc_coeff = m_pal / wtryskiwacz_time
wtryskiwacz_mfc.set_time_function( lambda t: np.mod(kat_obrotu(t) - wtryskiwacz_1, 4 * np.pi) < wlot_delta)

gas.TPX = Tot, pot, pow_otoczenia
wylot = ct.Reservoir(gas)

wylot = ct.Valve(cyl, wylot)
wylot_delta = np.mod(wylot_2 - wylot_1, 4 * np.pi)
wylot.valve_coeff = wylot_ni
wylot.set_time_function( lambda t: np.mod(kat_obrotu(t) - wylot_1, 4 * np.pi) < wylot_delta)

gas.TPX = Tpow, pwyl, pow_otoczenia
powietrze = ct.Reservoir(gas)

tlok = ct.Wall(powietrze, cyl)
tlok.area = A
tlok.set_velocity(predkosc_tloka)

sim = ct.ReactorNet([cyl])
sim.rtol, sim.atol = rtol, atol
cyl.set_advance_limit('temperature', delta_T_max)

states = ct.SolutionArray(
    cyl.thermo,
    extra=('t', 'ca', 'V', 'dWv_dt'),
)

dt = 1. / (360 * n)
t_stop = sim_n_revolutions / n /2
while sim.time < t_stop:

    sim.advance(sim.time + dt)

    dWv_dt = - (cyl.thermo.P - powietrze.thermo.P) * A * \
        predkosc_tloka(sim.time)

    states.append(cyl.thermo.state,
                  t=sim.time, ca=kat_obrotu(sim.time),
                  V=cyl.volume, dWv_dt=dWv_dt)

def ca_ticks(t):
    return np.round(kat_obrotu(t) * 180 / np.pi, decimals=1)

t = states.t

# poniżej - tworzenie wykresów

xlim1 = 330
xlim2 = xlim1 + 60
xticks = np.arange(300, 420, 5)
xminor = np.arange(300, 420, 0.5)
fig, ax = plt.subplots(nrows=2)
ax[0].plot(ca_ticks(t), states.P / 1.e5)
ax[0].set_title('$C_6H_1$$_2$')
ax[0].margins(x=-0.4)
ax[0].set_ylabel('$p$ [bar]')
ax[0].set_xlabel(r'$\phi$ $[\degree]$')
ax[0].set_xticks(xticks)
ax[0].set_xticklabels(xticks)
ax[0].set_xticks(xminor,minor=True)
ax[0].set_yticks(np.arange(0,300,50))
ax[0].set_yticks(np.arange(0,300,10),minor=True)
ax[0].grid(which='major',linewidth=1)
ax[0].grid(which='minor',linewidth=0.3)
ax[0].set_xlim([xlim1,xlim2])
ax[0].set_ylim([0,250])
ax[1].plot(ca_ticks(t), states.T)
ax[1].margins(x=-0.4)
ax[1].set_ylabel('$T$ [K]')
ax[1].set_xlabel(r'$\phi$ $[\degree]$')
ax[1].set_xticks(xticks)
ax[1].set_xticklabels(xticks)
ax[1].set_xticks(xminor,minor=True)
ax[1].set_yticks(np.arange(0,3500,500))
ax[1].set_yticks(np.arange(0,3500,100),minor=True)
ax[1].grid(which='major',linewidth=1)
ax[1].grid(which='minor',linewidth=0.3)
ax[1].set_xlim([xlim1,xlim2])
ax[1].set_ylim([0,5000])
plt.show()