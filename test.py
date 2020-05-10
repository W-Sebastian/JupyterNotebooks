# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# <a href="https://colab.research.google.com/github/W-Sebastian/JupyterNotebooks/blob/master/Structuri_U%C8%99oare_Grind%C4%83.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
# we need some extra python packages installed; let's make sure we get them now
# get_ipython().system('pip install pyaml')
# get_ipython().system('pip install pyswarms')

# %% [markdown]
# # Analiza grindei sandwich: studiu de caz pentru o trambulină
# 
# ## Introducere
# 
# ## Definirea problemei
# 
# Se propune pentru analiză studiul asupra unei trambuline de piscină de tip structură sandwich. Plecăm de la următoarea schiță:
# ![schita trambulina](img/schita.png)
# 
# Modelul are următoarele constrângeri:
# - Materialele folosite pentru înveliș respectic miez sunt date;
# - Lungimea grinzii trebuie să fie între 1 și 4 metri;
# - Se impune o rigiditate a grinzii de maxim 5 N/mm;
# - Un capăt de grindă este considerat încastrat;
# - Capătul liber are o condiție la limită definită prin masă de 150 Kg și accelerație de 9.834 m/s^2;
# - Lățimea grinzii este de 500 mm;
# - Factorul de siguranță pentru calculul tensiunilor este 5;
# - Costul materialelor pentru construcția grinzii nu trebuie să depășească 200€; prețul pentru fiecare material este dar în €/Kg.
# 
# Se propune alegerea materialelor pentru înveliș și miez și alegerea lungimii respectiv a grosimii miezului și învelișului astfel încât:
# - Să nu se treacă de costul impus;
# - Să se reducă masa cât mai mult.
# 
# Din datele problemei ne vom propune următoarele obiective:
# - Definirea modelelor analitice de calcul;
# - Corelarea rezultatelor analitice cu o analiză de element finit;
# - Găsirea combinației care ne oferă cea mai mică masă respectând limitările date;
# - Adițional vom explora și găsirea celui mai mic preț care să respecte limitările date;
# 
# ## Considerente limitative
# 
# - Toate valorile din document sunt ținute în MKS (meter/kilogram/second) pentru a păsta uniformitatea calculelor și a reduce potențiale probleme legate de conversia de unit-uri. Conversile din alte unităti (mm, MPa) sunt făcut înainte sau după calculele propriu-zise. Pentru presiuni vom folosi unitatea Pa.
# - Se consideră în calcule o grindă cu înveliș exterior aplicat doar deasupra și sub grindă, fără a modela marginile ei deoarece le vom considera neglijabile;
# - Pe cât posibil se vor folosi abrevierile din formule pentru concepte alăturate de comentarii care vor explica semnificația lor;
# - În cod, comentarile, explicațiile și numele simbolurilor sunt în Engleza; explicațile din document sunt în Română;
# - Se vor aplica simplificări la modelele matematice conform teoriei grinzilor sandwich.
# 
# # Definirea elementelor de bază pentru calculele analitice
# 
# Pentru a simplifica restul aplicației vom începe prin a defini o serie de structuri care să modeleze conceptele din modelul analitic.
# Începem prin a defini structurile pentru parametrii de material, pentru geometria grinzii și pentru simularea statică.
# 
# Vom defini 2 structuri diferite materiale, una va modela parametrii materialelor de înveliș iar cealalta parametrii materialelor de miez.  
# Pentru materialele de înveliș avem nevoie de:
# - Densitate ($\rho$);
# - Modulul elastic longitudinal ($E_f$);
# - Tensiunea maximă admsibilă ($\tau_{af}$);
# - Costul materialului în €/Kg.
# 
# Pentru materialele de miez avem nevoie de:
# - Densitate ($\rho$);
# - Modulul elastic longitudinal ($E_f$);
# - Modulul elastic transversal ($G_c$)
# - Tensiunea maximă admsibilă ($\tau_{ac}$);
# - Costul materialului în €/Kg.

# %%
class SkinMaterialParameters:
    def __init__(self, rho:float, Ef:float, tau_af:float, Cost:float):
        self.rho = rho
        self.Ef = Ef # Longitudinal Elastic Modulus
        self.tau_af = tau_af # Persmissible Shear Stress
        self.Cost = Cost # euro/kg

class CoreMaterialParameters:
    def __init__(self, rho:float, Ec:float, Gc:float, tau_ac:float, Cost:float):
        self.rho = rho
        self.Ec = Ec # Longitudinal Elastic Modulus
        self.Gc = Gc # Transversal Elastic Modulus
        self.tau_ac = tau_ac # Permissible Shear Stress
        self.Cost = Cost

# %% [markdown]
# Pentru geometria grinzii avem nevoie de următorii parametrii:
# 
# - Materialul pentru înveliș;
# - Materialul pentru miez;
# - Lungimea ($L$);
# - Lățimea ($b$);
# - Grosimea miezului ($tc$);
# - Grosimea învelișului ($tf$);

# %%
class BeamModel:
    def __init__(self, L:float, b:float, tf:float, tc:float, skinMat : SkinMaterialParameters, coreMat : CoreMaterialParameters):
        self.L = L # Lenght
        self.b = b # Width
        self.tf = tf # Skin height
        self.tc = tc # Core height
        self.SkinMat = skinMat # Material for the skin
        self.CoreMat = coreMat # Material for the core

# %% [markdown]
# Ramâne să definim parametrii pentru simularea modelului analitic. Aici vom utiliza următoarele date:
# - Masa aplicată la capătul liber al grinzii ($m$);
# - Accelerația folosită pentru aplicarea condiției la limită ($a$);
# - Rigiditatea maximă admisă în grindă ($k_m$);
# - Modelul de grindă (instanță a clasei `BeamModel`);
# - Factorul de siguranță folosit în calculele tensiunilor ($s_f$).

# %%
class BeamSimulation:
    def __init__(self, m:float, a:float, km:float, model : BeamModel, sf:float):
        self.m = m
        self.a = a
        self.km = km
        self.model = model
        self.sf = sf

# %% [markdown]
# Cu aceste clase putem modela complet o simulare a unei grinzi. 
# 
# ## Definirea propietătilor de material admise
# 
# Adăugăm, conform specificaților impuse, propietățile de materiale.

# %%
from enum import Enum

class SkinMaterials(Enum):
    Steel = 0
    Aluminium = 1
    GFRP = 2
    CFRP = 3

class CoreMaterials(Enum):
    DivinycellH60 = 0
    DivinycellH100 = 1
    DivinycellH130 = 2
    DivinycellH200 = 3

skin_materials = {
    SkinMaterials.Steel: SkinMaterialParameters(7800, 205000 * 1e6, 300 * 1e6, 0.4),
    SkinMaterials.Aluminium: SkinMaterialParameters(2700, 70000 * 1e6, 200 * 1e6, 0.7),
    SkinMaterials.GFRP: SkinMaterialParameters(1600, 20000 * 1e6, 400 * 1e6, 4),
    SkinMaterials.CFRP: SkinMaterialParameters(2700, 70000 * 1e6, 1000 * 1e6, 80)
    }
core_materials = {
    CoreMaterials.DivinycellH60: CoreMaterialParameters(60, 550 * 1e6, 22 * 1e6, 0.6 * 1e6, 6),
    CoreMaterials.DivinycellH100: CoreMaterialParameters(100, 95 * 1e6, 38 * 1e6, 1.2 * 1e6, 10),
    CoreMaterials.DivinycellH130: CoreMaterialParameters(130, 125 * 1e6, 47 * 1e6, 1.6 * 1e6, 13),
    CoreMaterials.DivinycellH200: CoreMaterialParameters(200, 195 * 1e6, 75 * 1e6, 3.0 * 1e6, 20),
    }


# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go

labels = ['Material', 'Density <br> (kg/m^3)', 'E <br> (N/mm)', 'G <br> (N/mm)', 'Permisible <br> Shear Stress <br> (MPa)', 'Cost <br> (€/kg)']
values = []

materials_skin = []
costs_skin = []
for mat in skin_materials.items():
    values.append([
        mat[0].name,
        mat[1].rho,
        mat[1].Ef * 1e-6,
        '-',
        mat[1].tau_af * 1e-6,
        mat[1].Cost
    ])
    materials_skin.append(mat[0].name)
    costs_skin.append(mat[1].Cost * mat[1].rho)

materials_core = []
costs_core = []
for mat in core_materials.items():
    values.append([
        mat[0].name,
        mat[1].rho,
        mat[1].Ec * 1e-6,
        mat[1].Gc * 1e-6,
        round(mat[1].tau_ac * 1e-6, 2),
        mat[1].Cost
    ])
    materials_core.append(mat[0].name)
    costs_core.append(mat[1].Cost * mat[1].rho)
values = list(map(list, zip(*values)))


# %% [markdown]
# ## Calculele de simulare
# 
# Vom continua prin a implementa formulele de calcul din modelul matematic.  
# 
# - Forța aplicată: $P = m a$;
# - Deformarea maximă admisă: $W_m = \frac{P}{k_n}$;
# - Tensunea de forfecare maximă admisă în înveliș (cu safety factor): $\tau_{af} = \frac{\tau_{af}}{s_f}$;
# - Tensiunea de forfecare maximă admisă în miez (cu safety factor): $\tau_{ac} = \frac{\tau_{ac}}{s_f}$;
# - Volumul învelișului: $V_f = 2t_f \cdot L \cdot b$;
# - Volumul miezului: $V_c = t_c \cdot L \cdot b$;
# - Costul total: $Cost = V_f \cdot \rho_f \cdot Cost_f + V_c \cdot \rho_c \cdot Cost_c$;
# - Grosimea totală: $ d = 2t_f + t_c $;
# - Rigiditatea la încovoiere din grindă: $ D = \frac{1}{2}E_f \cdot t_f \cdot d^2 \cdot b$;
# - Rigiditatea la forfecare din grindă: $ S = \frac{1}{t_c} G_c \cdot d^2 $ ;
# - Deformarea reală: $ W = \frac{P \cdot L^3}{3D} + \frac{P \cdot L}{S} $;
# - Tensiunea maximă de forfecare din înveliș: $ \tau_f = \frac{P}{D} \frac{E_f}{2} \cdot (\frac{t_c}{2} + t_f)^2 - L^2 $;
# - Tensiunea maximă de forfecare din miez: $ \tau_c = \frac{P}{D} [ \frac{E_c}{2} (\frac{t_c}{2}^2 - L^2) + \frac{E_f}{2}(t_f \cdot t_c + t_f^2) ]$;
# - Masa totală: $ M_t = V_f \cdot \rho_f + V_c \cdot \rho_c $
# 
# 

# %%
class Result:
    def __init__(self, simulation):
        self.simulation = simulation

    def Solve(self):
        simulation = self.simulation

        # Gather all the input variables to make the formulas look nice
        m = simulation.m
        a = simulation.a
        sf = simulation.sf
        km = simulation.km
        tau_af = simulation.model.SkinMat.tau_af
        tau_ac = simulation.model.CoreMat.tau_ac
        L = simulation.model.L
        b = simulation.model.b
        tf = simulation.model.tf
        tc = simulation.model.tc
        Ef = simulation.model.SkinMat.Ef
        Ec = simulation.model.CoreMat.Ec
        Gc = simulation.model.CoreMat.Gc
        rho_f = simulation.model.SkinMat.rho
        rho_c = simulation.model.CoreMat.rho
        cost_f = simulation.model.SkinMat.Cost
        cost_c = simulation.model.CoreMat.Cost

        # Now solve all the equations described above and store in data memebers the values we're interested in
        P = m*a
        Wm = P/km
        tau_af = tau_af / sf
        tau_ac = tau_ac / sf
        Vf = 2*tf * L *b
        Vc = tc * L * b
        Cost = Vf * rho_f * cost_f + Vc * rho_c * cost_c
        d = 2*tf+tc
        D = (Ef * tf * d**2 / 2)*b
        S = (1/tc) * Gc * d**2
        W = (P*L**3)/(3*D) + (P*L)/S
        tau_f = (P/D)*(Ef/2)*(tc/2 + tf)**2
        tau_c = (P/D)*( Ec/2 * ((tc**2)/2) + Ef/2 * (tf*tc + tf**2) )
        Mt = Vf * rho_f + Vc * rho_c

        # Store the values we're interested in
        self.W = W
        self.Wm = Wm

        self.tau_af = tau_af
        self.tau_f = tau_f
        self.tau_ac = tau_ac
        self.tau_c = tau_c

        self.Cost = Cost
        self.Mt:float = Mt

# %% [markdown]
# Înainte de a continua cu explorarea posibilelor combinații vom rula o primă simulare cu valori aleator alese pe care vom intenționa să o și corelăm.
# 
# ## Corelarea
# 
# Vom alege să rulăm o simulare pentru o grindă cu următorii parametrii:
# - Lungime de 4 m;
# - Grosime miez de 70 cm;
# - Grosime înveliș de 1mm;
# - Înveliș din oțel și miez din Divinycell H60.
# 

# %%
tf = 1*1e-3   # shell of 1 mm
tc = 70*1e-3  # core of 7 cmm
L = 4         # can vary between 1 and 4 meters; let's go with 4 for now
b = 500*1e-3  # this is hardcoded and will remain at 500mm
skinMat = skin_materials[SkinMaterials.Steel]
coreMat = core_materials[CoreMaterials.DivinycellH60]

model = BeamModel(L, b, tf, tc, skinMat, coreMat)

m = 150    # kg - this is fixed
a = 9.834  # m/s^2 - we would need to have higher accelerations to account for jumps
km = 5000  # N/m - fixed
sf = 5     # if too expensive, make this smaller :-)

simulation = BeamSimulation(m, a, km, model, sf)

res = Result(simulation)
res.Solve()

labels = []
values = []

labels.append("Displacement")
values.append("{:.2f} mm".format(res.W * 1e3))

labels.append("Admissable Displacement")
values.append("{:.2f} mm".format(res.Wm * 1e3))

labels.append("Core Stress")
values.append("{:.2f} MPa".format(res.tau_c * 1e-6))

labels.append("Admissable Core Stress")
values.append("{:.2f} MPa".format(res.tau_ac * 1e-6))

labels.append("Shell Stress")
values.append("{:.2f} MPa".format(res.tau_f * 1e-6))

labels.append("Admissable Shell Stress")
values.append("{:.2f} MPa".format(res.tau_af * 1e-6))

labels.append("Mass")
values.append("{:.2f} kg".format(res.Mt))

labels.append("Cost")
values.append("{:.2f} €".format(res.Cost))


# %% [markdown]
# ### Modelul cu element finit
# 
# Pentru cazul mai sus ales vom crea un model cu element finit pentru a corela rezultatul obținut.  
# Partea de CAD este trivială pentru cazul de grindă, modelăm miezul ca un solid:
# ![miez](img/miez_cad.png)  
# Pentru înveliș vom face modelarea folosind 2 suprafețe:
# ![invelis](img/invelis_cad.png)  
# 
# La discretizare, deoarece vom folosi o soluție liniară, pentru a putea captura totuși reduce eroarea și a captura comportamentul parabolic al modelului optăm pentru elemtene parabolice TETRA în partea de solid și elemente parabolice TRIA în partea de suprafețe:  
# ![mesh](img/mesh.png)
# 
# În total avem aproximativ ~638k de elemente pentru miez și 80k de elemente pentru înveliș.  
# Materialul pentru înveliș îl definim ca oțel (isotropic) însă pentru miez vom defini materialul ca orthotropic cu module de elasticitate diferite pentru axa +X respectiv +Y. Axa +Z al materialului poate fi ignorată.  
# 
# Pentru modelul de simulare folosim Flex Glue între înveliș și miez, încastrăm complet unul din capete (doar pe înveliș) și punem o forță egală cu masa * accelerația la capătul liber:  
# ![sim](img/sim.png)
# 
# Folosim NASTRAN SOL101 - Static Linear pentru obținerea deformațiilor:
# 
# ![sim](img/displacement.png)
# 
# Un lucru interesant este distribuția stresului Von-Mises pentru această simulare:
# 
# ![stress](img/stress.png).
# 
# În concluzie, analiza cu element finit a rezultat în valori apropiate de cele calculate analitic: 133 mm față de 122 mm. Diferența poate fi explicată de aproximările făcute atât în analiza cu element finit cât și în modelul matematic. O simulare mai bună ar include modificarea condiților la limită astfel încât să acopere realist utilizarea unei astfel de trambuline (ex: încastrarea să fie făcută pe modul de prindere, forța distribuită pe o suprafață mai mare a grinzii etc).
# %% [markdown]
# ## Căutarea ghidată în spațiul de soluții
# 
# 

# %%
import pyswarms as ps
from copy import deepcopy
import math

def F_Cost_single(params):
    L = params[0]
    tf = params[1]
    tc = params[2]
    matSkin = params[3]
    matCore = params[4]

    sim = deepcopy(simulation) # allow parallel computation
    sim.model.L = L
    sim.model.b = b
    sim.model.tf = tf
    sim.model.tc = tc

    matSkinIdx = math.floor(matSkin)
    matCoreIdx = math.floor(matCore)

    sim.model.CoreMat = core_materials[CoreMaterials(matCoreIdx)]
    sim.model.SkinMat = skin_materials[SkinMaterials(matSkinIdx)]

    res = Result(sim)
    res.Solve()

    # we compute the total function cost as the total mass of the beam
    # however, for every constraint not met we multiply with the absolute difference of the constraint
    # so if the cost is 300 euroes (100 over the limit of 200) we multiply the mass with 100
    # this harshly penalisez the algorithm for going over the constraints
    # Warning: the function cost (or cost function) is different than the cost in euroes for the beam
    #   The cost function is used in the guided search algorithm

    fCost = res.Mt

    if res.Cost > 200:
        fCost *= (res.Cost / 200) * 100
    if res.W > res.Wm:
        fCost *= (res.W / res.Wm) * 100
    if res.tau_f > res.tau_af:
        fCost *= (res.tau_f / res.tau_af) * 100
    if res.tau_c > res.tau_ac:
        fCost *= (res.tau_c / res.tau_ac) * 100

    return fCost


def F_Cost(params):
    n_particles = params.shape[0]
    return [F_Cost_single(params[i]) for i in range(n_particles)]

swarm_size = 50
dim = 5
options = {'c1':1.5, 'c2':1.5, 'w':0.5}
constraints =  (
    (1, 0.1 * 1e-3,  10 * 1e-3, 0, 0),
    (4, 10  * 1e-3, 100 * 1e-3, len(SkinMaterials) - 0.01, len(CoreMaterials) - 0.01)
)

optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dim, options=options, bounds=constraints)

cost, joint_vars = optimizer.optimize(F_Cost, 10000, None)

print(cost)
print(joint_vars)


L = joint_vars[0]
tf = joint_vars[1]
tc = joint_vars[2]
matSkin = joint_vars[3]
matCore = joint_vars[4]

sim = deepcopy(simulation) # allow parallel computation
sim.model.L = L
sim.model.b = b
sim.model.tf = tf
sim.model.tc = tc

matSkinIdx = math.floor(matSkin)
matCoreIdx = math.floor(matCore)

sim.model.CoreMat = core_materials[CoreMaterials(matCoreIdx)]
sim.model.SkinMat = skin_materials[SkinMaterials(matSkinIdx)]

res = Result(sim)
res.Solve()

print("Displacement")
print(res.W)
print(res.Wm)
print("Core Stress")
print(res.tau_c)
print(res.tau_ac)
print("Shell Stress")
print(res.tau_f)
print(res.tau_af)
print("Cost")
print(res.Cost)
print("Mass")
print(res.Mt)


# %%


