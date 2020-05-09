# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/W-Sebastian/JupyterNotebooks/blob/master/Structuri_U%C8%99oare_Grind%C4%83.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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
    def __init__(self, rho, Ef, tau_af, Cost):
        self.rho = rho
        self.Ef = Ef # Longitudinal Elastic Modulus
        self.tau_af = tau_af # Persmissible Shear Stress
        self.Cost = Cost # euro/kg

class CoreMaterialParameters:
    def __init__(self, rho, Ec, Gc, tau_ac, Cost):
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
    def __init__(self, L, b, tf, tc, skinMat : SkinMaterialParameters, coreMat : CoreMaterialParameters):
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
    def __init__(self, m, a, km, model : BeamModel, sf):
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
    Steel = 1
    Aluminium = 2
    GFRP = 3
    CFRP = 4

class CoreMaterials(Enum):
    DivinycellH60 = 1
    DivinycellH100 = 2
    DivinycellH130 = 3
    DivinycellH200 = 4

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

fig = go.Figure(data=[ go.Table(
    header=dict(
        values=labels,
        font=dict(size=10),
        align="center"
        ), 
    cells=dict(
        values=values,
        align="left"
        ))])
fig.show()

fig = make_subplots(rows=1, cols=2, subplot_titles=("Core Materials (cost per m^3)", "Skin Materials (cost per m^3)"))

fig.add_trace(go.Bar(x=materials_core, y=costs_core), row=1, col=1)
fig.add_trace(go.Bar(x=materials_skin, y=costs_skin), row=1, col=2)
fig.update_yaxes(row=1, col=1, ticksuffix='€')
fig.update_yaxes(row=1, col=2, ticksuffix='€')
fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)
fig.show()

# %% [markdown]
# ## Calculele de simulare
# 
# Vom continua prin a implementa formulele de calcul din modelul matematic.  
# 
# - Forța aplicată: $P = m a$;
# - Deformarea maximă admisă: $W_m = \frac{P}{k_n}$;
# - Tensunea de forfecare maximă admisă în înveliș (cu safety factor): $\tau_{af} = \tau_{af} \cdot s_f$;
# - Tensiunea de forfecare maximă admisă în miez (cu safety factor): $\tau_{ac} = \tau_{ac} \cdot s_c$;
# - Volumul învelișului: $V_f = 2t_f \cdot L \cdot b$;
# - Volumul miezului: $V_c = t_c \cdot L \cdot b$;
# - Costul total: $Cost = V_f \cdot \rho_f \cdot Cost_f + V_c \cdot \rho_c \cdot Cost_c$;
# - Grosimea totală: $ d = 2t_f + t_c $;
# - Rigiditatea la încovoiere din grindă: $ D = \frac{1}{2b}E_f \cdot t_f \cdot d^2 $;
# - Rigiditatea la forfecare din grindă: $ S = \frac{1}{t_c} G_c \cdot d^2 $ ;
# - Deformarea reală: $ W = \frac{P \cdot L^3}{3D} + \frac{P \cdot L}{S} $;
# - Tensiunea maximă de forfecare din înveliș: $ \tau_f = \frac{P}{D} \frac{E_f}{2} \cdot (\frac{t_c}{2} + t_f)^2 - L^2 $;
# - Tensiunea maximă de forfecare din miez: $ \tau_c = \frac{P}{D} [ \frac{E_c}{2} (\frac{t_c}{2}^2 - L^2) + \frac{E_f}{2}(t_f \cdot t_c + t_f^2) ]$;
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
        tau_af = tau_af * sf
        tau_ac = tau_ac * sf
        Vf = 2*tf * L *b
        Vc = tc * L * b
        Cost = Vf * rho_f * cost_f + Vc * rho_c * cost_c
        d = 2*tf+tc
        D = 1/(2*b) *Ef * tf * d**2
        S = (1/tc) * Gc * d**2
        W = (P*L**3)/(3*D) + (P*L)/S
        tau_f = (P/D)*(Ef/2)*(tc/2 + tf)**2
        tau_c = (P/D)*( Ec/2 * ((tc**2)/2) + Ef/2 * (tf*tc + tf**2) )

        # Store the values we're interested in
        self.W = W
        self.Wm = Wm

        self.tau_af = tau_af
        self.tau_f = tau_f
        self.tau_ac = tau_ac
        self.tau_c = tau_c

        self.Cost = Cost


tf = 1*1e-3   # shell of 1 mm
tc = 70*1e-3  # core of 7 cmm
L = 4         # can vary between 1 and 4 meters; let's go with 4 for now
b = 500*1e-3  # this is hardcoded and will remain at 500mm
skinMat = skin_materials[SkinMaterials.Steel]
coreMat = core_materials[CoreMaterials.DivinycellH60]

model = BeamModel(L, b, tf, tc, skinMat, coreMat)

m = 150                # kg - this is fixed
a = 9.834      # m/s^2 - we would need to have higher accelerations to account for jumps
km = 5000  # N/m - fixed
sf = 5          # if too expensive, make this smaller :-)

simulation = BeamSimulation(m, a, km, model, sf)

res = Result(simulation)
res.Solve()
print(res.W * 1e3)
print(res.Wm * 1e3)

print(res.tau_f * 1e-6)
print(res.tau_af * 1e-6)

print(res.tau_c * 1e-6)
print(res.tau_ac * 1e-6)



# %%


