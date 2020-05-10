
from enum import Enum

import pyswarms as ps
from copy import deepcopy
import math

from multiprocessing import Process, freeze_support

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

class BeamModel:
    def __init__(self, L:float, b:float, tf:float, tc:float, skinMat : SkinMaterialParameters, coreMat : CoreMaterialParameters):
        self.L = L # Lenght
        self.b = b # Width
        self.tf = tf # Skin height
        self.tc = tc # Core height
        self.SkinMat = skinMat # Material for the skin
        self.CoreMat = coreMat # Material for the core

class BeamSimulation:
    def __init__(self, m:float, a:float, km:float, model : BeamModel, sf:float):
        self.m = m
        self.a = a
        self.km = km
        self.model = model
        self.sf = sf

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
    SkinMaterials.CFRP: SkinMaterialParameters(1500, 70000 * 1e6, 1000 * 1e6, 80)
    }
core_materials = {
    CoreMaterials.DivinycellH60: CoreMaterialParameters(60, 55 * 1e6, 22 * 1e6, 0.6 * 1e6, 6),
    CoreMaterials.DivinycellH100: CoreMaterialParameters(100, 95 * 1e6, 38 * 1e6, 1.2 * 1e6, 10),
    CoreMaterials.DivinycellH130: CoreMaterialParameters(130, 125 * 1e6, 47 * 1e6, 1.6 * 1e6, 13),
    CoreMaterials.DivinycellH200: CoreMaterialParameters(200, 195 * 1e6, 75 * 1e6, 3.0 * 1e6, 20),
    }


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

def F_Cost_single(params, simulation):
    L = params[0]
    tf = params[1]
    tc = params[2]
    matSkin = params[3]
    matCore = params[4]

    sim = deepcopy(simulation) # allow parallel computation
    sim.model.L = L
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

    fCost = 1/res.Mt

    # if res.Cost > 200:
    #    fCost *= (res.Cost / 200) * 100
    if res.W > res.Wm:
        fCost *= (res.W / res.Wm) * 100
    if res.tau_f > res.tau_af:
        fCost *= (res.tau_f / res.tau_af) * 100
    if res.tau_c > res.tau_ac:
        fCost *= (res.tau_c / res.tau_ac) * 100

    return fCost


def F_Cost(params, simulation):
    n_particles = params.shape[0]
    return [F_Cost_single(params[i], simulation) for i in range(n_particles)]

from pyswarms.utils.plotters.formatters import Mesher
# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    freeze_support()

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

    swarm_size = 1000
    dim = 5
    
    #options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
    #options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 30, 'p':2}
    options = {'c1': 1.5, 'c2': 0.9, 'w': 0.4}

    constraints =  (
        (1, 0.1 * 1e-3,  10 * 1e-3, 0, 0),
        (4, 10  * 1e-3, 100 * 1e-3, len(SkinMaterials) - 0.01, len(CoreMaterials) - 0.01)
    )

    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dim, options=options, bounds=constraints)

    cost, joint_vars = optimizer.optimize(F_Cost, 5000, 12, simulation=simulation)

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

    print(CoreMaterials(matCoreIdx).name)
    print(SkinMaterials(matSkinIdx).name)

    print("L = {:.5f} m".format(L))
    print("tf = {:.5f} mm".format(tf * 1e3))
    print("tc = {:.1f} mm".format(tc * 1e3))

    print("Displacement = {:.5f} mm".format(res.W * 1e3))
    print("Admissible Displacement = {:.5f} mm".format(res.Wm * 1e3))

    print("Core Stress = {:.5f} MPa".format(res.tau_c * 1e-6))
    print("Admissible Core Stress = {:.5f} MPa".format(res.tau_ac * 1e-6))
    
    print("Shell Stress = {:.5f} MPa".format(res.tau_f * 1e-6))
    print("Admissible Shell Stress = {:.5f} MPa".format(res.tau_af * 1e-6))

    print("Cost = {:.2f} €".format(res.Cost))
    print("Mass = {:.5f} Kg".format(res.Mt))