from enum import Enum

import pyswarms as ps
from copy import deepcopy
import math

from multiprocessing import Process, freeze_support

from pyswarm import pso

import numpy as np
from scipy.optimize import minimize


class SkinMaterialParameters:
    def __init__(self, rho: float, Ef: float, sigma_af: float, Cost: float):
        self.rho = rho
        self.Ef = Ef  # Longitudinal Elastic Modulus
        self.sigma_af = sigma_af  # Permissible Stress
        self.Cost = Cost  # euro/kg


class CoreMaterialParameters:
    def __init__(self, rho: float, Ec: float, Gc: float, tau_ac: float, Cost: float):
        self.rho = rho
        self.Ec = Ec  # Longitudinal Elastic Modulus
        self.Gc = Gc  # Transversal Elastic Modulus
        self.tau_ac = tau_ac  # Permissible Shear Stress
        self.Cost = Cost


class BeamModel:
    def __init__(self, L: float, b: float, tf: float, tc: float, skinMat: SkinMaterialParameters,
                 coreMat: CoreMaterialParameters):
        self.L = L  # Length
        self.b = b  # Width
        self.tf = tf  # Skin height
        self.tc = tc  # Core height
        self.SkinMat = skinMat  # Material for the skin
        self.CoreMat = coreMat  # Material for the core


class BeamSimulation:
    def __init__(self, m: float, a: float, km: float, model: BeamModel, sf: float):
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


# noinspection DuplicatedCode,PyAttributeOutsideInit
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
        sigma_af = simulation.model.SkinMat.sigma_af
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

        # Now solve all the equations described above and store in data members the values we're interested in
        # Actual Load (Force here)
        P = m * a

        # Moment over the length of the beam
        M = P * L

        # Imposed displacement based on the rigidity
        Wm = P / km

        # Maximum stress
        sigma_af = sigma_af / sf
        tau_ac = tau_ac / sf

        # Volumes
        Vf = 2 * tf * L * b
        Vc = tc * L * b

        # Total cost
        Cost = Vf * rho_f * cost_f + Vc * rho_c * cost_c

        # Distance between middle of shell to middle of shell
        d = tf + tc

        D = (Ef * tf * d ** 2 / 2) * b
        S = (1 / tc) * Gc * d ** 2 * b
        W = (P * L ** 3) / (3 * D) + (P * L) / S

        # Actual shell stress
        sigma_f = M / D * Ef * d / 2

        # Actual core sheer stress
        tau_c = (P / D) * (Ec / 2 * ((tc ** 2) / 2) + Ef / 2 * (tf * tc + tf ** 2))

        # Total Mass
        Mt = Vf * rho_f + Vc * rho_c

        # Store the values we're interested in
        self.W = W
        self.Wm = Wm

        self.sigma_af = sigma_af
        self.sigma_f = sigma_f

        self.tau_ac = tau_ac
        self.tau_c = tau_c

        self.Cost = Cost
        self.Mt = Mt


# noinspection DuplicatedCode
def F_Cost_single(params, simulation):
    L = params[0]
    tf = params[1]
    tc = params[2]

    sim = deepcopy(simulation)  # allow parallel computation
    sim.model.L = L
    sim.model.tf = tf
    sim.model.tc = tc

    res = Result(sim)
    res.Solve()

    # we compute the total function cost as the total mass of the beam
    # however, for every constraint not met we multiply with the absolute difference of the constraint
    # so if the cost is 300 euros (100 over the limit of 200) we multiply the mass with 100
    # this harshly penalises the algorithm for going over the constraints
    # Warning: the function cost (or cost function) is different than the cost in euros for the beam
    #   The cost function is used in the guided search algorithm

    w_constraint = abs(res.W - res.Wm) * 1e3

    cost_constraint = max(res.Cost - 200, 1)
    sigma_constraint = max(res.sigma_f - res.sigma_af, 1)
    tau_constraint = max(res.tau_c - res.tau_ac, 1)

    fCost = res.Cost * w_constraint

    if cost_constraint > 1:
        fCost *= cost_constraint * 1e6
    if sigma_constraint > 1:
        fCost *= sigma_constraint * 1e6
    if tau_constraint > 1:
        fCost *= tau_constraint * 1e6

    return fCost


def F_Cost(params, simulation):
    n_particles = params.shape[0]
    return [F_Cost_single(params[i], simulation) for i in range(n_particles)]


class SimulationResult:
    
    def __init__(self, skin_mat, core_mat, L, tf, tc):
        self.tc = tc
        self.tf = tf
        self.L = L
        self.core_mat = core_mat
        self.skin_mat = skin_mat

def pyswarms_old():
    freeze_support()

    tf = 1 * 1e-3  # shell of 1 mm
    tc = 70 * 1e-3  # core of 7 cmm
    L = 4  # can vary between 1 and 4 meters; let's go with 4 for now
    b = 500 * 1e-3  # this is hardcoded and will remain at 500mm
    skinMat = skin_materials[SkinMaterials.Steel]
    coreMat = core_materials[CoreMaterials.DivinycellH60]

    model = BeamModel(L, b, tf, tc, skinMat, coreMat)

    m = 150  # kg - this is fixed
    a = 9.80665  # m/s^2 - we would need to have higher accelerations to account for jumps
    km = 5000  # N/m - fixed
    sf = 5  # if too expensive, make this smaller :-)

    simulation = BeamSimulation(m, a, km, model, sf)

    swarm_size = 500
    dim = 3

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.4}
    # options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 30, 'p':2}
    # options = {'c1': 1.5, 'c2': 0.9, 'w': 0.4} # good
    # options = {'c1': 2, 'c2': 2, 'w': 0.4}

    constraints = (
        (1, 0.1 * 1e-3, 10 * 1e-3),
        (4, 10 * 1e-3, 500 * 1e-3)
    )

    results = []

    for skin_mat in skin_materials.keys():
        for core_mat in core_materials.keys():

            used_core_mat = core_mat
            used_skin_mat = skin_mat

            simulation.model.CoreMat = core_materials[used_core_mat]
            simulation.model.SkinMat = skin_materials[used_skin_mat]

            optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size, dimensions=dim, options=options, bounds=constraints)
            cost, joint_vars = optimizer.optimize(F_Cost, 1000, 12, simulation=simulation)

            L = joint_vars[0]
            tf = joint_vars[1]
            tc = joint_vars[2]
            matSkin = simulation.model.CoreMat
            matCore = simulation.model.CoreMat

            result = SimulationResult(used_skin_mat, used_core_mat, L, tf, tc)
            results.append(result)


            simulation.model.L = result.L
            simulation.model.tf = result.tf
            simulation.model.tc = result.tc
            res = Result(simulation)
            res.Solve()

            print("Core Mat: {:>10}; Skin Mat: {:>10}; L: {:.5f}; tc: {:.5f}; tf: {:.5f}".format(
                result.core_mat.name,
                result.skin_mat.name,
                result.L,
                result.tc * 1e3,
                result.tf * 1e3
            ))
            print("W: {:.5f}=={:.5f}; CoreStress: {:.5f} < {:.5f}; ShellStress: {:.5f} < {:.5f}; Cost: {:.5f}; Mass: {:.5f}".format(
                    res.W * 1e3, res.Wm * 1e3,
                    res.tau_c * 1e-6, res.tau_ac * 1e-6,
                    res.sigma_f * 1e-6, res.sigma_af * 1e-6,
                    res.Cost, res.Mt
                ))


    for result in results:
        simulation.model.L = result.L
        simulation.model.tf = result.tf
        simulation.model.tc = result.tc
        simulation.model.CoreMat = core_materials[result.core_mat]
        simulation.model.SkinMat = skin_materials[result.skin_mat]
        res = Result(simulation)
        res.Solve()

        print("Core Mat: {:>10}; Skin Mat: {:>10}; L: {:.5f}; tc: {:.5f}; tf: {:.5f}".format(
            result.core_mat.name,
            result.skin_mat.name,
            result.L,
            result.tc * 1e3,
            result.tf * 1e3
        ))
        print("W: {:.5f}=={:.5f}; CoreStress: {:.5f} < {:.5f}; ShellStress: {:.5f} < {:.5f}; Cost: {:.5f}; Mass: {:.5f}".format(
            res.W * 1e3, res.Wm * 1e3,
            res.tau_c * 1e-6, res.tau_ac * 1e-6,
            res.sigma_f * 1e-6, res.sigma_af * 1e-6,
            res.Cost, res.Mt
        ))




def po_cost(params, *args):
    simulation, matSkin, matCore = args
    L, tf, tc = params

    sim = deepcopy(simulation)  # allow parallel computation
    sim.model.L = L
    sim.model.tf = tf
    sim.model.tc = tc

    sim.model.CoreMat = core_materials[matCore]
    sim.model.SkinMat = skin_materials[matSkin]

    res = Result(sim)
    res.Solve()

    return res.Mt

def po_constraints(params, *args):
    simulation, matSkin, matCore = args
    L, tf, tc = params

    sim = deepcopy(simulation)  # allow parallel computation
    sim.model.L = L
    sim.model.tf = tf
    sim.model.tc = tc

    sim.model.CoreMat = core_materials[matCore]
    sim.model.SkinMat = skin_materials[matSkin]

    res = Result(sim)
    res.Solve()

    w_constraint = abs(res.W - res.Wm)
    if w_constraint > 1e-2:
        w_constraint *= -1

    cost_constraint = 200 - res.Cost
    sigma_constraint = res.sigma_af - res.sigma_f
    tau_constraint = res.tau_ac - res.tau_c

    return [w_constraint, cost_constraint, sigma_constraint, tau_constraint]


def pyswarm_new():

    tf = 1 * 1e-3  # shell of 1 mm
    tc = 70 * 1e-3  # core of 7 cmm
    L = 4  # can vary between 1 and 4 meters; let's go with 4 for now
    b = 500 * 1e-3  # this is hardcoded and will remain at 500mm
    skinMat = skin_materials[SkinMaterials.Steel]
    coreMat = core_materials[CoreMaterials.DivinycellH60]

    model = BeamModel(L, b, tf, tc, skinMat, coreMat)

    m = 150  # kg - this is fixed
    a = 9.80665  # m/s^2 - we would need to have higher accelerations to account for jumps
    km = 5000  # N/m - fixed
    sf = 5  # if too expensive, make this smaller :-)

    simulation = BeamSimulation(m, a, km, model, sf)

    lb = [1, 0.1 * 1e-3, 10 * 1e-3]
    ub = [4, 10 * 1e-3, 500 * 1e-3]

    matSkin = SkinMaterials.CFRP
    matCore = CoreMaterials.DivinycellH60

    xopt, fopt = pso(po_cost, lb, ub, f_ieqcons=po_constraints,
                     args=(simulation, matSkin, matCore),
                     swarmsize=500, maxiter=500)

    L = xopt[0]
    tf = xopt[1]
    tc = xopt[2]

    sim = deepcopy(simulation)  # allow parallel computation
    sim.model.L = L
    sim.model.b = b
    sim.model.tf = tf
    sim.model.tc = tc

    sim.model.CoreMat = core_materials[matCore]
    sim.model.SkinMat = skin_materials[matSkin]

    res = Result(sim)
    res.Solve()

    print("L = {:.5f} m".format(L))
    print("tf = {:.5f} mm".format(tf * 1e3))
    print("tc = {:.1f} mm".format(tc * 1e3))

    print("Displacement = {:.5f} mm".format(res.W * 1e3))
    print("Admissible Displacement = {:.5f} mm".format(res.Wm * 1e3))

    print("Core Sheer Stress = {:.5f} MPa".format(res.tau_c * 1e-6))
    print("Admissible Core Sheer Stress = {:.5f} MPa".format(res.tau_ac * 1e-6))

    print("Shell Stress = {:.5f} MPa".format(res.sigma_f * 1e-6))
    print("Admissible Shell Stress = {:.5f} MPa".format(res.sigma_af * 1e-6))

    print("Cost = {:.2f} €".format(res.Cost))
    print("Mass = {:.5f} Kg".format(res.Mt))

    print("[{}. {} * 1e-3, {} * 1e-3, SkinMaterials.{}, CoreMaterials.{}]".format(L, tf, tc, matSkin.name, matCore.name))


if __name__ == '__main__':
    pyswarms_old()
