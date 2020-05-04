import logging
import yank
import os
import simtk.openmm as mm
from simtk.openmm import unit, version 
from simtk.openmm.app import PDBFile, Modeller, ForceField, PME, Simulation, DCDReporter, StateDataReporter, HBonds, metadynamics
import mdtraj as md
from query_klifs import query_klifs_database
import protein
import klifs
import copy

# output logs
logger = logging.getLogger(__name__)
logging.root.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
yank.utils.config_root_logger(verbose=True)

# set up basic parameters
pdbid = '2JIU'
chain = 'B'
experiment = '2_dimensional'
work_dir = f'/home/guoj1/data_projects/cv_selection/metadynamics/{experiment}/{pdbid}'
temperature = 310.15 * unit.kelvin
pressure = 1.0 * unit.atmospheres
cv_min_1 = -19.44
cv_max_1 = -0.82
cv_std_1 = 4.24
cv_min_2 = 0.48
cv_max_2 = 13.63
cv_std_2 = 2.30

# load pdb to Modeller
pdb = PDBFile(f'{pdbid}_chain{chain}_minequi.pdb')
molecule = Modeller(pdb.topology, pdb.positions)
print("Done loading pdb to Modeller.")
# load force field
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
print("Done loading force field.")
print("OpenMM version:", version.version)

# prepare the system (using heavy hydrogens and constrain all hygrogen atom-involved bonds)
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, rigidWater=True, nonbondedCutoff=1*unit.nanometer, hydrogenMass=4*unit.amu, constraints = HBonds)


# specify the set of key atoms and calculate key dihedrals and distances
traj = md.load_pdb(f'{pdbid}_chain{chain}_minequi.pdb')
klifs = query_klifs_database(pdbid, chain)
key_res = protein.key_klifs_residues(klifs.numbering)
(dih, dis) = protein.compute_simple_protein_features(traj, key_res)

# formalize unit
kT_md_units = (unit.MOLAR_GAS_CONSTANT_R * temperature).value_in_unit_system(unit.md_unit_system)

# add the custom cv force
# dihedrals
feat_dict = dict()

# feature 0 - 61: dihedral features 
    # 0 - 41: A-loop backbone dihedrals (Phi, Psi)
    # 42 - 53: P-loop backbone dihedrals (Phi, Psi)
    # 54: angle between aC and aE helices
    # 55 - 61: Dunbrack dihedrals
for i in range(62):
    feat_dict[i] = mm.CustomTorsionForce("theta")
    feat_dict[i].addTorsion(int(dih[i][0]), int(dih[i][1]), int(dih[i][2]), int(dih[i][3]))

# feature 62 - 68: distance features
for i in range(62, 69):
    feat_dict[i] = mm.CustomBondForce("r")
    feat_dict[i].addBond(int(dis[i-62][0]), int(dis[i-62][1]))

print("Done populating dihedrals and distances.")

# add weights of different collective variables for each reaciton coordinate
w1 = dict() 
w2 = dict()

sel_feat = list([66, 65, 62, 68, 19, 67, 63, 64]) # these are 1-based indices
weights1 = list([0.009626, -0.045500, -0.521299, 0.025095, -0.563564, -0.457624, -0.393545, 0.208691])
weights2 = list([-0.002846, 0.023837, 0.535975, 0.777245, 0.040945, 0.1803, -0.247547, 0.112163])
for n in range(8):
    w1[sel_feat[n]] = weights1[n] 
    w2[sel_feat[n]] = weights2[n]
    

# Specify a unique CustomCVForce
expression_1 = str()
for i in range(len(sel_feat) - 1):
    expression_1 += f'({w1[sel_feat[i]]} * feat_{i}) + '
expression_1 += f'({w1[sel_feat[i+1]]} * feat_{i+1})'
cv_force_1 = mm.CustomCVForce(expression_1)
for i in range(len(sel_feat)):
    print(w1[sel_feat[i]])
    print(feat_dict[sel_feat[i] - 1])
    cv_force_1.addCollectiveVariable(f'feat_{i}', feat_dict[sel_feat[i] - 1])
bv_1 = metadynamics.BiasVariable(cv_force_1, cv_min_1, cv_max_1, cv_std_1, periodic=False)

expression_2 = str()
for i in range(len(sel_feat) - 1):
    expression_2 += f'({w2[sel_feat[i]]} * feat_{i}) + '
expression_2 += f'({w2[sel_feat[i+1]]} * feat_{i+1})'
cv_force_2 = copy.deepcopy(cv_force_1)
cv_force_2.setEnergyFunction(expression_2)
bv_2 = metadynamics.BiasVariable(cv_force_2, cv_min_2, cv_max_2, cv_std_2, periodic=False)
print("Done defining forces.")


# Set up the context for mtd simulation
# at this step the CV and the system are separately passed to Metadynamics
meta = metadynamics.Metadynamics(system, variables=[bv_1, bv_2], temperature=temperature, biasFactor=10.0, height=1.5*unit.kilojoules_per_mole, frequency=250, saveFrequency=250, biasDir='./biases')
integrator = mm.LangevinIntegrator(temperature, 1.0/unit.picosecond, 0.004*unit.picoseconds)
print("Done specifying integrator.")
simulation = Simulation(molecule.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
print("Done specifying simulation.")

# equilibration
simulation.context.setVelocitiesToTemperature(temperature)
simulation.step(100)
print("Done 100 steps of equilibration.")

# set simulation reporters
simulation.reporters.append(DCDReporter('mtd_2JIU.dcd', reportInterval=250))
simulation.reporters.append(StateDataReporter('mtd_2JIU.out', reportInterval=5000, step=True,
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True, 
    speed=True, totalSteps=10000000, separator='\t'))

# Run small-scale simulation (20ns, 10^7 steps) and plot the free energy landscape
meta.step(simulation, 10000000)
#plot.imshow(meta.getFreeEnergy())
#plot.show()
print("Done with 10^7 steps of production run.")

