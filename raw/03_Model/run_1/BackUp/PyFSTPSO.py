from fstpso import FuzzyPSO
from numpy import array, append, delete, genfromtxt, loadtxt, mean, std
from multiprocess import Pool
from string import ascii_letters as al
from random import sample
import os, sys, platform
from shutil import copy
from glob import glob
from subprocess import getstatusoutput as gso
import proplot as plt
from utils.hypoellipse import runHypoellipse
from utils.hypo71 import runHypo71
from shutil import rmtree

"""
Calculate 1D velocity model using FST-PSO method
for more ifo look at > https://github.com/aresio/fst-pso.

input files: 

    hypoellipse.pha, hypoellipse.sta, default.cfg 

outputs:

log.out which includes all updated models.
result.png which is statistics. 

ChangeLogs:

25-Jul-2018 > Initial.
03-Nov-2018 > Add multiprocessing capability.
06-Nov-2018 > Add option for plotting results only.
"""

#___________________Read input parameter file

if not os.path.exists("fstpsoVel.par"):
    with open("fstpsoVel.par", "w") as f:
        f.write("""#################################################
#
# Parameter file for FST-PSO program.
#
#################################################
#
VEL_MIN  = 4.1, 4.7, 5.7, 5.6                    # Lower bound for velocity model, LVZ is not allowed in hypo71.
VEL_MAX  = 4.5, 5.1, 6.1, 6.0                    # Upper bound for velocity model, LVZ is not allowed in hypo71.
DEP_MIN  = 0.0, 4.0, 8.0,12.0                    # Lower bound for Depth layering, LVZ is not allowed in hypo71.
DEP_MAX  = 0.1, 8.0,12.0,18.0                    # Upper bound for Depth layering, LVZ is not allowed in hypo71.
VpVS_R   = 1.95                                  # Vp/Vs ratio.
NUM_RUN  = 5                                     # Number of FST-PSO runs for estimating statistic errors.
NUM_MD   = 0                                     # Number of models. If Set to 0 then FST-PSO will do it automatically.
NUM_IT   = 0                                     # Number of iteration. If Set to 0 then FST-PSO will do it automatically.
MDTP     = -20.0                                 # Maximum depth to plot.
MP_FLAG  = 1                                     # Multiprocessing mode, 0:Disable, 1:Autoset, N define number of processors.
MEASURE  = 2                                     # Quality metrics, 1: RMS and 2:location accuracy.
M_FUNC   = 2                                     # Choose misfit function between 1:Hypoellipse or 2:Hypo71"""
                )
                
fst_pso_par = dict(genfromtxt("fstpsoVel.par", dtype=str, skip_header=6, delimiter="=", autostrip=True))
for key in fst_pso_par.copy(): fst_pso_par[key.strip()] = fst_pso_par.pop(key)

for key in list(fst_pso_par.keys()):
    fst_pso_par[key] = array(fst_pso_par[key].split(","), dtype=float)
            
#___________________Define main class
class Main():
    def __init__(self, plot_flag):
        self.run_id = 1
        self.vpvs = float(fst_pso_par["VpVS_R"])
        self.vel_min = fst_pso_par["VEL_MIN"]
        self.vel_max = fst_pso_par["VEL_MAX"]
        self.dep_min = fst_pso_par["DEP_MIN"]
        self.dep_max = fst_pso_par["DEP_MAX"]
        self.num_md = fst_pso_par["NUM_MD"]
        self.num_it = fst_pso_par["NUM_IT"]
        self.mdtp = fst_pso_par["MDTP"]
        self.measure = int(fst_pso_par["MEASURE"])
        self.misfitFunction = int(fst_pso_par["M_FUNC"])
        self.low_bnd = array([self.vel_min, self.vel_max]).T.tolist()
        self.upr_bnd = array([self.dep_min, self.dep_max]).T.tolist()
        self.search_space = self.low_bnd + self.upr_bnd
        self.resultDir = "results"
        self.bestModelOut = os.path.join(self.resultDir, "best_model.dat")
    
    def makeResultDir(self):
        if os.path.exists(self.resultDir):
            rmtree(self.resultDir)
        os.mkdir(self.resultDir)

    def write_hypoellipse_prm(self, x, vpvs, name):
        model = x.reshape((2, x.size//2)).T
        with open("{0}.prm".format(name), "w") as f:
            for l in model:
                vp, z = l[0], l[1]
                f.write("VELOCITY             {vp:4.1f} {z:5.0f} {vpvs:4.2f}\n".format(vp=vp, z=z, vpvs=vpvs))

    def write_fst_pso_result(self, result):
        with open(self.bestModelOut, "a") as f:
            f.write(" ".join("{0:7.2f}".format(e) for e in result[0].X))
            f.write("\n")

    def write_best_model(self, best_model, best_std):
        with open(self.bestModelOut, "a") as f:
            f.write("Best Model (Average of all models and StdDev):\n")
            f.write(" ".join("{0:7.2f}".format(e) for e in best_model))
            f.write("\n")
            f.write(" ".join("{0:7.2f}".format(e) for e in best_std))
            f.write("\n")
            
    def misfit(self, new_model):
        loc_id = "".join(sample(al, 5))
        vpvs = self.vpvs
        self.write_hypoellipse_prm(array(new_model), vpvs, loc_id)
        copy("hypoellipse.sta", "{0}.sta".format(loc_id))
        copy("hypoellipse.pha", "{0}.pha".format(loc_id))
        if self.misfitFunction == 1:
            meanRMS, meanERH, meanERZ = runHypoellipse(loc_id)
        elif self.misfitFunction == 2:
            meanRMS, meanERH, meanERZ = runHypo71(loc_id)
        cmd = "rm {0}*".format(loc_id); os.system(cmd)
        if self.measure == 1:
            return meanRMS
        elif self.measure == 2:
            return meanERH + meanERZ           

    def parallel_fitness_function(self, particles):
        N = int(fst_pso_par["MP_FLAG"])
        if N == 1: N = os.cpu_count()
        p = Pool(N)
        all_results = p.map(self.misfit, particles)
        p.close()
        return all_results

    def run_fst_pso(self):
        FP = FuzzyPSO(logfile=os.path.join(self.resultDir, "log.dat"))
        FP.disable_fuzzyrule_minvelocity()
        FP.set_search_space(self.search_space)
        best_fitness_out = os.path.join(self.resultDir, "best_fitness_{0:d}.dat".format(self.run_id+1))
        if self.num_md != 0:
            FP.set_swarm_size(self.num_md)
        if fst_pso_par["MP_FLAG"] != 0:
            FP.set_parallel_fitness(self.parallel_fitness_function, skip_test=True)
            if self.num_it != 0:
                result = FP.solve_with_fstpso(max_iter=self.num_it, dump_best_fitness=best_fitness_out)
            else:
                result = FP.solve_with_fstpso(dump_best_fitness=best_fitness_out)
        else:
            FP.set_fitness(self.misfit, skip_test=True)
            if self.num_it != 0:
                result = FP.solve_with_fstpso(max_iter=self.num_it, dump_best_fitness=best_fitness_out)
            else:
                result = FP.solve_with_fstpso(dump_best_fitness=best_fitness_out)
        self.write_fst_pso_result(result)

    def plot_results(self):
        if self.measure == 1:
            self.ylabel = "RMS (s)"
        elif self.measure == 2:
            self.ylabel = "Hypocentral error (km)"

        with plt.rc.context(abc="a)"):
            fig, axs = plt.subplots(ncols=2, share=0)
        
        best_fits = [loadtxt(best_fitness) for best_fitness in glob(os.path.join(self.resultDir, "best_fitness_*"))]
        try:
            best_model = mean(loadtxt(self.bestModelOut), axis=0)
        except ValueError:
            print("\n+++ Remove lines 3,4,5 from 'file:results/best_model.dat' and run again.")
            sys.exit()
        bm_std = std(loadtxt(self.bestModelOut), axis=0)
        self.write_best_model(best_model, bm_std)
        self.write_hypoellipse_prm(best_model, self.vpvs, os.path.join(self.resultDir, "Final"))

        ax = axs[0]
        ax.format(title="Loss")
        ax.set_facecolor("#fafafa")
        ax.grid(True, linestyle=":", linewidth=.5, color="k", alpha=.3)
        for best_fit in best_fits: ax.plot(range(1, best_fit.size+1), best_fit)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(self.ylabel)
        
        ax = axs[1]
        ax.format(title="Velocity Model")
        ax.set_facecolor("#fafafa")
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Depth (km)")
        ax.grid(True, linestyle=":", linewidth=.5, color="k", alpha=.3)
        vel_l = array([(i,j) for i,j in zip(self.vel_min,self.vel_min)]).flatten()
        vel_u = array([(i,j) for i,j in zip(self.vel_max,self.vel_max)]).flatten()
        dep_u = array([(i,j) for i,j in zip(self.dep_min,self.dep_min)]).flatten()
        dep_l = array([(i,j) for i,j in zip(self.dep_max,self.dep_max)]).flatten()
        dep_l = delete(dep_l,0,0)
        dep_u = delete(dep_u,0,0)
        dep_l = append(dep_l, self.mdtp)
        dep_u = append(dep_u, self.mdtp)
        ax.plot(vel_l, dep_l, linestyle=":", color="k", label="Lower band")
        ax.plot(vel_u, dep_u, linestyle="--", color="k", label="Upper band")
        final_model = best_model.reshape((2, best_model.size//2)).T
        finvel = array([(i,j) for i,j in zip(final_model[:,0],final_model[:,0])]).flatten()
        findep = array([(i,j) for i,j in zip(final_model[:,1],final_model[:,1])]).flatten()
        findep = delete(findep,0,0)
        findep = append(findep, self.mdtp)
        ax.plot(finvel, findep, color="r", label="Best model")
        c=0
        for i in range(finvel.size//2):
            ax.fill_between([finvel[c]-bm_std[i], finvel[c]+bm_std[i]], findep[c+1], findep[c], color="r", alpha=.2)
            c+=2
        c=0
        for j in range(finvel.size//2,finvel.size-1):
            ax.fill_between([finvel[c+1], finvel[c+2]], findep[c+1]-bm_std[j], findep[c+1]+bm_std[j], color="r", alpha=.2)
            c+=2
        # ax.invert_xaxis()
        ax.invert_yaxis()
        ax.legend(loc=3, ncols=1)
        fig.save(os.path.join(self.resultDir, "fst-pso_result.png"), bbox_inches="tight")
        plt.close()

#___________________Run
if __name__ == "__main__":
    ans = input("\n++++ New run [n] or Only plot results [p]:\n\n")
    if ans.upper() == "N":
        instance = Main(plot_flag=False)
        instance.makeResultDir()
        for i in range(int(fst_pso_par["NUM_RUN"])):
            instance.run_id = i
            instance.run_fst_pso()
        instance.plot_results()
        sys.exit()
    elif ans == "p":
        instance = Main(plot_flag=True)
        instance.plot_results()
        sys.exit()
    else:
        print("+++ wrong choice!")
