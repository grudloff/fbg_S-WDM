import numpy as np
import matplotlib.pyplot as plt
import fbg_swdm.visualization as vis
import fbg_swdm.simulation as sim
import fbg_swdm.evolutionary as ev
import fbg_swdm.variables as vars
vars.set_base_dir('C:\\Users\\parac\\Desktop\\Gabriel\\Experimentos')

ev.clone_module(vars)
ev.vars.setattrs(
    exp_name = 'baseline',
    Q = Q,
    A = np.array([1]*Q),
    Δλ = np.array([0.2*vars.n]*Q),
    I = np.array([0.5, 0.9]),
    N = 2000,
    topology = 'serial',
    multiprocessing=True
)

Q = 2
vars.setattrs(
    exp_name = 'baseline_attenuation',
    Q = Q,
    A = np.array([1, 0.8]),
    Δλ = np.array([0.2*vars.n]*Q),
    I = np.array([0.5, 0.9]),
    N = 2000,
    topology = 'serial'
)

if __name__ == '__main__':
    import time
    model_dict = {
        'genetic_binary' : ev.genetic_algorithm_binary(),
        'genetic_real' : ev.genetic_algorithm_real(pop_size=150, max_generation=500, std=1*vars.n, patience=50, swap=True),
        'swap_diff' : ev.swap_differential_evolution(),
        'distributed_estimation' : ev.DistributedEstimation(pop_size=200, top_size=50, patience=50),
        'DMS_particle_swarm' : ev.dynamic_multi_swarm_particle_swarm_optimization(vel_init='gaussian', max_plateau=50)
    }

    for model_name, model in model_dict.items():
        vars.setattrs(tag = model_name)
        start = time.perf_counter()
        vis.plot_sweep(model, norm=False, rec_error=True)
        finish = time.perf_counter()
        vars.log(f'Finished sweep in {round(finish-start, 2)} seconds')

        start = time.perf_counter()
        vis.plot_sweep(model, norm=False, rec_error=True, noise=1e-2)
        finish = time.perf_counter()
        vars.log(f'Finished sweep with noise in {round(finish-start, 2)} seconds')

        start = time.perf_counter()
        vis.error_snr(model, norm=False, min_snr= 0, max_snr=40, M=9)
        finish = time.perf_counter()
        vars.log(f'Finished SNR curve in {round(finish-start, 2)} seconds')