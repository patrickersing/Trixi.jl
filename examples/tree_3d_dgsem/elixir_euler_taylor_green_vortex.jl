using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

"""
    initial_condition_taylor_green_vortex(x, t, equations::CompressibleEulerEquations3D)

The classical inviscid Taylor-Green vortex.
"""
function initial_condition_taylor_green_vortex(x, t,
                                               equations::CompressibleEulerEquations3D)
    A = 1.0 # magnitude of speed
    Ms = 0.1 # maximum Mach number

    rho = 1.0
    v1 = A * sin(x[1]) * cos(x[2]) * cos(x[3])
    v2 = -A * cos(x[1]) * sin(x[2]) * cos(x[3])
    v3 = 0.0
    p = (A / Ms)^2 * rho / equations.gamma # scaling to get Ms
    p = p +
        1.0 / 16.0 * A^2 * rho *
        (cos(2 * x[1]) * cos(2 * x[3]) + 2 * cos(2 * x[2]) + 2 * cos(2 * x[1]) +
         cos(2 * x[2]) * cos(2 * x[3]))

    return prim2cons(SVector(rho, v1, v2, v3, p), equations)
end
initial_condition = initial_condition_taylor_green_vortex

volume_flux = flux_ranocha
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0, -1.0) .* pi
coordinates_max = (1.0, 1.0, 1.0) .* pi
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 1.4)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
