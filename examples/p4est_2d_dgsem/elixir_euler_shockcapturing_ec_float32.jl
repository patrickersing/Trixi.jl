using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4f0)

initial_condition = initial_condition_weak_blast_wave

surface_flux = flux_ranocha
volume_flux = flux_ranocha
polydeg = 4
basis = LobattoLegendreBasis(Float32, polydeg)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max = 1.0f0,
                                         alpha_min = 0.001f0,
                                         alpha_smooth = true,
                                         variable = density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg = volume_flux,
                                                 volume_flux_fv = surface_flux)

solver = DGSEM(polydeg = polydeg, surface_flux = surface_flux,
               volume_integral = volume_integral, RealT = Float32)

###############################################################################

coordinates_min = (-1.0f0, -1.0f0)
coordinates_max = (+1.0f0, +1.0f0)

trees_per_dimension = (4, 4)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 4, initial_refinement_level = 2,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 periodicity = true, RealT = Float32)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0f0, 2.0f0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 1.0f0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback)
###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0f0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
