# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function limiter_zhang_shu!(u, threshold::Real, variable,
                            mesh::AbstractMesh{2}, equations, dg::DGSEM, cache;
                            refined_elements = [])
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    @trixi_timeit timer() "limiter! refined_elements" if !isempty(refined_elements)
        element_id = 1
        theta_vec = ones(eltype(u), 2^ndims(mesh))
        u_mean_vec = zeros(eltype(u), size(u, 1), 2^ndims(mesh))

        for element_id_old in 1:refined_elements[end]
            if element_id_old in refined_elements
                # Increment `element_id` on the refined mesh with the number
                # of children, i.e., 4 in 2D
                element_id += 2^ndims(mesh)

                theta_vec .= one(eltype(u))
                u_mean_vec .= zero(eltype(u))
                # Iterate over the children of the current element
                for new_element in 1:(2^ndims(mesh))
                    new_element_id = element_id + new_element - 1 - 2^ndims(mesh)

                    # determine minimum value
                    value_min = typemax(eltype(u))
                    for j in eachnode(dg), i in eachnode(dg)
                        u_node = get_node_vars(u, equations, dg, i, j, new_element_id)
                        value_min = min(value_min, variable(u_node, equations))
                    end

                    # compute mean value
                    u_mean = zero(get_node_vars(u, equations, dg, 1, 1, new_element_id))
                    total_volume = zero(eltype(u))
                    for j in eachnode(dg), i in eachnode(dg)
                        volume_jacobian = abs(inv(get_inverse_jacobian(inverse_jacobian,
                                                                       mesh,
                                                                       i, j,
                                                                       new_element_id)))
                        u_node = get_node_vars(u, equations, dg, i, j, new_element_id)
                        u_mean += u_node * weights[i] * weights[j] * volume_jacobian
                        total_volume += weights[i] * weights[j] * volume_jacobian
                    end
                    # normalize with the total volume
                    u_mean = u_mean / total_volume
                    u_mean_vec[:, new_element] .= u_mean

                    # detect if limiting is necessary
                    value_min < threshold || continue

                    # We compute the value directly with the mean values, as we assume that
                    # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
                    value_mean = variable(u_mean, equations)
                    theta = (value_mean - threshold) / (value_mean - value_min)
                    theta_vec[new_element] = theta
                end
                theta_global = minimum(theta_vec)
                theta_global < 1 || continue

                # Iterate again over the children to apply synchronized shifting
                for new_element in 1:(2^ndims(mesh))
                    new_element_id = element_id + new_element - 1 - 2^ndims(mesh)
                    for j in eachnode(dg), i in eachnode(dg)
                        u_node = get_node_vars(u, equations, dg, i, j, new_element_id)
                        u_mean = get_node_vars(u_mean_vec, equations, dg, new_element)
                        set_node_vars!(u,
                                       theta_global * u_node +
                                       (1 - theta_global) * u_mean,
                                       equations, dg, i, j, new_element_id)
                    end
                end
            else
                # Increment `element_id` on the unrefined mesh
                element_id += 1
            end
        end
    end

    @threaded for element in eachelement(dg, cache)
        # determine minimum value
        value_min = typemax(eltype(u))
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            value_min = min(value_min, variable(u_node, equations))
        end

        # detect if limiting is necessary
        value_min < threshold || continue

        # compute mean value
        u_mean = zero(get_node_vars(u, equations, dg, 1, 1, element))
        total_volume = zero(eltype(u))
        for j in eachnode(dg), i in eachnode(dg)
            volume_jacobian = abs(inv(get_inverse_jacobian(inverse_jacobian, mesh,
                                                           i, j, element)))
            u_node = get_node_vars(u, equations, dg, i, j, element)
            u_mean += u_node * weights[i] * weights[j] * volume_jacobian
            total_volume += weights[i] * weights[j] * volume_jacobian
        end
        # normalize with the total volume
        u_mean = u_mean / total_volume

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                           equations, dg, i, j, element)
        end
    end

    return nothing
end
end # @muladd
