import torch


def rademacher(shape, gpu=True):
    """
    Creates a random tensor of size [shape] under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)
    """
    x = torch.empty(shape)
    if gpu:
        x = x.cuda()
    x.random_(0, 2)
    x[x == 0] = -1
    return x


def second_directional_derivative(G, z,  x, G_z, epsilon, path_idx=None, path_matrix=None):
    """
    Computes the second directional derivative of G path_idx.r.t. its input at z in the direction x
    """
    if path_idx is None:  # Apply the Hessian Penalty in Z-space
        return (G(z + x,  path_matrix=path_matrix) - 2 * G_z + G(z - x,  path_matrix=path_matrix)) / (epsilon ** 2)
    else:  # Apply it in path_idx-space
        return (G(z,  path_idx=path_idx+x, path_matrix=path_matrix) - 2 * G_z + G(z, path_idx=path_idx-x, path_matrix=path_matrix)) / (epsilon ** 2)


def multi_layer_second_directional_derivative(G, z,x, G_z, epsilon, path_idx=None, path_matrix=None):
    """
    Same as second_directional_derivative, but assumes G returns multiple outputs in a list
    """
    if path_idx is None:
        _, G_to_x = G(z + x, c, return_bn=True, path_matrix=path_matrix)
        _, G_from_x = G(z - x, c, return_bn=True, path_matrix=path_matrix)
    else:
        _, G_to_x = G(z, c, path_idx=path_idx+x, return_bn=True, path_matrix=path_matrix)
        _, G_from_x = G(z, c, path_idx=path_idx-x, return_bn=True, path_matrix=path_matrix)

    eps_sqr = epsilon ** 2
    sdd = [(G2x - 2 * G_z_base + Gfx) / eps_sqr for G2x, G_z_base, Gfx in zip(G_to_x, G_z, G_from_x)]
    return sdd


def stack_var_and_reduce(list_of_activations, reduction=torch.max):
    second_orders = torch.stack(list_of_activations)  # KxBxCxHxW
    var_tensor = torch.var(second_orders, dim=0, unbiased=True)  # BxCxHxW
    penalty = reduction(var_tensor)  # scalar
    return penalty


def multi_stack_var_and_reduce(sdds, reduction=torch.max, return_separately=False):
    sum_of_penalties = 0 if not return_separately else []
    for activ_n in zip(*sdds):
        penalty = stack_var_and_reduce(activ_n, reduction)
        sum_of_penalties += penalty if not return_separately else [penalty]
    return sum_of_penalties


def hessian_penalty(G, z, path_idx=None, G_z=None, k=2, epsilon=0.1, reduction=torch.mean,
                    multiple_layers=True, return_separately=False, path_matrix=None,dir_count=None):
    """
    Version of the Hessian Penalty that allows taking the Hessian path_idx.r.t. the path_idx input instead of z
    Note: path_idx here refers to the coefficients for the learned directions in path_matrix, it has nothing to do with path_idx-space
    as in StyleGAN, etc.

    :param G: Function that maps z to either a tensor or a size-N list of tensors (activations)
    :param z: (N, dim_z) input to G
    :param c: Class label input to G (not regularized in this version of hessian penalty)
    :param path_idx: (N, ndirs) tensor that represents how far to move z in each of the ndirs directions stored in path_matrix.
              If specified, Hessian is taken path_idx.r.t. path_idx instead of path_idx.r.t. z.
    :param k: Number of Hessian directions to sample (must be >= 2)
    :param G_z: Pre-cached G(z) computation (i.e., a size-N list)
    :param epsilon: Amount to blur G before estimating Hessian (must be > 0)
    :param reduction: Many-to-one function to reduce each pixel's individual hessian penalty into a final loss
    :param multiple_layers: If True, G is expected to return a list of tensors that are all regularized jointly
    :param return_separately: If True, returns hessian penalty for each layer separately, rather than combining them
    :param path_matrix: (ndirs, nz) matrix of directions (rows correspond to directions)

    :return: A differentiable scalar (the hessian penalty), or a list of hessian penalties if return_separately is True
    """
    if G_z is None:
        G_z = G(z, path_idx=path_idx, path_matrix=path_matrix)
        if multiple_layers:
            G_z = G_z[1]
    if path_idx is not None:
        xs = rademacher(torch.Size((k, *path_idx.size()))) * epsilon#k,batch_size,n_dirs
    else:
        xs = rademacher(torch.Size((k, *z.size()))) * epsilon
    for i in range(k):
        for b in range(path_idx.size()[0]):
            for d in range(dir_count,path_idx.size()[1]):
                xs[i,b,d]=0
    second_orders = []
    for i in range(k):#公式5
        x = xs[i]
        if multiple_layers:
            central_second_order = multi_layer_second_directional_derivative(G, z,  x, G_z, epsilon, path_idx=path_idx, path_matrix=path_matrix)
        else:
            central_second_order = second_directional_derivative(G, z,  x, G_z, epsilon, path_idx=path_idx, path_matrix=path_matrix)
        second_orders.append(central_second_order)
    #下面进行公式3,4
    if multiple_layers:
        penalty = multi_stack_var_and_reduce(second_orders, reduction, return_separately)
    else:
        penalty = stack_var_and_reduce(second_orders, reduction)
    return penalty


def _test_hessian_penalty():
    batch_size = 10
    nz = 2
    z = torch.randn(batch_size, nz)
    reduction = lambda x: x.abs().mean()
    G = lambda z, path_matrix: [z[:, 0] * z[:, 1], (z[:, 0] ** 2) * z[:, 1]]
    ground_truth = [4, reduction(16 * z[:, 0] ** 2).item()]
    predicted = hessian_penalty(G, z, None, G_z=None, k=100, reduction=reduction, return_separately=True)
    predicted = [p.item() for p in predicted]
    print('Ground Truth: %s' % ground_truth)
    print('Approximation: %s' % predicted)  # This should be "close-ish" to ground_truth, but not exactly correct
    print('Difference: %s' % [str(100 * abs(p - gt) / gt) + '%' for p, gt in zip(predicted, ground_truth)])
