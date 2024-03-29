from mmsr.models.archs import _arch_modules


def dynamical_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
        files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class： Instantiated class.
    """

    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_} is not found.')
    return cls_(**opt)


# generator
def define_net_g(opt):
    opt_net = opt['network_g']
    network_type = opt_net.pop('type')
    net_g = dynamical_instantiation(_arch_modules, network_type, opt_net)

    return net_g


# Discriminator
def define_net_d(opt):
    opt_net = opt['network_d']
    network_type = opt_net.pop('type')

    net_d = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_d
