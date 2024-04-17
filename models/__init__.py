def create_model(opt):
    model = None
    print(opt.model)
    # Domain Adaptation
    if opt.model == 'DA':
        opt.dataset_mode = 'unaligned'
        from .DA import DA
        model = DA()
    # Image-to-Image
    elif opt.model == 'I2I':
        opt.dataset_mode = 'aligned'
        from .I2I import I2I
        model = I2I()        
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model %s was created" % (model.name()))
    return model
