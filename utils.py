def print_config(config, args):
    print_args =['std', 'num_workers', 'root_dir', 'warmup_batches', 'name', 'precision', 'gpus', 'max_epochs', 'max_time']

    print("{:<15}| {:<10}".format('Var', 'Value'))
    print('-'*22)
    for key in config:
        print("{:<15}| {:<10} ".format(key, config[key]))

    for arg in print_args:
        print("{:<15}| {:<10} ".format(arg, getattr(args, arg)))

    if args.gan:
        print("{:<15}| {:<10} ".format('GAN', 'relativistic average'))
    else:
        print("{:<15}| {:<10} ".format('GAN', 'vanilla'))
