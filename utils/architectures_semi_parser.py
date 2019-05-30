import architectures as arch
import sys

def arch_semi_parse(parser):
    base = arch.Architecture
    all = [ (name, cls) for name, cls in arch.__dict__.items()
            if isinstance(cls, type) and issubclass(cls, base) and cls != base]


    parser.add_argument('-a', '--architecture', type=str,
                        help='architecture to use (' + ','.join([name for name, cls in all]) + ')', required = True)
    args = vars(parser.parse_args())
    net_cls = None
    for name, cls in all:
        if name.lower().startswith(args['architecture'].lower()):
            net_cls = cls
    if net_cls is None :
        parser.print_help()
        sys.exit(1)

    return args, net_cls