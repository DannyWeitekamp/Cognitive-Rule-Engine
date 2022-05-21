import sys, argparse

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Utilities that help with cre usage')
    parser.add_argument('--clear-cache', dest='clear_cache', action='store_true',
         help="Clears the cre caches.")

    try:
        args = parser.parse_args(argv)
    except Exception:
        parser.print_usage()
        sys.exit()

    if(args.clear_cache):
        from cre.caching import clear_cache
        clear_cache()
    else:
        parser.print_usage()
        sys.exit()

def main():
    args = parse_args(sys.argv[1:])

