import time
from main import raytrace


def main():
    iters = 30
    tot = 0
    for i in range(iters):
        start = time.time()
        raytrace()
        tim = time.time() - start
        print(f"This iteration took {tim} seconds to run.")
        tot += tim
    print(f"Average time was {tot / iters} seconds")


if __name__ == '__main__':
    main()
