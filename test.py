#!/usr/bin/python
import os.path
import optparse

import dynamic_benchmark

FILENAME = os.path.basename(__file__)
THREE_OBJ_PROBLEMS = ["DB9a", "DB9m", "DB10a", "DB10m"]

def file_to_list(**kwargs):
    with open(kwargs["filename"], "r") as fid:
        content = [line.strip() for line in fid.readlines()]
    return content

def process(**kwargs):
    tolerance = kwargs["tolerance"]
    filename  = "test_{}.dat".format(kwargs["problem"])
    content   = file_to_list(filename=filename)
    line_num  = len(content)
    benchmark_func = getattr(dynamic_benchmark, kwargs["problem"])
    fail_num = 0
    obj_num = 3 if kwargs["problem"] in THREE_OBJ_PROBLEMS else 2

    for line in content:
        line = [ float(e) for e in line.split() ]
        if obj_num == 2:
            (t, *var, f1, f2) = line
        if obj_num == 3:
            (t, *var, f1, f2, f3) = line

        f_vec = benchmark_func(var, t)
        if abs(f1 - f_vec[0]) > tolerance or abs(f2 - f_vec[1]) > tolerance:
            fail_num = fail_num + 1
    print("Problem: {}\tPassing Rate: {}%\tTotal Check: {}".\
            format(kwargs["problem"], 100*(line_num-fail_num)/float(line_num),\
            line_num))

def test():
    # Declare parser to parse command line arguments
    description = FILENAME + " -p <benchmark_problem> -t <tolerant>"
    parser = optparse.OptionParser(description)
    parser.add_option("-p", dest="problem", type="string", \
            help="specify name of the benchmark problem [DB1a]")
    parser.add_option("-t", dest="tolerance", type="float", \
            help="specify the tolerance of the difference [1e-3]")

    # Get the command line arguments
    (options, args) = parser.parse_args()
    problem         = options.problem or "DB1a"
    tolerance       = options.tolerance or 1e-4
    
    # Process the test
    process(problem=problem, tolerance=tolerance)

if __name__ == "__main__":
    test()
