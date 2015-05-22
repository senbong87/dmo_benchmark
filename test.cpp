#include <random>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "dynamic_benchmark.h"

int main(int argc, char* argv[])
{
    const size_t inst_num = 1e3;
    const size_t var_num  = 21;
    
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    // function pointer to store desired benchmark function
    std::vector<double> (*benchmark_func)(const vector<double> &, double);

    // Check number of arguments
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <benchmark functions>" << std::endl;
        return 1;
    }

    size_t count = 1;
    while(count <= argc - 1) {

        // Assign correct benchmark function to the function pointer
        std::string problem_string(argv[count]);
        if(problem_string.compare("DB1a") == 0)
            benchmark_func = DB1a;
        else if(problem_string.compare("DB1m") == 0)
            benchmark_func = DB1m;
        else if(problem_string.compare("DB2a") == 0)
            benchmark_func = DB2a;
        else if(problem_string.compare("DB2m") == 0)
            benchmark_func = DB2m;
        else if(problem_string.compare("DB3a") == 0)
            benchmark_func = DB3a;
        else if(problem_string.compare("DB3m") == 0)
            benchmark_func = DB3m;
        else if(problem_string.compare("DB4a") == 0)
            benchmark_func = DB4a;
        else if(problem_string.compare("DB4m") == 0)
            benchmark_func = DB4m;
        else if(problem_string.compare("DB5a") == 0)
            benchmark_func = DB5a;
        else if(problem_string.compare("DB5m") == 0)
            benchmark_func = DB5m;
        else if(problem_string.compare("DB6a") == 0)
            benchmark_func = DB6a;
        else if(problem_string.compare("DB6m") == 0)
            benchmark_func = DB6m;
        else if(problem_string.compare("DB7a") == 0)
            benchmark_func = DB7a;
        else if(problem_string.compare("DB7m") == 0)
            benchmark_func = DB7m;
        else if(problem_string.compare("DB8a") == 0)
            benchmark_func = DB8a;
        else if(problem_string.compare("DB8m") == 0)
            benchmark_func = DB8m;
        else if(problem_string.compare("DB9a") == 0)
            benchmark_func = DB9a;
        else if(problem_string.compare("DB9m") == 0)
            benchmark_func = DB9m;
        else if(problem_string.compare("DB10a") == 0)
            benchmark_func = DB10a;
        else if(problem_string.compare("DB10m") == 0)
            benchmark_func = DB10m;
        else if(problem_string.compare("DB11a") == 0)
            benchmark_func = DB11a;
        else if(problem_string.compare("DB11m") == 0)
            benchmark_func = DB11m;
        else if(problem_string.compare("DB12a") == 0)
            benchmark_func = DB12a;
        else if(problem_string.compare("DB12m") == 0)
            benchmark_func = DB12m;
        else {
            std::cerr << "[ERROR] Benchmark problem \"" << problem_string << "\" is not found." << std::endl;
            return 1;
        } // end if/else
        
        // Store in test.dat file
        std::string filename = "test_" + std::string(argv[count]) + ".dat";
        std::ofstream fid;
        fid.open(filename);

        int obj_num = 2;
        if(problem_string.compare("DB9a") == 0 || problem_string.compare("DB9m") == 0 ||
                problem_string.compare("DB10a") == 0 || problem_string.compare("DB10m") == 0 ||
                problem_string.compare("DB11a") == 0 || problem_string.compare("DB11m") == 0 ||
                problem_string.compare("DB12a") == 0 || problem_string.compare("DB12m") == 0) {
            obj_num = 3;
        } // end if 

        double t, interval = 0.1;
        for(size_t k = 0; k <= 100; ++k) {
            t = interval * static_cast<double>(k);

            for(size_t i = 0; i < inst_num; ++i) {
                std::vector<double> x_var;
                x_var.push_back(distribution(generator));

                if(obj_num == 3)
                    x_var.push_back(distribution(generator));

                for(size_t j = 1; j < var_num; ++j)
                    x_var.push_back(2.0*distribution(generator)-1);

                std::vector<double> y_obj = benchmark_func(x_var, t);

                // Write to the file
                fid << t << " ";
                for(size_t j = 0; j < x_var.size(); ++j)
                    fid << x_var[j] << " ";

                for(size_t j = 0; j < y_obj.size(); ++j)
                    fid << y_obj[j] << " ";
                fid << std::endl;
            } // end for
        } // end outer for

        // Close the file
        fid.close();

        ++count;
    } // end while
    return 0;
} // end main
