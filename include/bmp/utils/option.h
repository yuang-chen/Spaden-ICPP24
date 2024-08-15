
#pragma once
#include <getopt.h>
#include <string>

namespace bmp {

struct Config {
    std::string input_file;
    int exec_iterations = 100;
};

std::string option_hints = "              [-i input_file]\n"
                           "              [-c columns_of_dense_matrix\n"
                           "              [-e execution_iterations]\n";

auto program_options(int argc, char *argv[]) {
    Config config;
    int opt;
    if (argc == 1) {
        printf("Usage: %s ... \n%s", argv[0], option_hints.c_str());
        std::exit(EXIT_FAILURE);
    }
    while ((opt = getopt(argc, argv, "e:i:c:v:")) != -1) {
        switch (opt) {
        case 'i':
            config.input_file = optarg;
            break;
        case 'e':
            config.exec_iterations = std::stoi(optarg);
            break;
        default:
            printf("Usage: %s ... \n%s", argv[0], option_hints.c_str());
            exit(EXIT_FAILURE);
        }
    }
    if (config.input_file.empty()) {
        printf(" input file is empty\n");
        exit(EXIT_FAILURE);
    }
    printf("--------experimental setting--------\n");
    printf("input path: %s, iterations: %d",
           config.input_file.c_str(), config.exec_iterations);

    return config;
}
}  // namespace bmp