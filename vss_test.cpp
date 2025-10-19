#include "runner.h"
using namespace vss;

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dim> <data_dir> <index_type>\n";
        return 1;
    }

    VSSRunner runner(std::stoi(argv[1]), argv[2], argv[3]);
    runner.run_build();
    runner.run_search();

    return 0;
}