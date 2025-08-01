#include <stdio.h>
#include "nn.h"
#include "test.h"

int main(int argc, char const *argv[])
{
    if (argc > 1 && argv[1][0] == 't')
    {
        run_tests(argc, argv);
        return 0;
    }
    run_training(argc, argv);
    return 0;
}