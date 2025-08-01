#include "nn.h"
#include "test.h"
#include <stdio.h>
#include <math.h>

int assert(int condition, const char *message)
{
    if (!condition)
    {
        fprintf(stderr, "Assertion failed: %s\n", message);
        return 1;
    }

    return 0;
}

int feed_forward_should_compute()
{
    float input[] = {-1, 2, 3, -4};
    // Transposed weights to match weights[input][output] format
    float weights[][3] = {
        {.2f, .5f, .9f},
        {.3f, .6f, 1.0f},
        {.1f, .7f, 1.1f},
        {.4f, .8f, 1.2f}
    };
    float output[] = {0, 0, 0};
    float bias[] = {0.1f, 0.2f, 0.3f};

    float output0 = 0
        + -1.0f * 0.2f 
        +  2.0f * 0.3f 
        +  3.0f * 0.1f 
        + -4.0f * 0.4f;
    output0 = tanh(output0 + .1f);

    float output1 = 0
        + -1.0f * 0.5f 
        +  2.0f * 0.6f 
        +  3.0f * 0.7f 
        + -4.0f * 0.8f;
    output1 = tanh(output1 + .2f);

    float output2 = 0
        + -1.0f * 0.9f 
        +  2.0f * 1.0f 
        +  3.0f * 1.1f 
        + -4.0f * 1.2f;
    output2 = tanh(output2 + .3f);

    forward(4, input, 3, output, weights, bias);

    return assert(
        roundf(output0) == roundf(output[0]) &&
        roundf(output1) == roundf(output[1]) && 
        roundf(output2) == roundf(output[2]),

        "Feed forward should compute the correct output");
}


int run_tests(int argc, char const *argv[])
{
    int n = feed_forward_should_compute();
    
    if (n == 0)
    {
        printf("All tests passed!\n");
    }
    else
    {
        printf("%d tests failed.\n", n);
    }

    return n;
}