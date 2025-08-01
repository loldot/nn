void stochastic_gradient_descent();
int run_training(int argc, char const *argv[]);

void forward(
    const int m,
    const float input[m],
    const int n,
    float output[n],
    const float weights[m][n],
    const float bias[n]);