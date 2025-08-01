int run_model(int argc, char const *argv[]);
float identity(float x);

float activation(float x);
float activation_prime(float tanh);

void forward(
    const int m,
    const float input[m],
    const int n,
    float output[n],
    const float weights[m][n],
    const float bias[n],
    float (*activation)(float));

void softmax(const int m, float input[m]);

void stochastic_gradient_descent();
