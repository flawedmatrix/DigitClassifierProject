// Basically hardcoding the structure of the files here.

struct train_small {
    float *images[7];
    float *labels[7];
};

struct train_full {
    float *images;
    float *labels;
};

struct test {
    float *images;
    float *labels;
};