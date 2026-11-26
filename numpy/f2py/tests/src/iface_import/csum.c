int c_sum(int n, int *x) {
    int i, s;
    s = 0;
    for (i = 0; i < n; ++i) {
        s += x[i];
    }
    return s;
}
