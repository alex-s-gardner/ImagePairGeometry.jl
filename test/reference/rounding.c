#include <stdio.h>
#include <math.h>
#include <stdint.h>
/* Pins the reference's three double->integer conversions:
     std::round then (GInt32)   -> cround32
     plain (GInt32)             -> ctrunc32   */
int main(void) {
    double xs[] = {0.5,-0.5,1.5,-1.5,2.5,-2.5,3.5,0.49999999999999994,
                   22.666666666666668,22.99,-22.99,340.0/240.0*16.0,
                   0.0,-0.0,1e-300,299.5,-299.5,294.5,-294.5,
                   16.000000000000004,15.999999999999998,
                   1.0000000000000002,-1.0000000000000002};
    int n = sizeof(xs)/sizeof(double);
    printf("x_hex,x_dec,std_round_hex,cround32,ctrunc32\n");
    for (int i=0;i<n;i++){
        double x = xs[i];
        double r = round(x);          /* C ties away from zero */
        int32_t ri = (int32_t)r;      /* implicit conversion after round */
        int32_t ti = (int32_t)x;      /* implicit conversion alone: truncation */
        uint64_t xb, rb;
        __builtin_memcpy(&xb,&x,8); __builtin_memcpy(&rb,&r,8);
        printf("0x%016llx,%.17g,0x%016llx,%d,%d\n",
               (unsigned long long)xb, x, (unsigned long long)rb, ri, ti);
    }
    return 0;
}
