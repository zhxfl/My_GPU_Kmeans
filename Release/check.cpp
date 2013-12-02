#include <stdio.h>
#include <cmath>
using namespace std;
int main()
{
    FILE *f1,*f2;
    char a[123],b[123];
    scanf("%s%s",a,b);
    f1 = fopen(a,"r");
    f2 = fopen(b,"r");
    int i =0;
    float x,y,z = 0;
    while(fscanf(f1, "%f", &x)==1)
    {
        fscanf(f2,"%f", &y);
        z += fabs(x-y);
        if(fabs(x-y)>0.01)
            printf("%f %f %d\n", x, y, i);
        i++;
    }
    printf("%f %f %d\n", z, z/i, i);
    return 0;
}
