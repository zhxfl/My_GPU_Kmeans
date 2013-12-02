#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
int main()
{
    FILE * infile;
        char filename[125] = "4.bin";
        if ((infile = fopen(filename, "w+")));
        int numObjs = 4;
        int numCoords = 2;
        fwrite((void*)&numObjs, sizeof(int),1, infile);
        fwrite( (void*)&numCoords, sizeof(int),1,infile);

        /* allocate space for objects[][] and read all objects */
        int len = (numObjs) * (numCoords);
        float **objects    = (float**)malloc((numObjs) * sizeof(float*));
        objects[0] = (float*) malloc(len * sizeof(float));
        for (int i=1; i<(numObjs); i++)
            objects[i] = objects[i-1] + (numCoords);

        for(int i = 0; i < numObjs;i++)
        {
            for(int j = 0; j < numCoords;j++)
            {
                objects[i][j] = rand()% 100;
            }
        }
        fwrite(objects[0],sizeof(float),len,infile);
        fclose(infile);
	return 0;
}
