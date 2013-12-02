#input='edge17695.bin'
#input='color17695.bin'
#input='texture17695.bin'
input='10000.bin'
for k in 256,; do
    optirun ./my-kmeans -o -n $k -b -i Image_data/${input}
    
    #cudaTime=$(optirun ./kmeans-gpu -o -n $k -b -i Image_data/${input} | grep 'Computation' | awk '{print $4}')
    #echo "k = $(printf "%3d" $k) cudaTime = ${cudaTime}s"
done
