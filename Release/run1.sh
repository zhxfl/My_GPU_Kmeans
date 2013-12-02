#input='color17695.bin'
#input='texture17695.bin'
input='100000.bin'
for k in 2 4 8 16 32 64 128 256; do
    #optirun ./kmeans-gpu -o -n $k -b -i Image_data/${input}
    
    cudaTime=$(optirun ./kmeans-gpu -o -n $k -b -i Image_data/${input} | grep 'Computation' | awk '{print $4}')
    echo "k = $(printf "%3d" $k) cudaTime = ${cudaTime}s"
done
