nvcc -o nccl_test nccl_test.cc -lnccl -std=c++11

for i in {1..100}; do
    echo "test "${i}
   ./nccl_test
done
