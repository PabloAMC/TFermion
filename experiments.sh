molecules=("water" "methane" "ammonia" "CO2" "NO2")
methods=("qdrift" "rand_ham" "taylor_naive" "taylor_on_the_fly" "configuration_interaction" "low_depth_trotter" "low_depth_taylor" "low_depth_taylor_on_the_fly" "linear_t" "sparsity_low_rank" "interaction_picture")

for molecule in "${molecules[@]}"; do
    echo "">./results/$molecule.txt
    for method in "${methods[@]}"; do
        python3 main.py $molecule $method >> ./results/$molecule.txt
        echo "$molecule with method $method calculated"
    done
    echo ""
done