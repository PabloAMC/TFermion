molecules=("water" "methane" "ammonia" "CO2", "NO2")
methods=("qdritf" "rand_ham" "taylor_naive" "taylor_on_the_fly", "configuration_interaction", "low_depth_trotter", "low_depth_taylor", "low_depth_taylor_on_the_fly", "linear_t", "sparsity_low_rank", "interaction_picture")

for molecule in "${molecules[@]}"; do
    for method in "${methods[@]}"; do
        touch $molecule.txt
        python3 main.py $molecule $method > $molecule.txt
        '\n' > $molecule.txt
        echo "$molecule with method $method calculated"
    done
    echo "\n"
done