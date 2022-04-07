molecules=("hydrofluoric_acid"  "ammonia"  "water" "methane" "o2" "CO2" "NO2" "NaCl")

methods=("qdrift" "rand_ham" "taylor_naive" "taylor_on_the_fly" "configuration_interaction" "low_depth_trotter" "shc_trotter" "low_depth_taylor" "low_depth_taylor_on_the_fly" "linear_t" "sparsity_low_rank" "interaction_picture")



for molecule in "${molecules[@]}"; do
    echo "">./results/$molecule.txt
    INDEX_METHOD=1
    NUMBER_METHODS="${#methods[@]}"
    for method in "${methods[@]}"; do
        python3 main.py $molecule $method >> ./results/$molecule.txt
	echo "[$INDEX_METHOD/$NUMBER_METHODS] $molecule with method $method calculated"
	INDEX_METHOD=$((INDEX_METHOD+1))
    done
    echo ""
done
