set -e

# Right now record needs to be checked and wsc needs to be added
#datasets=(boolq  cb  copa  multirc  record  rte  wic  wsc)
datasets=(boolq  cb  copa  multirc  rte  wic)
models=(bert-base-cased facebook/bart-base)

for model in ${models[@]}; do
    for dataset in ${datasets[@]}; do
        echo "RAVIOLI Running with model ${model} on dataset ${dataset}"
        python -m src.main --config-name ${dataset} \
            hydra.env.model_name=${model} \
            +overwrite_cache=True \
            +overwrite_output_dir=True \
            output_dir=models/${dataset}_test/${model} \
            num_train_epochs=1
        echo "RAVIOLI FINISHED Running with model ${model} on dataset ${dataset}"
    done
done
