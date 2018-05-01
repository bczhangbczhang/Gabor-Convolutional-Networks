export dataset=CIFAR100
export savedir=logs/gcn/${dataset}
mkdir -p ${savedir}
curData=$(date "+%Y%m%d%H%M%S")

python CIFAR.py --gpu 0 \
                --model gcn \
                --dataset ${dataset} \
                --depth 40 \
                --widen-factor 2 \
                --nchannel 4 \
                --nscale 1 \
                --dropout-rate 0 \
                | tee ${savedir}/${curData}.txt