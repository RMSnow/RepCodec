# Example 1: dump from HuBERT base layer 9 
# (for data2vec, simply change "model_type" to data2vec and "ckpt_path" to the path of data2vec model)


layer=18

# # librispeech dev
# CUDA_VISIBLE_DEVICES=0 python3 examples/dump_feature.py \
#     --model_type hubert \
#     --tsv_path /data/home/xueyao/workspace/RepCodec/dev.tsv \
#     --ckpt_path "/data/home/xueyao/workspace/RepCodec/pretrained/hubert_large_ll60k.pt"  \
#     --layer ${layer} \
#     --feat_dir /fsx-project/xueyao/data/hubert_large_l${layer}/dev

# librispeech train
CUDA_VISIBLE_DEVICES=0 python3 examples/dump_feature.py \
    --model_type hubert \
    --tsv_path /data/home/xueyao/workspace/RepCodec/train.tsv \
    --ckpt_path "/data/home/xueyao/workspace/RepCodec/pretrained/hubert_large_ll60k.pt"  \
    --layer ${layer} \
    --feat_dir /fsx-project/xueyao/data/hubert_large_l${layer}/train