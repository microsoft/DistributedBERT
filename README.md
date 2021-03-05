# DistributedBERT

DistributedBERT is based on the official TensorFlow BERT with following improvements
- Higher performance: Support distributed training through Horovod with nearly linear acceleration, support mixed-precision training
- Higher accuracy: Bug fixes and integrated with more advanced techs such as LAMB
- Easier to use: Customized with more settings
- More robust: Preemption and failure recovery
- Easy to leverage: Easy to apply in other BERT-like models such as RoBERTa, ALBERT, ...

## Requirements

- NVIDIA CUDA 10.0+
- Open MPI 3.1.0+
- Tensorflow 1.13.1+
- Horovod 0.16.0+

## Example Training Command

```
export CODE_PATH=/your/path/DistributedBERT
export MODEL_PATH=/your/path/uncased_L-24_H-1024_A-16
export OUTPUT_PATH=/your/path/output
export TRAIN_DATA=/your/path/train
export TEST_DATA=/your/path/test

mpirun -np 4 -H localhost:4 -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python $CODE_PATH/run_classifier.py \
    --data_dir $TRAIN_DATA \
    --test_data_dir $TEST_DATA \
    --output_dir $OUTPUT_PATH \
    --vocab_file $MODEL_PATH/vocab.txt \
    --bert_config_file $MODEL_PATH/bert_config.json \
    --init_checkpoint $MODEL_PATH/bert_model.ckpt \
    --do_train \
    --do_predict \
    --task_name=qk \
    --label_list=0,1,2,3 \
    --max_seq_length=32 \
    --train_batch_size=64 \
    --num_train_epochs=3 \
    --learning_rate=1e-5 \
    --adjust_lr \
    --xla \
    --reduce_log \
    --keep_checkpoint_max=1 \
```
