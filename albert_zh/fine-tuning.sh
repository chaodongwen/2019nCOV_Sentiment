#　large
# 跑非官方预训练的large显存不够
#TASK_NAME="2019ncov"
#MODEL_NAME="albert_large_zh"
#CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
## 模型
#export ALBERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
#export ALBERT_LARGE_DIR=$ALBERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
## 数据
#export GLUE_DATA_DIR=$CURRENT_DIR/chineseGLUEdatasets
#export TEXT_DIR=$GLUE_DATA_DIR/$TASK_NAME
## 运行程序(若模型改了，bert_config_file也需要改)
#nohup python3 run_classifier.py   --task_name=$TASK_NAME   --do_train=true   --do_eval=true  --do_predict=false  --data_dir=$TEXT_DIR   --vocab_file=$ALBERT_LARGE_DIR/vocab.txt  \
#    --bert_config_file=$ALBERT_LARGE_DIR/albert_config_large.json --max_seq_length=128  --train_batch_size=32   --learning_rate=1e-5  --num_train_epochs=5 \
#    --output_dir=$CURRENT_DIR/${TASK_NAME}_${MODEL_NAME}/ --init_checkpoint=$ALBERT_LARGE_DIR/albert_model.ckpt &

##　base
# 跑非官方预训练的base效果并不好
TASK_NAME="2019ncov"
MODEL_NAME="albert_base_zh_add"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
# 模型
export ALBERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
export ALBERT_BASE_DIR=$ALBERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
# 数据
export GLUE_DATA_DIR=$CURRENT_DIR/chineseGLUEdatasets
export TEXT_DIR=$GLUE_DATA_DIR/$TASK_NAME
# 运行程序(若模型改了，bert_config_file也需要改)
nohup python3 run_classifier.py   --task_name=$TASK_NAME   --do_train=true   --do_eval=true  --do_predict=false  --data_dir=$TEXT_DIR   --vocab_file=$ALBERT_BASE_DIR/vocab.txt  \
    --bert_config_file=$ALBERT_BASE_DIR/albert_config_base.json --max_seq_length=128 --train_batch_size=64   --learning_rate=5e-5  --num_train_epochs=4 \
    --output_dir=$CURRENT_DIR/${TASK_NAME}_${MODEL_NAME}/ --init_checkpoint=$ALBERT_BASE_DIR/albert_model.ckpt &

##　tiny
# 跑非官方预训练的tiny效果不错，可以达到0.698
#TASK_NAME="2019ncov"
#MODEL_NAME="albert_tiny_zh"
#CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
## 模型
#export ALBERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
#export ALBERT_TINY_DIR=$ALBERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
## 数据
#export GLUE_DATA_DIR=$CURRENT_DIR/chineseGLUEdatasets
#export TEXT_DIR=$GLUE_DATA_DIR/$TASK_NAME
## 运行程序(若模型改了，bert_config_file也需要改)
#nohup python3 run_classifier.py   --task_name=$TASK_NAME   --do_train=false  --do_eval=true  --do_predict=false  --data_dir=$TEXT_DIR   --vocab_file=$ALBERT_TINY_DIR/vocab.txt  \
#    --bert_config_file=$ALBERT_TINY_DIR/albert_config_tiny.json --max_seq_length=128 --train_batch_size=64   --learning_rate=1e-5  --num_train_epochs=5 \
#    --output_dir=$CURRENT_DIR/${TASK_NAME}_${MODEL_NAME}/ --init_checkpoint=$ALBERT_TINY_DIR/albert_model.ckpt &