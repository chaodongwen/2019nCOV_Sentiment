##　large
#TASK_NAME="2019ncov"
#MODEL_NAME="albert_large_zh_google"
#CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
## 模型
#export ALBERT_CONFIG_DIR=$CURRENT_DIR/albert_config
#export ALBERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
#export ALBERT_LARGE_DIR=$ALBERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
## 数据
#export GLUE_DATA_DIR=$CURRENT_DIR/chineseGLUEdatasets
#export TEXT_DIR=$GLUE_DATA_DIR/$TASK_NAME
## 运行程序(若模型改了，bert_config_file也需要改)
#nohup python3 run_classifier_self.py   --task_name=$TASK_NAME   --do_train=false   --do_eval=false  --do_predict=true  --data_dir=$TEXT_DIR   --vocab_file=$ALBERT_LARGE_DIR/vocab_chinese.txt  \
#    --albert_config_file=$ALBERT_LARGE_DIR/albert_config.json --max_seq_length=128 --train_batch_size=32   --learning_rate=1e-5  --num_train_epochs=10 \
#    --output_dir=$CURRENT_DIR/${TASK_NAME}_${MODEL_NAME}/ --init_checkpoint=$CURRENT_DIR/${TASK_NAME}_${MODEL_NAME}/model.ckpt-31000 &

#　base
TASK_NAME="2019ncov"
MODEL_NAME="albert_base_zh_google"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
# 模型
export ALBERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
export ALBERT_BASE_DIR=$ALBERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
# 数据
export GLUE_DATA_DIR=$CURRENT_DIR/chineseGLUEdatasets
export TEXT_DIR=$GLUE_DATA_DIR/$TASK_NAME
# 运行程序(若模型改了，bert_config_file也需要改)
nohup python3 run_classifier_self.py   --task_name=$TASK_NAME   --do_train=false   --do_eval=false  --do_predict=true  --data_dir=$TEXT_DIR   --vocab_file=$ALBERT_BASE_DIR/vocab_chinese.txt  \
    --albert_config_file=$ALBERT_BASE_DIR/albert_config.json --max_seq_length=128 --train_batch_size=64   --learning_rate=1e-5  --num_train_epochs=10 --save_checkpoints_steps=2000 \
    --output_dir=$CURRENT_DIR/${TASK_NAME}_${MODEL_NAME}/ --init_checkpoint=$CURRENT_DIR/${TASK_NAME}_${MODEL_NAME}/model.ckpt-4000 &
