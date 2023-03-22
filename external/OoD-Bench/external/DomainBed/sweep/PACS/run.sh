dataset=PACS
command=$1
launcher=$2
data_dir=$3
algorithms=$4
n_hparams=$5
n_trials=$6
teacher_arch=$7

#ERM IRM VREx MMD RSC ARM CORAL SagNet GroupDRO Mixup MLDG DANN MTL ANDMask IGA ERDG

python -m domainbed.scripts.sweep ${command}\
       --datasets ${dataset}\
       --algorithms ${algorithms}\
       --data_dir ${data_dir}\
       --command_launcher ${launcher}\
       --single_test_envs\
       --steps 5001\
       --holdout_fraction 0.1\
       --n_hparams ${n_hparams}\
       --n_trials ${n_trials}\
       --skip_confirmation\
       --hparams "$(<sweep/${dataset}/hparams.json)"\
       --output_dir "/srv/anisio/does_kd/output_ood_models/${dataset}/"${teacher_arch}\
       --teacher_arch ${teacher_arch}
