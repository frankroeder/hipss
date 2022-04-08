N_MPI=16
git checkout main

# NOTE
# Before executing this script, check the parameters and replace the echo commands with eval.

main() {
  ENV=${1-NLReach}
  export CUDA_VISIBLE_DEVICES=${2-}

  echo "RUNNING ON CUDA DEVICE: $CUDA_VISIBLE_DEVICES";

  PRJ_NAME="ICDL_2022_HIPSS"
  EMODES=("2Shape" "2ColorShape" "2" "2Color")

  for n in {1..5}; do
    for ENV_MODE in "${EMODES[@]}"; do
      for ENV_ID in "Panda${ENV}${ENV_MODE}-v0" "Panda${ENV}${ENV_MODE}HI-v0"; do
        if [[ $ENV_MODE == "2" ]]; then
          EPOCH=20
        elif [[ $ENV_MODE == "2ColorShape" ]]; then
          EPOCH=50
        else
          EPOCH=30
        fi
        OPTS="wandb=True env_name=$ENV_ID"
        if [[ $CUDA_VISIBLE_DEVICES ]]; then
          OPTS+=" cuda=True"
        fi
        if [[ $ENV_ID == "Panda${ENV}${ENV_MODE}HI-v0" ]]; then
          for HINDSIGHT in "heir" "hipss"; do
            echo "mpirun -np $N_MPI python -u train.py project_name=${PRJ_NAME} save_dir=${PRJ_NAME} n_epochs=$EPOCH $OPTS hindsight=$HINDSIGHT";
          done
        else
          echo "mpirun -np $N_MPI python -u train.py project_name=${PRJ_NAME} save_dir=${PRJ_NAME} n_epochs=$EPOCH $OPTS";
        fi
      done
    done
  done
}
main "$@";
