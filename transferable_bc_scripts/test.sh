while getopts n: flag
do
    case "${flag}" in
        n) n_blocks=${OPTARG};;
      *) echo "usage: $0 [-n]" >&2
         exit 1 ;;
    esac
done
export N_BLOCKS=$n_blocks; echo "N_BLOCKS: ${N_BLOCKS}"
