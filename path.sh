# export KALDI_ROOT=/misc/scratch01/reco/osterrfr/kaldi_hg_builds/build_2019-03-24_33_1ac8c922cbf6b2c34756d4b467cfa6067a6dba90
export KALDI_ROOT=`pwd`/../../..
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

# For now, don't include any of the optional dependenices of the main
# librispeech recipe
