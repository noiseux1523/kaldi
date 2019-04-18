#!/bin/bash

# Change this location to somewhere where you want to put the data.
data=./corpus/

data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

stage=0
. utils/parse_options.sh

set -euo pipefail

mkdir -p $data

"""
1. Obtain a written transcript of the speech data

For a more precise alignment, utterance (~sentence) level start and end times are helpful, but not necessary.
"""

# Download dev and test sets
# Saved in ./corpus
#
# name: Mini LibriSpeech ASR corpus
# summary: Subset of LibriSpeech corpus for purpose of regression testing
# category: speech
# license: CC BY 4.0
# file: dev-clean-2.tar.gz   development set, "clean" speech
# file: train-clean-5.tar.gz test set, "clean" speech
# file: md5sum.txt           md5 checksums of files

for part in dev-clean-2 train-clean-5; do
  local/download_and_untar.sh $data $data_url $part
done

"""
2. Format transcripts for Kaldi

Kaldi requires various formats of the transcripts for acoustic model training. You’ll need the start and end times of each utterance, the speaker ID of each utterance, and a list of all words and phonemes present in the transcript.
"""

# Download language models
# Saved in ./corpus with symlink in data/local/lm
#
# name: LibriSpeech language models, vocabulary and G2P models
# summary: Language modelling resources, for use with the LibriSpeech ASR corpus
# category: text
# license: Public domain
# file: librispeech-lm-corpus.tgz  14500 public domain books, used as training material for the LibriSpeech's LM
# file: librispeech-lm-norm.txt.gz Normalized LM training text
# file: librispeech-vocab.txt      200K word vocabulary for the LM
# file: librispeech-lexicon.txt    Pronunciations, some of which G2P auto-generated, for all words in the vocabulary
# file: 3-gram.arpa.gz             3-gram ARPA LM, not pruned
# file: 3-gram.pruned.1e-7.arpa.gz 3-gram ARPA LM, pruned with theshold 1e-7
# file: 3-gram.pruned.3e-7.arpa.gz 3-gram ARPA LM, pruned with theshold 3e-7
# file: 4-gram.arpa.gz             4-gram ARPA LM, usually used for rescoring
# file: g2p-model-5                Fifth order Sequitur G2P model

if [ $stage -le 0 ]; then
  local/download_lm.sh $lm_url $data data/local/lm
fi

# Format the data as Kaldi data directories
if [ $stage -le 1 ]; then

  for part in dev-clean-2 train-clean-5; do
    # Use underscore-separated names in data directories.
    # Returns files in in data/dev_clean_2 and data/train_clean_5:
    #     wav.scp, text, utt2spk, spk2gender, utt2dur
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
  
  # Prepares the dictionary and auto-generates the pronunciations for the words (lexicon),
  # that are in our vocabulary but not in CMUdict.
  # Saved in data/local/dict_nosp.
  #     silence_phones, optional_phones, nonsil_phones, extra_questions and lexicons
  # "nosp" refers to the dictionary before silence probabilities and pronunciation
  # probabilities are added.
  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
    data/local/lm data/local/lm data/local/dict_nosp

  # This script prepares a directory such as data/lang/, in the standard format,
  # given a source directory containing a dictionary lexicon.txt in a form like:
  #     word phone1 phone2 ... phoneN
  # per line (alternate prons would be separate lines).
  #
  # Or a dictionary with probabilities called lexiconp.txt in a form:
  #     word pron-prob phone1 phone2 ... phoneN
  # (with 0.0 < pron-prob <= 1.0); 
  # note: if lexiconp.txt exists, we use it even if lexicon.txt exists.
  #
  # Also files silence_phones.txt, nonsilence_phones.txt, optional_silence.txt
  # and extra_questions.txt 
  #
  # See http://kaldi-asr.org/doc/data_prep.html#data_prep_lang_creating for more info.
  utils/prepare_lang.sh data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

  # Prepares the test time language model(G) transducers
  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
  
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_tglarge
fi

"""
3. Extract acoustic features from the audio

Mel Frequency Cepstral Coefficients (MFCC) are the most commonly used features, but Perceptual Linear Prediction (PLP) features and other features are also an option. These features serve as the basis for the acoustic models.
"""

if [ $stage -le 2 ]; then
  mfccdir=mfcc
  # spread the mfccs over various machines, as this data-set is quite large.
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
    mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
    utils/create_split_dir.pl /export/b{07,14,16,17}/$USER/kaldi-data/egs/librispeech/s5/$mfcc/storage \
      $mfccdir/storage
  fi

  for part in dev_clean_2 train_clean_5; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done

  # Get the shortest 500 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/train_clean_5 500 data/train_500short
fi

"""
4. Train monophone models

A monophone model is an acoustic model that does not include any contextual information about the preceding or following phone. It is used as a building block for the triphone models, which do make use of contextual information.

*Note: from this point forward, we will be assuming a Gaussian Mixture Model/Hidden Markov Model (GMM/HMM) framework. This is in contrast to a deep neural network (DNN) system.
"""

# train a monophone system
if [ $stage -le 3 ]; then
  # TODO(galv): Is this too many jobs for a smaller dataset?
  steps/train_mono.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train_500short data/lang_nosp exp/mono
  # TODO: Understand why we use lang_nosp here...
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
      exp/mono exp/mono/graph_nosp_tgsmall
    for test in dev_clean_2; do
      steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/mono/graph_nosp_tgsmall \
        data/$test exp/mono/decode_nosp_tgsmall_$test
    done
  )&
  
  """
  5. Align audio with the acoustic models

    The parameters of the acoustic model are estimated in acoustic training steps; however, the process can be better optimized by cycling through training and alignment phases. This is also known as Viterbi training (related, but more computationally expensive procedures include the Forward-Backward algorithm and Expectation Maximization). By aligning the audio to the reference transcript with the most current acoustic model, additional training algorithms can then use this output to improve or refine the parameters of the model. Therefore, each training step will be followed by an alignment step where the audio and text can be realigned.
    """

  steps/align_si.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train_clean_5 data/lang_nosp exp/mono exp/mono_ali_train_clean_5
fi

"""
6. Train triphone models

While monophone models simply represent the acoustic parameters of a single phoneme, we know that phonemes will vary considerably depending on their particular context. The triphone models represent a phoneme variant in the context of two other (left and right) phonemes.

At this point, we’ll also need to deal with the fact that not all triphone units are present (or will ever be present) in the dataset. There are (# of phonemes)3 possible triphone models, but only a subset of those will actually occur in the data. Furthermore, the unit must also occur multiple times in the data to gather sufficient statistics for the data. A phonetic decision tree groups these triphones into a smaller amount of acoustically distinct units, thereby reducing the number of parameters and making the problem computationally feasible.
"""

# train a first delta + delta-delta triphone system on all utterances
if [ $stage -le 4 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_clean_5 data/lang_nosp exp/mono_ali_train_clean_5 exp/tri1

  # decode using the tri1 model
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
      exp/tri1 exp/tri1/graph_nosp_tgsmall
    for test in dev_clean_2; do
      steps/decode.sh --nj 5 --cmd "$decode_cmd" exp/tri1/graph_nosp_tgsmall \
      data/$test exp/tri1/decode_nosp_tgsmall_$test
      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
        data/$test exp/tri1/decode_nosp_{tgsmall,tgmed}_$test
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
        data/$test exp/tri1/decode_nosp_{tgsmall,tglarge}_$test
    done
  )&

  # Realignement
  steps/align_si.sh --nj 5 --cmd "$train_cmd" \
    data/train_clean_5 data/lang_nosp exp/tri1 exp/tri1_ali_train_clean_5
fi

"""
7. Re-align audio with the acoustic models & re-train triphone models

Repeat steps 5 and 6 with additional triphone training algorithms for more refined models. These typically include delta+delta-delta training, LDA-MLLT, and SAT. The alignment algorithms include speaker independent alignments and FMLLR.
"""

# Train an LDA+MLLT system.
if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train_clean_5 data/lang_nosp exp/tri1_ali_train_clean_5 exp/tri2b

  # decode using the LDA+MLLT model
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
      exp/tri2b exp/tri2b/graph_nosp_tgsmall
    for test in dev_clean_2; do
      steps/decode.sh --nj 10 --cmd "$decode_cmd" exp/tri2b/graph_nosp_tgsmall \
        data/$test exp/tri2b/decode_nosp_tgsmall_$test
      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
        data/$test exp/tri2b/decode_nosp_{tgsmall,tgmed}_$test
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
        data/$test exp/tri2b/decode_nosp_{tgsmall,tglarge}_$test
    done
  )&

  # Align utts using the tri2b model
  steps/align_si.sh  --nj 5 --cmd "$train_cmd" --use-graphs true \
    data/train_clean_5 data/lang_nosp exp/tri2b exp/tri2b_ali_train_clean_5
fi

# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 6 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train_clean_5 data/lang_nosp exp/tri2b_ali_train_clean_5 exp/tri3b

  # decode using the tri3b model
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
      exp/tri3b exp/tri3b/graph_nosp_tgsmall
    for test in dev_clean_2; do
      steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
        exp/tri3b/graph_nosp_tgsmall data/$test \
        exp/tri3b/decode_nosp_tgsmall_$test
      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
        data/$test exp/tri3b/decode_nosp_{tgsmall,tgmed}_$test
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
        data/$test exp/tri3b/decode_nosp_{tgsmall,tglarge}_$test
    done
  )&
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 7 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train_clean_5 data/lang_nosp exp/tri3b
    
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang_tmp data/lang

  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge

  steps/align_fmllr.sh --nj 5 --cmd "$train_cmd" \
    data/train_clean_5 data/lang exp/tri3b exp/tri3b_ali_train_clean_5
fi


if [ $stage -le 8 ]; then
  # Test the tri3b system with the silprobs and pron-probs.

  # decode using the tri3b model
  utils/mkgraph.sh data/lang_test_tgsmall \
                   exp/tri3b exp/tri3b/graph_tgsmall
                   
  for test in dev_clean_2; do
    steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
                          exp/tri3b/graph_tgsmall data/$test \
                          exp/tri3b/decode_tgsmall_$test
                          
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                       data/$test exp/tri3b/decode_{tgsmall,tgmed}_$test
                       
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri3b/decode_{tgsmall,tglarge}_$test
  done
fi

# Train a chain model
if [ $stage -le 9 ]; then
  local/chain/run_tdnn.sh --stage 0
fi

# local/grammar/simple_demo.sh

# Don't finish until all background decoding jobs are finished.
wait
