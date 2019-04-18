# Kaldi

# Recipe for MiniLibriSpeech

Source:

* https://groups.google.com/forum/#!topic/kaldi-help/tzyCwt7zgMQ

* https://towardsdatascience.com/how-to-start-with-kaldi-and-speech-recognition-a9b7670ffff6

* https://eleanorchodroff.com/tutorial/kaldi/training-overview.html

* https://jrmeyer.github.io/asr/2016/12/15/DNN-AM-Kaldi.html

* https://kaldi-asr.org/doc/kaldi_for_dummies.html

* https://kaldi-asr.org/doc/tutorial_running.html for commands to view results and models

Recipe taken from crim Kaldi repo cloned with:

* `git clone https://www.crim.ca/stash/scm/reco/crim_kaldi_egs.git`

It is located at:

* `crim_kaldi_egs/mini_librispeech/s5`

Snippets of code are taken from:

* `crim_kaldi_egs/mini_librispeech/s5/run.sh`

## Prepare directory structure and symbolic links

### Create symbolic links in `crim_kaldi_egs/mini_librispeech/s5` for:

* `steps`: `ln -s ../wsj/s5/steps .`
    
* `utils`: `ln -s ../wsj/s5/utils .`
    
### Create directories if not already present in recipe:

* `conf`: Configuration file for specific recipe. The directory `conf`local requires one file mfcc.conf, which contains the parameters for MFCC feature extraction.
    
* `local`: Local contains data for this specific recipe or project.
