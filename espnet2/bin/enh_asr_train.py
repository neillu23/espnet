#!/usr/bin/env python3
from espnet2.tasks.enh_asr import EnhASRTask


def get_parser():
    parser = EnhASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_s3prl_enh_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_s3prl_enh_train.py --config conf/train_asr.yaml
    """
    EnhASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

