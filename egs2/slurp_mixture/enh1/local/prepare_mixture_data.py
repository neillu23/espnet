#!/usr/bin/env python3

# Copyright 2020 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import os
import sys
import subprocess
import re

slurp_dir = sys.argv[1]
slurp_mix_dir = sys.argv[2]
libritrans_mix_dir = sys.argv[3]

spk = {}
# Prepare slurp speaker 
with open(os.path.join(slurp_dir, "dataset", "slurp", "metadata" + ".json")) as meta:
    records = json.load(meta)
    for record in records.values():
        for filename in record["recordings"].keys():
            spk[filename[6:-5]] = record["recordings"][filename]["usrid"]

libritrans_dirs = {
    "train": ["tr"],
    "devel": ["cv"],
    "test": [],
    "test_qut": [],
}

slurp_dirs = {
    "train": ["tr_real","tr_synthetic"],
    "devel": ["cv"],
    "test": ["tt"],
    "test_qut": ["tt_qut"],
}



recordid_unique = {}
for subset in ["train", "devel", "test", "test_qut"]:
    odir = os.path.join("data", subset)
    os.makedirs(odir, exist_ok=True)

    with  open(os.path.join(odir, "text"), "w", encoding="utf-8") as text, open(
        os.path.join(odir, "spk1.scp"), "w") as spkscp, open(
        os.path.join(odir, "wav.scp"), "w") as wavscp, open(
        os.path.join(odir, "utt2spk"), "w"
    ) as utt2spk:


        for subdir in libritrans_dirs[subset]:
            mix_path = os.path.join(libritrans_mix_dir,subdir)
            meta_json = os.path.join(mix_path,"metadata.json")
            with open(meta_json, "r") as f:
                data = json.load(f)
                for dic in data:
                    uttid = "libritrans_" + dic["speech"][0]["source"]["file"].split("/")[-1][:-4]
                    speaker = dic["speech"][0]["source"]["spkid"]

                    # writing 
                    utt2spk.write("{} libritrans_{}\n".format(uttid, speaker))
                    text.write("{} {}\n".format(uttid, "dummy"))
                    spkscp.write("{} {}\n".format(uttid, os.path.join(mix_path, "s0_dry",dic["id"]+".wav")))
                    wavscp.write("{} {}\n".format(uttid, os.path.join(mix_path, "mixture",dic["id"]+".wav")))

        for subdir in slurp_dirs[subset]:
            mix_path = os.path.join(slurp_mix_dir,subdir)
            meta_json = os.path.join(mix_path,"metadata.json")
            with open(meta_json, "r") as f:
                data = json.load(f)
                for dic in data:
                    utt_name = dic["speech"][0]["source"]["file"].split("/")[-1]
                    recoid = utt_name[6:-5]
                    # skipped covered speech
                    if recoid in recordid_unique:
                        print("Already covered")
                        continue
                    elif subset in ["train", "devel"]:
                        recordid_unique[recoid] = 1

                    if subdir == "tr_synthetic":
                        speaker = "synthetic"
                    else:
                        speaker = spk[recoid]
                        
                    # writing 
                    uttid = "slurp_{}_{}".format(speaker, recoid)
                    utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
                    text.write("{} {}\n".format(uttid, "dummy"))
                    spkscp.write("{} {}\n".format(uttid, os.path.join(mix_path, "s0_dry",dic["id"]+".wav")))
                    wavscp.write("{} {}\n".format(uttid, os.path.join(mix_path, "mixture",dic["id"]+".wav")))



# for subset in ["train", "devel", "test"]:
#     odir = os.path.join("data", subset)
#     os.makedirs(odir, exist_ok=True)

#     with open(os.path.join(idir, "dataset", "slurp", subset + ".jsonl")) as meta, open(
#         os.path.join(odir, "text"), "w", encoding="utf-8"
#     ) as text, open(os.path.join(odir, "raw_wav.scp"), "w") as wavscp, open(
#         os.path.join(odir, "utt2spk"), "w"
#     ) as utt2spk:

#         for line in meta:
#             prompt = json.loads(line.strip())
#             transcript = prompt["sentence"]
#             transcript = transcript.replace("@", " at ")
#             transcript = transcript.replace("#", " hashtag ")
#             transcript = transcript.replace(",", "")
#             transcript = transcript.replace(".", "")
#             transcript = re.sub(" +", " ", transcript)
#             words = "{}".format(
#                 prompt["scenario"] + "_" + prompt["action"] + " " + transcript
#             ).replace("<unk>", "unknown")
#             for recording in prompt["recordings"]:
#                 recoid = recording["file"][6:-5]
#                 if recoid in recordid_unique:
#                     print("Already covered")
#                     continue
#                 recordid_unique[recoid] = 1
#                 wav = os.path.join(idir, "audio", "slurp_real", recording["file"])
#                 speaker = spk[recoid]
#                 uttid = "slurp_{}_{}".format(speaker, recoid)
#                 text.write("{} {}\n".format(uttid, words))
#                 utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
#                 wavscp.write("{} {}\n".format(uttid, wav))
#         if subset == "train":
#             meta = open(os.path.join(idir, "dataset", "slurp", "train_synthetic.jsonl"))
#             for line in meta:
#                 prompt = json.loads(line.strip())
#                 transcript = prompt["sentence"]
#                 transcript = transcript.replace("@", " at ")
#                 transcript = transcript.replace("#", " hashtag ")
#                 transcript = transcript.replace(",", "")
#                 transcript = transcript.replace(".", "")
#                 transcript = re.sub(" +", " ", transcript).lower()
#                 words = "{}".format(
#                     prompt["scenario"] + "_" + prompt["action"] + " " + transcript
#                 ).replace("<unk>", "unknown")
#                 for recording in prompt["recordings"]:
#                     recoid = recording["file"][6:-5]
#                     if recoid in recordid_unique:
#                         print("Already covered")
#                         continue
#                     recordid_unique[recoid] = 1
#                     wav = os.path.join(idir, "audio", "slurp_synth", recording["file"])
#                     speaker = "synthetic"
#                     uttid = "slurp_{}_{}".format(speaker, recoid)
#                     text.write("{} {}\n".format(uttid, words))
#                     utt2spk.write("{} slurp_{}\n".format(uttid, speaker))
#                     wavscp.write("{} {}\n".format(uttid, wav))
