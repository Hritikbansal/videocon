# coding=utf-8
# Copyright 2023 the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd 
import datasets
import json
from huggingface_hub import hf_hub_url

_INPUT_CSV = "videocon_human.csv"

_REPO_ID = "videocon/videocon"

class Dataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="TEST", version=VERSION, description="test"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                 {
                "video_url": datasets.Value("string")
                "caption": datasets.Value("string"),
                "neg_caption": datasets.Value("string"),
                "nle": datasets.Value("string"),
                "hard": datasets.Value("bool"),
                }
            ),
            task_templates=[],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        hf_auth_token = dl_manager.download_config.use_auth_token
        if hf_auth_token is None:
            raise ConnectionError(
                "Please set use_auth_token=True or use_auth_token='<TOKEN>' to download this dataset"
            )

        repo_id = _REPO_ID
        data_dir = dl_manager.download_and_extract({
            "examples_csv": hf_hub_url(repo_id=repo_id, repo_type='dataset', filename=_INPUT_CSV),
        })

        return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs=data_dir)]


    def _generate_examples(self, examples_csv, images_dir):
        """Yields examples."""
        df = pd.read_csv(examples_csv)
        for r_idx, r in df.iterrows():
            r_dict = r.to_dict()
            yield r_idx, r_dict