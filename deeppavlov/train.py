"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from pathlib import Path
import sys
import os
import time

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.utils import get_project_root, is_empty
from deeppavlov.core.common.file import read_json, save_json
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="Path to folder with data files.", type=str)
parser.add_argument("output_path", help="Path to a folder with trained data.", type=str)
args = parser.parse_args()
input_path = Path(args.input_path).resolve()
output_path = Path(args.output_path).resolve()
output_path.mkdir(parents=True, exist_ok=True)

TEMPLATE_CONFIG_PATH = str(get_project_root()) + '/deeppavlov/configs/odqa/ru_ranker_template.json'
NEW_CONFIG_PATH = str(get_project_root()) + '/deeppavlov/configs/odqa/generic_ranker.json'

DIR_LEN = 0


def generate_config(template_path, db_path, tfidf_path):
    config = read_json(template_path)
    config['dataset_reader']['data_path'] = str(input_path)
    db_path = os.path.join(output_path, db_path)
    config['dataset_reader']['save_path'] = config['dataset_iterator']['load_path'] = \
        config['chainer']['pipe'][1]['load_path'] = db_path

    try:
        os.remove(db_path)
    except OSError:
        pass

    config['chainer']['pipe'][0]['vectorizer']['save_path'] = \
        config['chainer']['pipe'][0]['vectorizer'][
            'load_path'] = os.path.join(output_path, tfidf_path)

    return config


def train():
    train_config = generate_config(TEMPLATE_CONFIG_PATH, 'data_copy.db', 'tfidf_copy.npz')
    train_evaluate_model_from_config(train_config, pass_config=True)
    log.info("Successfully trained and stored result in {}".format(output_path))


if __name__ == "__main__":

    infer_config = generate_config(TEMPLATE_CONFIG_PATH, 'data.db', 'tfidf.npz')
    save_json(infer_config, NEW_CONFIG_PATH)

    lorem_ipsum_path = input_path / 'lorem_ipsum.txt'

    while True:
        if is_empty(input_path):
            print("No files in input path.")
            with open(lorem_ipsum_path, 'w') as fout:
                fout.write("Lorem ipsum")
            time.sleep(10)
            continue
        else:
            if lorem_ipsum_path.exists():
                lorem_ipsum_path.unlink()
            # stamp = max((f.stat().st_mtime, f) for f in input_path.iterdir())[0]
            dir_len = len(list(input_path.iterdir()))
            if dir_len > DIR_LEN:
                DIR_LEN = dir_len
                log.info("Locked.")
                lock_path = Path(output_path / '.lock')
                lock_path.touch(mode=0o777)
                train()
                lock_path.unlink()
                log.info("Sleeping...\n")
            time.sleep(10)
            continue
