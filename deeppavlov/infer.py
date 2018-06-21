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

from pathlib import Path
import sys
from datetime import datetime, timedelta
import time
import shutil

p = (Path(__file__) / ".." / "..").resolve()
sys.path.append(str(p))

from deeppavlov.core.commands.train import build_model_from_config
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.utils import get_project_root

log = get_logger(__name__)

CONFIG_PATH = str(get_project_root()) + '/deeppavlov/configs/odqa/generic_ranker.json'

PERIOD = timedelta(minutes=5)
NEXT_TIME = datetime.now() + PERIOD
MINUTES = 0


def check_data_changed():
    global NEXT_TIME
    global MINUTES
    if NEXT_TIME <= datetime.now():
        MINUTES += 1
        NEXT_TIME += PERIOD
        return True
    else:
        NEXT_TIME = datetime.now() + PERIOD
        MINUTES = 0
        return False


def build_chainer(check_time):
    config = read_json(CONFIG_PATH)
    save_path = Path(config['dataset_reader']['save_path']).parent
    db_copy_path = save_path / 'data_copy.db'
    db_path = save_path / 'data.db'

    tfidf_path = save_path / 'tfidf.npz'
    tfidf_copy_path = save_path / 'tfidf_copy.npz'

    lock_path = save_path / '.lock'
    if check_time:
        if check_data_changed():
            if not lock_path.exists():
                if db_copy_path.exists() and tfidf_copy_path.exists():
                    shutil.copy(str(db_copy_path), str(db_path))
                    shutil.copy(str(tfidf_copy_path), str(tfidf_path))
                    db_copy_path.unlink()
                    config = read_json(CONFIG_PATH)
                    chainer = build_model_from_config(config)
                    return chainer
    else:
        if not lock_path.exists():
            if db_copy_path.exists() and tfidf_copy_path.exists():
                shutil.copy(str(db_copy_path), str(db_path))
                shutil.copy(str(tfidf_copy_path), str(tfidf_path))
                db_copy_path.unlink()
                config = read_json(CONFIG_PATH)
                chainer = build_model_from_config(config)
                return chainer


if __name__ == "__main__":
    try:
        _chainer = build_chainer(check_time=False)
    except Exception:
        time.sleep(2)
        _chainer = build_chainer(check_time=False)
    while True:
        try:
            new_chainer = build_chainer(check_time=True)
            if new_chainer:
                _chainer = new_chainer
            query = input("Question: ")
            print()
            context = _chainer([query.strip()])[0][0]
            print(context)
        except Exception:
            continue
