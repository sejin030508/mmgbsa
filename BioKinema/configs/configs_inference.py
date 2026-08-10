# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Attribution-NonCommercial 4.0 International
# License (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the
# License at

#     https://creativecommons.org/licenses/by-nc/4.0/

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0114
from protenix.config.extend_types import ListValue, RequiredValue

inference_configs = {
    "seeds": ListValue([101]),
    "dump_dir": "./output",
    "input_file": RequiredValue(str),
    "coarse_frame_num": 20,
    "coarse_interval": 5,
    "fine_frame_num": 5,
    "W_H": 1,
    "W_G": 10,
    "steps_per_stage": 0,
    "history_noise": 0.0,
    "history_t": 0.0,
    "conformation_sampling": False,
    "conformation_chunk_size": 25,
    "need_atom_confidence": False,
    "input_json_path": RequiredValue(str),
    "load_checkpoint_path": RequiredValue(str),
    "num_workers": 16,
}
