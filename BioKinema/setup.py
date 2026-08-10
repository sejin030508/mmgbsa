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

from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="biokinema",
    python_requires=">=3.10",
    version="0.2.0",
    description="BioKinema: a physically grounded generative model for continuous-time, "
    "all-atom biomolecular trajectories (built on Protenix / AlphaFold 3).",
    url="https://github.com/your-org/biokinema",
    packages=find_packages(
        exclude=(
            "assets",
            "benchmarks",
            "figures",
            "example",
            "example_runs",
            "*.egg-info",
        )
    ),
    include_package_data=True,
    data_files=["requirements.txt"],
    package_data={
        "protenix": ["model/layer_norm/kernel/*"],
    },
    install_requires=install_requires,
    license="Apache-2.0",
    platforms="manylinux1",
)
