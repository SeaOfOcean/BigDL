#
# Copyright 2016 The BigDL Authors.
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
#

import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import *
from bigdl.transform.vision.image import ImageFrame

if sys.version >= '3':
    long = int
    unicode = str

class Dataset(JavaValue):

    def __init__(self, data, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        if isinstance(data, ImageFrame):
            return callBigDlFunc(self.bigdl_type, "createDatasetFromImageFrame", data)


    def transform(self, transformer):
        return callBigDlFunc(self.bigdl_type, "transformDataset", transformer)