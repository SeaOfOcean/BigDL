/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.keras.KerasBaseSpec
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

class PyramidROIAlignSpec extends KerasBaseSpec {
  "PyramidRoiAlign forward" should "work properly" in {
    val rois = Tensor[Float](T(0.07630829, 0.77991879, 0.43840923, 0.72346518,
      0.97798951, 0.53849587, 0.50112046, 0.07205113,
      0.26843898, 0.4998825, 0.67923, 0.80373904,
      0.38094113, 0.06593635, 0.2881456, 0.90959353,
      0.21338535, 0.45212396, 0.93120602, 0.02489923)).resize(1, 5, 4)
    val channel = 256
    val p2 = Tensor[Float](1, channel, 256, 256).randn()
    val p3 = Tensor[Float](1, channel, 128, 128).randn()
    val p4 = Tensor[Float](1, channel, 64, 64).randn()
    val p5 = Tensor[Float](1, channel, 32, 32).randn()
    val input = T(rois, p2, p3, p4, p5)
    val layer = new PyramidROIAlign(7, 7, 400, 500, 3)
    layer.forward(input)
  }
}
