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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

class ProposalMaskRcnn(preNmsTopNTest: Int, postNmsTopNTest: Int, val ratios: Array[Float],
  val scales: Array[Int], steps: Array[Int],
  rpnPreNmsTopNTrain: Int, rpnPostNmsTopNTrain: Int)(
  implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float] {


  override def updateOutput(input: Table): Tensor[Float] = {
    val scores = input(1)
    val anchors = input[Tensor[Float]](3)


  }


  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    gradInput = null
    gradInput
  }
}

object ProposalMaskRcnn {
  def apply(preNmsTopNTest: Int, postNmsTopNTest: Int,
    ratios: Array[Float],
    scales: Array[Int],
    steps: Array[Int],
    rpnPreNmsTopNTrain: Int = 12000, rpnPostNmsTopNTrain: Int = 2000)(
    implicit ev: TensorNumeric[Float]): ProposalMaskRcnn =
    new ProposalMaskRcnn(preNmsTopNTest, postNmsTopNTest, ratios, scales, steps,
      rpnPreNmsTopNTrain, rpnPostNmsTopNTrain)
}
