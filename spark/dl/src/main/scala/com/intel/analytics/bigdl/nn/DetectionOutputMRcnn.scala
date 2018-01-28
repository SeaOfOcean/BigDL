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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.Table

class DetectionOutputMRcnn(confidence: Double = 0.7)(
  implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Activity, Float] {

  override def updateOutput(input: Table): Activity = {
    val rois = input[Tensor[Float]](1)
    val mrcnn_class = input[Tensor[Float]](2)
    val mrcnn_bbox = input[Tensor[Float]](3)
    val image_meta = input[Tensor[Float]](4)
    val (_, _, window, _) = parseImageMeta(image_meta)
    refineDetection(rois, mrcnn_class, mrcnn_bbox, window)
    output
  }


  override def updateGradInput(input: Table, gradOutput: Activity): Table = {
    gradInput = null
    gradInput
  }


  private def refineDetection(rois: Tensor[Float], probs: Tensor[Float],
    deltas: Tensor[Float], window: Tensor[Float]) = {

    // class id for each roi
    val (classScores, classIds) = probs.topk(1, 2, increase = false)
    val deltaSpecific = Tensor[Float](classIds.nElement(), 4)
    (1 to classIds.nElement()).foreach(i => {
      deltaSpecific(i).copy(deltas(i).select(2, classIds.valueAt(i, 1).toInt))
    })
    deltaSpecific.narrow(2, 1, 2).mul(0.1f)
    deltaSpecific.narrow(2, 3, 2).mul(0.2f)
    BboxUtil.scaleBBox(deltaSpecific, 1024, 1024)
    // Clip boxes to image window
    val boxes = BboxUtil.clipToWindows(window, deltaSpecific)
    boxes.apply1(x => x.round)
    // Filter out background boxes
    val keep = classIds.storage().array().zip(classScores.storage().array()).zip(Stream.from(1))
      .filter(x => x._1._1 > 1 && x._1._2 > confidence).map(_._2)

  }


  private def parseImageMeta(meta: Tensor[Float])
  : (Tensor[Float], Tensor[Float], Tensor[Float], Tensor[Float]) = {
    val id = meta.narrow(2, 1, 1)
    val imageShape = meta.narrow(2, 2, 4)
    val window = meta.narrow(2, 5, 4)
    val activeClassIds = meta.narrow(2, 9, meta.size(2) - 9)
    (id, imageShape, window, activeClassIds)
  }
}

object DetectionOutputMRcnn {
  def apply()(implicit ev: TensorNumeric[Float]): DetectionOutputMRcnn = new DetectionOutputMRcnn()
}
