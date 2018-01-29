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
import com.intel.analytics.bigdl.transform.vision.image.label.roi.RoiLabel
import com.intel.analytics.bigdl.transform.vision.image.util.BboxUtil
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class DetectionOutputMRcnn(confidence: Double = 0.7, nmsThresh: Float = 0.3f,
  DETECTION_MAX_INSTANCES: Int = 100)(
  implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float] {

  override def updateOutput(input: Table): Tensor[Float] = {
    val rois = input[Tensor[Float]](1)
    val mrcnn_class = input[Tensor[Float]](2)
    val mrcnn_bbox = input[Tensor[Float]](3)
    val image_meta = input[Tensor[Float]](4)
    val (_, _, window, _) = parseImageMeta(image_meta)
    refineDetection(rois, mrcnn_class, mrcnn_bbox, window)
    // Pad with zeros if detections < DETECTION_MAX_INSTANCES
    val gap = DETECTION_MAX_INSTANCES - output.size(1)
    require(gap >= 0)
    if (gap > 0) {
      val origin = output.clone()
      output.resize(1, DETECTION_MAX_INSTANCES, 6).zero()
      if (origin.size(1) > 0) output.narrow(1, 1, origin.size(1)).copy(origin)
    }

    output
  }


  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    gradInput = null
    gradInput
  }


  def refineDetection(rois: Tensor[Float], probs: Tensor[Float],
    deltas: Tensor[Float], window: Tensor[Float]): Tensor[Float] = {
    deltas.squeeze(2)
    // class id for each roi
    val (classScores, classIds) = probs.topk(1, 2, increase = false)
    val deltaSpecific = Tensor[Float](classIds.nElement(), 4)
    (1 to classIds.nElement()).foreach(i => {
      deltaSpecific(i).copy(deltas(i)(classIds.valueAt(i, 1).toInt))
    })
    deltaSpecific.narrow(2, 1, 2).mul(0.1f)
    deltaSpecific.narrow(2, 3, 2).mul(0.2f)
    val refinedRois = BboxUtil.bboxTransformInv(rois, deltaSpecific, true)
    BboxUtil.scaleBBox(refinedRois, 1024, 1024)
    // Clip boxes to image window
    val boxes = BboxUtil.clipToWindows(window, refinedRois)
    boxes.apply1(x => x.round)
    // Filter out background boxes
    var keep = classIds.storage().array().zip(classScores.storage().array()).zip(Stream.from(1))
      .filter(x => x._1._1 > 1 && x._1._2 > confidence).map(_._2)
    // Apply per-class NMS
    val preNmsClassIds = BboxUtil.selectTensor(classIds, keep, 1)
    val preNmsScores = BboxUtil.selectTensor(classScores, keep, 1)
    val preNmsRois = BboxUtil.selectTensor(boxes, keep, 1)
    val classNum = probs.size(2)
    var i = 2
    val nmsKeep = new mutable.HashSet[Int]()
    while (i <= classNum) {
      if (preNmsClassIds.storage().array().contains(i.toFloat)) {
        val ixs = (1 to preNmsClassIds.nElement()).filter(k => {
          preNmsClassIds.valueAt(k, 1) == i
        }).toArray
        val scores = BboxUtil.selectTensor(preNmsScores, ixs, 1).squeeze()
        val bboxes = BboxUtil.selectTensor(preNmsRois, ixs, 1).squeeze()
        val resultIndices = new Array[Int](ixs.length)
        val num = nmsTool.nms(scores, bboxes, nmsThresh, resultIndices, false)
        (0 until(num)).foreach(n => {
          val elem = keep(ixs(resultIndices(n) - 1) - 1)
          nmsKeep.add(elem)
        })
        println(nmsKeep)

//        postProcessOneClass(preNmsClassIds, preNmsScores, preNmsRois, i)
      }
      i += 1
    }
    keep = nmsKeep.toArray.sorted
    if (keep.length > 0) {
      println("keep.length", keep.length)
      val (sortedScore, sortedIds) =
        BboxUtil.selectTensor(classScores, keep, 1).topk(keep.length, 1, false)
      val topIds = if (keep.length > DETECTION_MAX_INSTANCES) {
        sortedIds.narrow(1, 1, DETECTION_MAX_INSTANCES)
      } else {
        sortedIds
      }
      keep = (1 to topIds.nElement()).map(i => {
        keep(topIds.valueAt(i, 1).toInt - 1)
      }).toArray
      val finalRois = BboxUtil.selectTensor(refinedRois, keep, 1)
      val finalClassIds = BboxUtil.selectTensor(classIds, keep, 1)
      val finalScores = BboxUtil.selectTensor(classScores, keep, 1)
      output.resize(finalRois.size(1), 6)
      output.narrow(2, 1, 4).copy(finalRois)
      output.narrow(2, 5, 1).copy(finalClassIds)
      output.narrow(2, 6, 1).copy(finalScores)
    } else {
      output.resize(0, 6)
    }
  }

  var nmsTool: Nms = new Nms()
//  private def postProcessOneClass(classIds: Tensor[Float],
//    scores: Tensor[Float], boxes: Tensor[Float],
//    clsInd: Int): RoiLabel = {
//    val inds = (1 to scores.size(1)).filter(ind =>
//      scores.valueAt(ind, clsInd + 1) > thresh).toArray
//    if (inds.length == 0) return null
//    val clsScores = BboxUtil.selectTensor(scores.select(2, clsInd + 1), inds, 1)
//    val clsBoxes = BboxUtil.selectTensor(boxes.narrow(2, clsInd * 4 + 1, 4), inds, 1)
//
//    val keepN = nmsTool.nms(clsScores, clsBoxes, nmsThresh, inds)
//
//    val bboxNms = selectTensor(clsBoxes, inds, 1, keepN)
//    val scoresNms = selectTensor(clsScores, inds, 1, keepN)
//    if (bboxVote) {
//      if (areas == null) areas = Tensor[Float]
//      BboxUtil.bboxVote(scoresNms, bboxNms, clsScores, clsBoxes, areas)
//    } else {
//      RoiLabel(scoresNms, bboxNms)
//    }
//  }


  private def parseImageMeta(meta: Tensor[Float])
  : (Tensor[Float], Tensor[Float], Tensor[Float], Tensor[Float]) = {
    val id = meta.narrow(2, 1, 1)
    val imageShape = meta.narrow(2, 2, 4)
    val window = meta.narrow(2, 5, 4).squeeze()
    val activeClassIds = meta.narrow(2, 9, meta.size(2) - 9)
    (id, imageShape, window, activeClassIds)
  }
}

object DetectionOutputMRcnn {
  def apply()(implicit ev: TensorNumeric[Float]): DetectionOutputMRcnn = new DetectionOutputMRcnn()
}
