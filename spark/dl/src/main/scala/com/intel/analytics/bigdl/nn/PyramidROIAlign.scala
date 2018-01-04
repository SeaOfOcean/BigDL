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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}

@SerialVersionUID(-1562995431845030993L)
class PyramidROIAlign(poolH: Int, poolW: Int, imgH: Int, imgW: Int, imgC: Int)
  (implicit ev: TensorNumeric[Float]) extends AbstractModule[Table, Tensor[Float], Float] {
  val resize = ResizeBilinear(poolH, poolW)
  val concat = JoinTable(1, 4)
  val concat2 = JoinTable[Int](1, 2)
  override def updateOutput(input: Table): Tensor[Float] = {
    // Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    val boxes = input[Tensor[Float]](1).squeeze(1)
    require(boxes.dim() == 2 && boxes.size(2) == 4, "boxes should be batchxNx4 tensor," +
      s" while actual is ${boxes.size().mkString("x")}")
    val channelDim = 2
    // Assign each ROI to a level in the pyramid based on the ROI area
    val splits = boxes.split(2)
    val y1 = splits(0)
    val x1 = splits(1)
    val y2 = splits(2)
    val x2 = splits(3)
    // todo: optimize
    val h = y2 - y1
    val w = x2 - x1
    // Equation 1 in the Feature Pyramid Networks paper. Account for
    // the fact that our coordinates are normalized here.
    // e.g. a 224x224 ROI (in pixels) maps to P4
    val imageArea = imgH * imgW
    val roiLevel = log2((h.clone().cmul(w)).sqrt() / (224.0f / Math.sqrt(imageArea).toFloat))
    roiLevel.apply1(x => {
      if (x.equals(Float.NaN)) Float.NegativeInfinity else Math.round(x)
    })
    roiLevel.apply1(x => {
      Math.min(5, Math.max(2, 4 + x.round))
    })
    roiLevel.squeeze()
    // Loop through levels and apply ROI pooling to each. P2 to P5.
    var i = 1
    var level = 2
    val boxToLevel = T()
    val pooledTable = T()
    while (i <= 4) {
      val ix = (1 to roiLevel.nElement()).filter(roiLevel.valueAt(_) == level).toArray
      if (ix.length > 0) {
        boxToLevel.insert(Tensor[Int](Storage(ix)).resize(ix.length, 1))
      }

      val cropResize = Tensor[Float](ix.length, poolH, poolW,
        input[Tensor[Float]](2).size(channelDim))
      val featureMap = input[Tensor[Float]](i + 1).transpose(2, 3).transpose(3, 4).contiguous()
//        println(featureMap.size().mkString("x"))
      // todo: hdim, wdim
      val hdim = 2
      val wdim = 3
      ix.zip(Stream from(1)).foreach(ind => {
        val box = boxes(ind._1)
        val height = featureMap.size(hdim)
        val width = featureMap.size(wdim)
        var hsOff = (height - 1) * box.valueAt(1)
        var heOff = (height - 1) * (1 - box.valueAt(3))
        var wsOff = (width - 1) * box.valueAt(2)
        var weOff = (width - 1) * (1 - box.valueAt(4))
        if (hsOff + heOff > height || weOff + wsOff > width) {
          hsOff = 0
          heOff = 0
          wsOff = 0
          weOff = 0
        }
        val crop = Cropping2D(Array(hsOff.toInt, heOff.toInt),
          Array(wsOff.toInt, weOff.toInt), DataFormat.NHWC)
        crop.forward(featureMap)
//        println(crop.output.size().mkString("x"))
        resize.forward(crop.output)
//        println(resize.output.size().mkString("x"))
//        println(i)
//        println(cropResize.size().mkString("x"))
        cropResize(ind._2).copy(resize.output.squeeze(1))
      })
      if (cropResize.nElement() > 0) pooledTable.insert(cropResize)
      println(cropResize.size().mkString("x"))




      // Crop and Resize
      // From Mask R-CNN paper: "We sample four regular locations, so
      // that we can evaluate either max or average pooling. In fact,
      // interpolating only a single value at each bin center (without
      // pooling) is nearly as effective."
      //
      // Here we use the simplified approach of a single value per bin,
      // which is how it's done in tf.crop_and_resize()
      // Result: [batch * num_boxes, pool_height, pool_width, channels]

      i += 1
      level += 1
    }
    // Pack pooled features into one tensor
    val pooled = concat.forward(pooledTable).asInstanceOf[Tensor[Float]]
    println(boxToLevel)
    val boxToLevels = concat2.forward(boxToLevel)
    println(pooled.size().mkString("x"))
    val ix = boxToLevels.squeeze()
      .toArray().asInstanceOf[Array[Int]].zipWithIndex.sortBy(_._1).map(_._2)
    println(ix.mkString(","))

    // Feature Maps. List of feature maps from different level of the
    // feature pyramid. Each is [batch, height, width, channels]

    i = 0


    output.resizeAs(pooled)
    while (i < ix.length) {
      i += 1
      output(i).copy(pooled(ix(i - 1) + 1))
    }
    // Assign each ROI to a level in the pyramid based on the ROI area.
    output.resize(1, output.size(1), output.size(2), output.size(3), output.size(4))
  }

  override def updateGradInput(input: Table, gradOutput: Tensor[Float]): Table = {
    throw new NotImplementedError()
  }

  private def log2(x: Tensor[Float]): Tensor[Float] = {
    x.log().div(Math.log(2).toFloat)
  }
}

object PyramidROIAlign {
  def apply(poolH: Int, poolW: Int, imgH: Int, imgW: Int, imgC: Int)
    (implicit ev: TensorNumeric[Float]): PyramidROIAlign =
    new PyramidROIAlign(poolH, poolW, imgH, imgW, imgC)
}
