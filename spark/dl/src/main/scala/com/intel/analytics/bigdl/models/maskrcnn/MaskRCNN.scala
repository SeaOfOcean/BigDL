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

package com.intel.analytics.bigdl.models.maskrcnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

object MaskRCNN {
  // The strides of each layer of the FPN Pyramid. These values
  // are based on a Resnet101 backbone.
  val BACKBONE_STRIDES = Array(4, 8, 16, 32, 64)
  val IMAGE_MAX_DIM = 1024

  // Input image size
  val IMAGE_SHAPE = Array(IMAGE_MAX_DIM, IMAGE_MAX_DIM, 3)

  // Length of square anchor side in pixels
  val RPN_ANCHOR_SCALES = Array(32, 64, 128, 256, 512)

  // Ratios of anchors at each cell (width/height)
  // A value of 1 represents a square anchor, and 0.5 is a wide anchor
  val RPN_ANCHOR_RATIOS = Array[Float](0.5f, 1, 2)

  // Anchor stride
  // If 1 then anchors are created for each cell in the backbone feature map.
  // If 2, then anchors are created for every other cell, and so on.
  val RPN_ANCHOR_STRIDE = 1

  // Compute backbone size from input image size

  val BACKBONE_SHAPES = BACKBONE_STRIDES.map(stride => {
    ((IMAGE_SHAPE(1) / stride).ceil, (IMAGE_SHAPE(2) / stride).ceil)
  })

  // Pooled ROIs
  val POOL_SIZE = 7
  val MASK_POOL_SIZE = 14

  /**
   * Builds the Region Proposal Network.
   * It wraps the RPN graph so it can be used multiple times with shared weights.
   *
   * @param anchorStride Controls the density of anchors. Typically 1 (anchors for
   * every pixel in the feature map), or 2 (every other pixel).
   * @param anchorsPerLocation number of anchors per pixel in the feature map
   * @param depth Depth of the backbone feature map.
   */
  def buildRpnModel(anchorStride: Int, anchorsPerLocation: Int, depth: Int): Module[Float] = {
    val featureMap = Input()
    val shared = SpatialConvolution(256, 512, 3, 3, anchorStride, anchorStride)
      .setName("rpn_conv_shared").inputs(featureMap)
    // Anchor Score. [batch, height, width, anchors per location * 2].
    var x = SpatialConvolution(512, 2 * anchorsPerLocation, 1, 1).setName("rpn_class_raw")
      .inputs(shared)
    // Reshape to [batch, anchors, 2]
    val rpnClassLogits = InferReshape(Array(0, -1, 2)).inputs(x)
    // Softmax on last dimension of BG/FG.
    val rpnProbs = SoftMax().setName("rpn_class_xxx").inputs(rpnClassLogits)
    // Bounding box refinement. [batch, H, W, anchors per location, depth]
    // where depth is [x, y, log(w), log(h)]
    x = SpatialConvolution(512, anchorsPerLocation * 4, 1, 1).setName("rpn_bbox_pred")
      .inputs(shared)
    // Reshape to [batch, anchors, 4]
    val rpnBbox = InferReshape(Array(0, -1, 4)).inputs(x)

    Graph(input = featureMap, output = Array(rpnClassLogits, rpnProbs, rpnBbox))
  }

  def apply(NUM_CLASSES: Int = 81): Module[Float] = {
    val data = Input()
    val imInfo = Input()
    val resnet = ResNet.graph(1000,
      T("shortcutType" -> ShortcutType.B,
        "depth" -> 101, "dataset" -> DatasetType.ImageNet)).asInstanceOf[Graph[Float]]
    val conv1 = resnet.node("conv1")
    data -> conv1
    val c1 = resnet.node("pool1")
    val c2 = resnet.node("res2c_out")
    val c3 = resnet.node("res3d_out")
    val c4 = resnet.node("res4w_out")
    val c5 = resnet.node("res5c_out")
//    Graph(input, Array(c1, c2, c3, c4, c5))

    val p5_add = SpatialConvolution(256, 256, 1, 1).setName("fpn_c5p5").inputs(c5)
    val fpn_p5upsampled = UpSampling2D(Array(2, 2)).setName("fpn_p5upsampled").inputs(p5_add)
    val fpn_c4p4 = SpatialConvolution(256, 256, 1, 1).setName("fpn_c4p4").inputs(c4)
    val p4_add = CAddTable().inputs(fpn_p5upsampled, fpn_c4p4)

    val fpn_p4upsampled = UpSampling2D(Array(2, 2)).setName("fpn_p4upsampled").inputs(p4_add)
    val fpn_c3p3 = SpatialConvolution(256, 256, 1, 1).setName("fpn_c3p3").inputs(c3)
    val p3_add = CAddTable().inputs(fpn_p4upsampled, fpn_c3p3)

    val fpn_p3upsampled = UpSampling2D(Array(2, 2)).setName("fpn_p3upsampled").inputs(p3_add)
    val fpn_c2p2 = SpatialConvolution(256, 256, 1, 1).setName("fpn_c2p2").inputs(c2)
    val p2_add = CAddTable().inputs(fpn_p3upsampled, fpn_c2p2)

    // Attach 3x3 conv to all P layers to get the final feature maps.
    val p2 = SpatialConvolution(256, 256, 3, 3).setName("fpn_p2").inputs(p2_add)
    val p3 = SpatialConvolution(256, 256, 3, 3).setName("fpn_p3").inputs(p3_add)
    val p4 = SpatialConvolution(256, 256, 3, 3).setName("fpn_p4").inputs(p4_add)
    val p5 = SpatialConvolution(256, 256, 3, 3).setName("fpn_p5").inputs(p5_add)
    // P6 is used for the 5th anchor scale in RPN. Generated by
    // subsampling from P5 with stride of 2.
    val p6 = SpatialMaxPooling(1, 1, 2, 2).setName("fpn_p6").inputs(p5)
    // Note that P6 is used in RPN, but not in the classifier heads.
    val rpn_feature_maps = Array(p2, p3, p4, p5, p6)
    val mrcnn_feature_maps = Array(p2, p3, p4, p5)

    val priorBoxes = rpn_feature_maps.indices.map(i => {
      PriorBox(Array(RPN_ANCHOR_SCALES(i)), _aspectRatios = RPN_ANCHOR_RATIOS,
        imgSize = 1, step = BACKBONE_STRIDES(i), isFlip = false, offset = 0)
        .inputs(rpn_feature_maps(i))
    }).toArray

    val anchors = JoinTable(2, 3).inputs(priorBoxes)

    // RPN Model
    val rpn = buildRpnModel(RPN_ANCHOR_STRIDE, RPN_ANCHOR_RATIOS.length, 256)
    val mapTable = MapTable(rpn).inputs(rpn_feature_maps)

    // Concatenate layer outputs
    // Convert from list of lists of level outputs to list of lists
    // of outputs across levels.
    // e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    val rpn_class_logits = JoinTable(2, 2).inputs(select(RPN_ANCHOR_RATIOS.length, 1, mapTable))
    val rpn_class = JoinTable(2, 2).inputs(select(RPN_ANCHOR_RATIOS.length, 2, mapTable))
    val rpn_bbox = JoinTable(2, 2).inputs(select(RPN_ANCHOR_RATIOS.length, 3, mapTable))

    val rpn_rois = Proposal(6000, 1000).inputs(rpn_class, rpn_bbox, imInfo, anchors)

    val (mrcnn_class_logits, mrcnn_class, mrcnn_bbox) =
      fpnClassifierGraph(rpn_rois, mrcnn_feature_maps, IMAGE_SHAPE, POOL_SIZE, NUM_CLASSES)

    val detections = DetectionOutputFrcnn().inputs(rpn_rois, mrcnn_class, mrcnn_bbox, imInfo)
    // TODO: fix it
    val detection_boxes = detections
    val mrcnn_mask = buildFpnMaskGraph(detection_boxes, mrcnn_feature_maps,
      IMAGE_SHAPE,
      MASK_POOL_SIZE,
      NUM_CLASSES)

    Graph(Array(data, imInfo), Array(detections, mrcnn_class, mrcnn_bbox, mrcnn_mask,
      rpn_rois, rpn_class, rpn_bbox)).setName("mask_rcnn")
  }

  private def select(total: Int, dim: Int, input: ModuleNode[Float]): Array[ModuleNode[Float]] = {
    require(dim >= 1 && dim <= 3)
    (1 to total).map(i => {
      val level = SelectTable(i).inputs(input)
      SelectTable(dim).inputs(level)
    }).toArray
  }

  /**
   * Builds the computation graph of the feature pyramid network classifier
   * and regressor heads.
   *
   * @param rois [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized coordinates.
   * @param featureMaps List of feature maps from diffent layers of the pyramid,
   * [P2, P3, P4, P5]. Each has a different resolution.
   * @param imageShape : [height, width, depth]
   * @param poolSize : The width of the square feature map generated from ROI Pooling.
   * @param numClasses : number of classes, which determines the depth of the results
   * @return
   * logits: [N, NUM_CLASSES] classifier logits (before softmax)
   * probs: [N, NUM_CLASSES] classifier probabilities
   * bbox_deltas: [N, (dy, dx, log(dh), log(dw))] Deltas to apply to proposal boxes
   */
  def fpnClassifierGraph(rois: ModuleNode[Float], featureMaps: Array[ModuleNode[Float]],
    imageShape: Array[Int], poolSize: Int, numClasses: Int)
  : (ModuleNode[Float], ModuleNode[Float], ModuleNode[Float]) = {
    // ROI Pooling
    // Shape: [batch, num_boxes, pool_height, pool_width, channels]
    var x = PyramidROIAlign(poolSize, poolSize,
      imgH = imageShape(0), imgW = imageShape(1), imgC = imageShape(2))
      .inputs(Array(rois) ++ featureMaps)
    // Two 1024 FC layers (implemented with Conv2D for consistency)
    x = SpatialConvolution(4096, 1024, poolSize, poolSize).setName("mrcnn_class_conv1").inputs(x)
    x = SpatialBatchNormalization(1024).setName("mrcnn_class_bn1").inputs(x)
    x = ReLU(true).inputs(x)
    x = SpatialConvolution(1024, 1024, 1, 1).setName("mrcnn_class_conv2").inputs(x)
    x = SpatialBatchNormalization(1024).setName("mrcnn_class_bn2").inputs(x)
    x = ReLU(true).inputs(x)
    val shared = Squeeze(2).inputs(Squeeze(3).inputs(x))
    // Classifier head
    val mrcnn_class_logits = Linear(1024, numClasses).setName("mrcnn_class_logits").inputs(shared)
    val mrcnn_probs = SoftMax().inputs(mrcnn_class_logits)

    // BBox head
    // [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = Linear(1024, numClasses * 4).setName("mrcnn_bbox_fc").inputs(shared)
    // Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    val mrcnn_bbox = InferReshape(Array(0, numClasses, 4)).setName("mrcnn_bbox").inputs(x)
    (mrcnn_class_logits, mrcnn_probs, mrcnn_bbox)
  }

  /**
   *
   * """Builds the computation graph of the mask head of Feature Pyramid Network.
   * *
   * rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
   * coordinates.
   * feature_maps: List of feature maps from diffent layers of the pyramid,
   * [P2, P3, P4, P5]. Each has a different resolution.
   * image_shape: [height, width, depth]
   * pool_size: The width of the square feature map generated from ROI Pooling.
   * num_classes: number of classes, which determines the depth of the results
   * *
   * Returns: Masks [batch, roi_count, height, width, num_classes]
   * """
   *
   * @param rois
   * @param featureMaps
   * @param imageShape
   * @param poolSize
   * @param num_classes
   */
  def buildFpnMaskGraph(rois: ModuleNode[Float], featureMaps: Array[ModuleNode[Float]],
    imageShape: Array[Int], poolSize: Int, num_classes: Int): ModuleNode[Float] = {

    // ROI Pooling
    // Shape: [batch, boxes, pool_height, pool_width, channels]
    var x = PyramidROIAlign(poolSize, poolSize,
      imgH = imageShape(0), imgW = imageShape(1), imgC = imageShape(2)).setName("roi_align_mask")
      .inputs(Array(rois) ++ featureMaps)

    // Conv layers
    x = SpatialConvolution(1024, 256, 3, 3).setName("mrcnn_mask_conv1").inputs(x)
    x = SpatialBatchNormalization(256).setName("mrcnn_mask_bn1").inputs(x)
    x = ReLU(true).inputs(x)

    x = SpatialConvolution(256, 256, 3, 3).setName("mrcnn_mask_conv2").inputs(x)
    x = SpatialBatchNormalization(256).setName("mrcnn_mask_bn2").inputs(x)
    x = ReLU(true).inputs(x)


    x = SpatialConvolution(256, 256, 3, 3).setName("mrcnn_mask_conv3").inputs(x)
    x = SpatialBatchNormalization(256).setName("mrcnn_mask_bn3").inputs(x)
    x = ReLU(true).inputs(x)

    x = SpatialConvolution(256, 256, 3, 3).setName("mrcnn_mask_conv4").inputs(x)
    x = SpatialBatchNormalization(256).setName("mrcnn_mask_bn4").inputs(x)
    x = ReLU(true).inputs(x)

    // TODO: not sure
//    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
//      name="mrcnn_mask_deconv")(x)
//    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
//      name="mrcnn_mask")(x)
    x
  }

}
