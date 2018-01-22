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

import java.io.{File, FileNotFoundException}
import java.nio.file.Paths

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.{Graph, Input, Utils}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table
import org.scalatest.{FlatSpec, Matchers}

import scala.io.Source

class MaskRCNNSpec extends FlatSpec with Matchers {

  def loadFeaturesFullName(s: String, hasSize: Boolean = true,
    middleRoot: String = middleRoot): Tensor[Float] = {
    loadFeaturesFullPath(Paths.get(middleRoot, s).toString, hasSize)
  }

  def loadFeaturesFullPath(s: String, hasSize: Boolean = true): Tensor[Float] = {
//    println(s"load $s from file")

    if (hasSize) {
      val size = s.substring(s.lastIndexOf("-") + 1, s.lastIndexOf("."))
        .split("_").map(x => x.toInt)
      Tensor(Storage(Source.fromFile(s).getLines()
        .map(x => x.toFloat).toArray)).reshape(size)
    } else {
      Tensor(Storage(Source.fromFile(s).getLines()
        .map(x => x.toFloat).toArray))
    }
  }

  var middleRoot = "/home/jxy/data/maskrcnn/weights/"

  def loadFeatures(s: String, middleRoot: String = middleRoot)
  : Tensor[Float] = {
    if (s.contains(".txt")) {
      loadFeaturesFullName(s, hasSize = true, middleRoot)
    } else {
      val list = new File(middleRoot).listFiles()
      list.foreach(x => {
        if (x.getName.matches(s"$s-.*txt")) {
          return loadFeaturesFullName(x.getName, hasSize = true, middleRoot)
        }
      })
      throw new FileNotFoundException(s"cannot map $s")
    }
  }

  "compare resnet" should "work" in {
    val input = Tensor[Float](1, 3, 128, 128).fill(1)

    val model = MaskRCNN().evaluate()
    val layers = Utils.getNamedModules(model)

    loadWeights(model, "/home/jxy/data/maskrcnn/weights3/")

    val out = model.forward(input).toTable
//    middleRoot = "/home/jxy/data/maskrcnn/weights3/C1"
//    val expected = loadFeatures("C1")
//    toHWC(out[Tensor[Float]](1)).contiguous().map(expected, (a, b) => {
//      assert(Math.abs(a - b) < 1e-5); a
//    })

    middleRoot = "/home/jxy/data/maskrcnn/weights3/C2"
    var expected2 = loadFeatures("C2")
    var outout = toHWC(out[Tensor[Float]](2)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5); a
    })


    middleRoot = "/home/jxy/data/maskrcnn/weights3/C3"
    expected2 = loadFeatures("C3")
    outout = toHWC(out[Tensor[Float]](3)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5); a
    })


    middleRoot = "/home/jxy/data/maskrcnn/weights3/C4"
    expected2 = loadFeatures("C4")
    outout = toHWC(out[Tensor[Float]](4)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5); a
    })


    middleRoot = "/home/jxy/data/maskrcnn/weights3/C5"
    expected2 = loadFeatures("C5")
    outout = toHWC(out[Tensor[Float]](5)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5); a
    })
//    layers.foreach(x => {
//      println(s"=========================================================, ${x._1}")
//      println(toHWC(x._2.output.toTensor))
//      if (x._1.contains("SpatialZeroPadding")) {
//        middleRoot = "/home/jxy/data/maskrcnn/weights2/padding"
//        val expected = loadFeatures("padding")
//        toHWC(x._2.output.toTensor).contiguous() should be (expected)
//      }
//      if (x._1 == "conv1") {
//        middleRoot = "/home/jxy/data/maskrcnn/weights2/conv1"
//        val expected = loadFeatures("conv1")
//        toHWC(x._2.output.toTensor).contiguous().map(expected, (a, b) => {
//          assert(Math.abs(a - b) < 1e-5); a
//        })
//      }
//      if (x._1 == "relu") {
//        middleRoot = "/home/jxy/data/maskrcnn/weights2/relu"
//        val expected = loadFeatures("relu")
//        toHWC(x._2.output.toTensor).contiguous().map(expected, (a, b) => {
//          assert(Math.abs(a - b) < 1e-5); a
//        })
//      }
//    })
//    println(model.output)
  }

  "compare feature map" should "work" in {
    val input = Tensor[Float](1, 3, 128, 128).fill(1)

    val model = MaskRCNN().evaluate()

    loadWeights(model, "/home/jxy/data/maskrcnn/weights5/")

    val out = model.forward(input).toTable
//    middleRoot = "/home/jxy/data/maskrcnn/weights3/C1"
//    val expected = loadFeatures("C1")
//    toHWC(out[Tensor[Float]](1)).contiguous().map(expected, (a, b) => {
//      assert(Math.abs(a - b) < 1e-5); a
//    })

    middleRoot = "/home/jxy/data/maskrcnn/weights5/p2"
    var expected2 = loadFeatures("p2")
    var outout = toHWC(out[Tensor[Float]](1)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5); a
    })

    middleRoot = "/home/jxy/data/maskrcnn/weights5/p3"
    expected2 = loadFeatures("p3")
    outout = toHWC(out[Tensor[Float]](2)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5); a
    })


    middleRoot = "/home/jxy/data/maskrcnn/weights5/p4"
    expected2 = loadFeatures("p4")
    outout = toHWC(out[Tensor[Float]](3)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5); a
    })

    middleRoot = "/home/jxy/data/maskrcnn/weights5/p5"
    expected2 = loadFeatures("p5")
    outout = toHWC(out[Tensor[Float]](4)).contiguous()
    outout.map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5); a
    })

  }


  "compare convblock" should "work" in {
    val input = Tensor[Float](1, 3, 128, 128).fill(1)

    val in = Input[Float]()
    val convBlock = MaskRCNN.convBlock(3, in, 3, Array(64, 64, 256), stage = 2,
      block = 'a', strides = (1, 1))

    val model = Graph(in, convBlock).evaluate()

    loadWeights(model, "/home/jxy/data/maskrcnn/weights4/")

    val out = model.forward(input)

    middleRoot = "/home/jxy/data/maskrcnn/weights3/x0"
    val expected2 = loadFeatures("x0")
    toHWC(out.toTensor).contiguous().map(expected2, (a, b) => {
      assert(Math.abs(a - b) < 1e-5); a
    })

//    middleRoot = "/home/jxy/data/maskrcnn/weights3/C21"
//    val expected21 = loadFeatures("C21")
//    toHWC(out[Tensor[Float]](6)).contiguous().map(expected21, (a, b) => {
//      assert(Math.abs(a - b) < 1e-5); a
//    })
//
//    middleRoot = "/home/jxy/data/maskrcnn/weights3/C2"
//    val expected2 = loadFeatures("C2")
//    toHWC(out[Tensor[Float]](2)).contiguous().map(expected2, (a, b) => {
//      assert(Math.abs(a - b) < 1e-5); a
//    })
//    layers.foreach(x => {
//      println(s"=========================================================, ${x._1}")
//      println(toHWC(x._2.output.toTensor))
//      if (x._1.contains("SpatialZeroPadding")) {
//        middleRoot = "/home/jxy/data/maskrcnn/weights2/padding"
//        val expected = loadFeatures("padding")
//        toHWC(x._2.output.toTensor).contiguous() should be (expected)
//      }
//      if (x._1 == "conv1") {
//        middleRoot = "/home/jxy/data/maskrcnn/weights2/conv1"
//        val expected = loadFeatures("conv1")
//        toHWC(x._2.output.toTensor).contiguous().map(expected, (a, b) => {
//          assert(Math.abs(a - b) < 1e-5); a
//        })
//      }
//      if (x._1 == "relu") {
//        middleRoot = "/home/jxy/data/maskrcnn/weights2/relu"
//        val expected = loadFeatures("relu")
//        toHWC(x._2.output.toTensor).contiguous().map(expected, (a, b) => {
//          assert(Math.abs(a - b) < 1e-5); a
//        })
//      }
//    })
//    println(model.output)
  }

  "MaskRCNN forward" should "work" in {
    val input = loadFeatures("data").transpose(2, 4).transpose(3, 4).contiguous()
    val model = MaskRCNN()

    loadWeights(model)
    model.forward(input)
    val out = model.forward(input)
    middleRoot = "/home/jxy/data/maskrcnn/weights/rpn_class_logits"
    var expected = loadFeatures("rpn_class_logits")
    toHWC(out.toTable[Tensor[Float]](1)).contiguous().map(expected, (a, b) => {
      assert(Math.abs(a - b) < 1e-5);
      a
    })
//    middleRoot = "/home/jxy/data/maskrcnn/weights2/C2"
//    expected = loadFeatures("C2")
//    toHWC(out.toTable[Tensor[Float]](2)).contiguous().map(expected, (a, b) => {
//      assert(Math.abs(a - b) < 1e-5); a
//    })
  }

  def loadWeights(model: Module[Float], root: String = "/home/jxy/data/maskrcnn/weights/"): Unit = {
    val modules = Utils.getNamedModules(model)
    modules.foreach(x => {
      val name = x._1
      val layer = x._2
      if (layer.getParametersTable() != null) {
        middleRoot = root + s"$name"
        println(s"load for $middleRoot")
        val pt = layer.getParametersTable()
        if (pt.contains(name)) {
          pt[Table](name).keySet.foreach(x => {
            val param = pt[Table](name)[Tensor[Float]](x)
            if (x != "gradWeight" && x != "gradBias") {

              val load = if (x == "weight") {
                if (layer.getClass.getCanonicalName.contains("BatchNorm")) {
                  loadFeatures("gamma:0")
                } else {
                  var w = loadFeatures("kernel:0")
//                  w = toCHW(w).contiguous()
                  w = w.transpose(1, 3).transpose(2, 4).transpose(1, 2).contiguous()
                  w
                }
              } else if (x == "bias") {
                if (layer.getClass.getCanonicalName.contains("BatchNorm")) {
                  loadFeatures("beta:0")
                } else {
                  loadFeatures("bias:0")
                }
              } else if (x == "runningMean") {
                loadFeatures("moving_mean:0")
              } else {
                // if (x == "runningVar") {
                require(x == "runningVar")
                loadFeatures("moving_variance:0")
              }
              if (param.nElement() > 0) {
                println(s"load $name $x..............................")
                compareShape(param.size(), load.size())
                param.copy(load)
              }
            }
          })
        }
      }
    })
  }

  def compareShape(size1: Array[Int], size2: Array[Int]): Unit = {
    val s = if (size1.length != size2.length) {
      size1.slice(1, 5)
    } else size1
    s.zip(size2).foreach(x => {
      if (x._1 != x._2) {
        println(s"compare ${size1.mkString("x")} with ${size2.mkString("x")}")
        return
      }
    })
  }

  def toHWC(tensor: Tensor[Float]): Tensor[Float] = {
    require(tensor.dim() == 4)
    tensor.transpose(2, 3).transpose(3, 4)
  }

  def toCHW(tensor: Tensor[Float]): Tensor[Float] = {
    require(tensor.dim() == 4)
    tensor.transpose(3, 4).transpose(2, 3)
  }
}
