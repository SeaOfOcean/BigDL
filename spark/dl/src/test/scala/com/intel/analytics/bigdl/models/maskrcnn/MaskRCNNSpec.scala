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
import com.intel.analytics.bigdl.nn.Utils
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.Table
import org.scalatest.FlatSpec

import scala.io.Source

class MaskRCNNSpec extends FlatSpec {

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

  "MaskRCNN forward" should "work" in {
    val input = loadFeatures("data").transpose(2, 4).transpose(3, 4).contiguous()
    val model = MaskRCNN()

    model.forward(input)
    loadWeights(model)
    println(model.output.toTable(2))
  }

  def loadWeights(model: Module[Float]): Unit = {
    val modules = Utils.getNamedModules(model)
    modules.foreach(x => {
      val name = x._1
      val layer = x._2
      if (layer.getParametersTable() != null) {
        middleRoot = s"/home/jxy/data/maskrcnn/weights/$name"
        val pt = layer.getParametersTable()
        if (pt.contains(name)) {
          pt[Table](name).keySet.foreach(x => {
            val param = pt[Table](name)[Tensor[Float]](x)
            if (x != "gradWeight" && x != "gradBias") {

              val load = if (x == "weight") {
                if (layer.getClass.getCanonicalName.contains("BatchNorm")) {
                  loadFeatures("beta:0")
                } else {
                  var w = loadFeatures("kernel:0")
                  w = w.transpose(1, 4).transpose(2, 3).contiguous()
                  w
                }
              } else if (x == "bias") {
                if (layer.getClass.getCanonicalName.contains("BatchNorm")) {
                  loadFeatures("gamma:0")
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
              compareShape(param.size(), load.size())
              param.copy(load)
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
}
