#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

// DataLayer, 公开继承BasePrefetchingDataLayer
template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  // 虚方法， 一次导入batch_size大小的数据，之后进行DataTransformer变换
  virtual void load_batch(Batch<Dtype>* batch);
  
  // DataReader负责从硬盘读数据到一个队列， 之后提供给data_layer使用。
  // 计时并行运行多个solver，也只有一个线程来读数据， 这样可以确保顺
  // 序取数据. 不同的solver取到的数据不同.
  // DataReader没有bottom和top， 如果没有标签， blob数量为1, 有标签blob
  // 数量为2
  DataReader reader_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
