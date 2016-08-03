#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  // 显式建构子， 避免潜在（implicit）类型转换错误
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
 
  // 虚方法。该函数只被BasePrefetchingDataLayer类overrides
  // 做layer-specific setup
  // 参数: 
  //  bottom:the reshaped input blobs, blob的data fields存储着该层的输入数据
  //  top: allocated but unshaped output blobs.
  //  该方法只做one-time layer specific setups. 设置top blobs的shapes, 设置Reshape方法的
  //  internal buffers. Reshape方法会在每一次forward pass的时候调整top blob 的sizes.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
  // Data layers should be shared by multiple solvers in parallel
  // 该函数返回一个bool类型的值， 表示该数据层要被多个网络并行共享.
  //（a layer should be shared by multiple nets during data parallelism）
  //  By default， 除了data layers, 其余所有层不会被共享 
  virtual inline bool ShareInParallel() const { return true; }
  
  // 虚方法，做DataLayerSetUp的设置
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
      
  // Data layers have no bottoms, so reshaping is trivial.
  // 数据层没有bottom， 故而没有Reshape一说
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  
  // Using the CPU device, compute the gradients for any parameters and for
  // the bottom blobs if propagate_down is true
  // 虚方法，在CPU中计算关于该层参数的梯度和并根据propagate_down是否为true觉得是否计算
  // 该层输入数据（bottom blobs）的梯度
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  
  // Using the GPU device, compute the gradients for any parameters and for 
  // the bottom blobs if propagate_down is true. Fall back to Backward_cpu() if unavailable
  // 虚方法, 使用GPU， 计算该层参数的梯度信息和并根据propagate_down决定是否计算输入数
  // 据(bottom blobs)的梯度信息 
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected: // Protected Attributes
  // Applies common transformations to the input data, such as scaling, mirroring, substracting the image mean...
  // 对该数据层的输入数据进行相应的变换， 比如尺度变换， 去均值， cropping等操作
  // 参见caffe.proto
  TransformationParameter transform_param_;
  
  // 是DataTransformer类型的shared_ptr
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  
  // 是否有标签，无标签可以是无监督学习
  bool output_labels_;
};

// 类Batch是和批相关的类， 只是把2个数据结构封装成1个
// 包含数据和其数据(data)对应的标签（label）
template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

//类BasePrefetchingDataLayer继承自BaseDataLayer, InternalThread.
//其中InternalThread是封装了线程， 通过InternalThreadEntry来执行
//线程函数， 用一个单独的线程函数来取数据 
template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
   // 显式建构子 
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  // 默认为3
  static const int PREFETCH_COUNT = 3;

 protected:
  //虚方法，是线程执行的函数，用来取数据的线程
  virtual void InternalThreadEntry();
  
  // 纯虚方法，由派生类实现该方法，用于载入batch size大小的数据和label
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  // 阻塞队列1：从free队列取数据结构，填充数据结构到full队列
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  // 阻塞队列2：从full队列取数据，使用数据结构，清空数据结构，
  // 放到free队列
  BlockingQueue<Batch<Dtype>*> prefetch_full_;

  // 用来辅助实现图片变换操作
  Blob<Dtype> transformed_data_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
