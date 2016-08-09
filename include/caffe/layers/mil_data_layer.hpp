#ifndef MIL_DATA_LAYER_HPP_
#define MIL_DATA_LAYER_HPP_

#include "hdf5.h"

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/data_layer.hpp"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {
/**
 * @brief Uses a text file which specifies the image names, and a hdf5 file for the labels.
 *        Note that each image can have multiple positive labels.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MILDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MILDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MILDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top); // this method are herrited from data_layer

  virtual inline const char* type() const { return "MILData"; } // inherrits from layer 
  virtual inline int ExactNumBottomBlobs() const { return 0; } // data layer, so no bottom blobs
  virtual inline int ExactNumTopBlobs() const { return 2; }   // contains data and labels

 protected:
  virtual unsigned int PrefetchRand(); // its own method

  // virtual void InternalThreadEntry(); // inheritted from internal thread
  
  virtual void load_batch(Batch<Dtype>*batch); // inherited from BasePrefetchingDataLayer(a pure virtual function) 

  int num_images_;
  unsigned int counter_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector< std::pair<std::string, std::string > > image_database_;
  hid_t label_file_id_; // hid_t is a type in hdf5 and keep track of reference node for file.
  
  vector<float> mean_value_; // vector of float values   
  Blob<Dtype> label_blob_;  // label_blob_ is a blob
};

}  // namespace caffe

#endif  // MIL_DATA_LAYER_HPP_
