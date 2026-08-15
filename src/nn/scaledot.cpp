#include "scaledot.h"
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

Tensor ScaledDot::forward(const Tensor Q, const Tensor K, const Tensor V){
       // Use positive indices for transpose: transpose last two dimensions
       size_t ndim = Q.shape().size();
       size_t d1 = ndim - 2;
       size_t d2 = ndim - 1;
       Tensor QKV = Q.matmul(K.transpose_view(d1, d2));

       int64_t d_k = Q.shape()[Q.shape().size()-1];
       float scale = std::sqrt(static_cast<float>(d_k));
       // Build scale tensor on CPU first (safe), then move to Q's device.
       Tensor scale_tensor({1}, Q.dtype(), Device::CPU);
       scale_tensor.at<float>(0) = scale;
       if (Q.device() != Device::CPU) {
           scale_tensor = scale_tensor.toDevice(Q.device());
       }
       QKV = QKV / scale_tensor;

      return (QKV.softmax(Q.shape().size()-1).matmul(V));

}

std::vector<Tensor> ScaledDot::parameters(){
return {};}

void ScaledDot::zero_grad(){

}