#include "scaledot.h"
#include <cmath>

Tensor ScaledDot::forward(const Tensor Q, const Tensor K, const Tensor V){
       // Use positive indices for transpose: transpose last two dimensions
       size_t ndim = Q.shape().size();
       size_t d1 = ndim - 2;
       size_t d2 = ndim - 1;
       Tensor QKV = Q.matmul(K.transpose_view(d1, d2));

       int64_t d_k = Q.shape()[Q.shape().size()-1];  // or Q.shape()[2] for 3D
        float scale = std::sqrt(static_cast<float>(d_k));
        Tensor scale_tensor({1}, Q.dtype(), Q.device());
        scale_tensor.at<float>(0) = scale;
        QKV = QKV / scale_tensor;

      return (QKV.softmax(Q.shape().size()-1).matmul(V));

}

std::vector<Tensor> ScaledDot::parameters(){
return {};}

void ScaledDot::zero_grad(){

}