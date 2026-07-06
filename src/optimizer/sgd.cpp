#include "sgd.h"

void sgd::step(float learningRate){
    for(Tensor& t : processing_){
        if (t.requiredGrad() && t.grad()){
                    // Create scalar tensor for learning rate
            Tensor lr_tensor({1}, t.dtype(), t.device());
            lr_tensor.at<float>(0) = learningRate;
            
            // Update: parameter = parameter - learning_rate * gradient
            t = t - (*t.grad() * lr_tensor);
        }
    }
}