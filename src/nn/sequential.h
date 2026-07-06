#ifndef SEQUENTIAL
#define SEQUENTIAL

#include "module.h"
#include <memory>
#include <vector>

class Sequential : public Module {
public:
    Sequential() = default;
    void add(std::shared_ptr<Module> layer);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    void zero_grad() override;
    
private:
    std::vector<std::shared_ptr<Module>> layers_;
};

#endif