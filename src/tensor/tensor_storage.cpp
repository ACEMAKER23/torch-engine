#include "tensor_storage.h"
#include <cstdlib>
#include <memory>

std::shared_ptr<Storage> Storage::allocate(size_t bytes,const shared_ptr<Allocator>& all){
    return std::make_shared<Storage>(bytes, all);
};

Storage::Storage(size_t bytes,const shared_ptr<Allocator>& all)
: bytes_(bytes), allocator_(all), device_(all->device())
{
    data_=allocator_->allocate(bytes);
};

Storage::~Storage(){
    allocator_->deallocate(data_);
}

