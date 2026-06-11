#ifndef TENSOR_STORAGE
#define TENSOR_STORAGE

#include <vector>
#include <memory>
#include "../core/dtype.h"
#include "../core/allocators.h"

using namespace std;
class Storage {
public:
    explicit Storage(size_t bytes,const shared_ptr<Allocator>& all);
    ~Storage();

    static std::shared_ptr<Storage> allocate(size_t bytes, const shared_ptr<Allocator>& all);

    void* data() const {return data_;};
    size_t bytes() const {return bytes_;};
    Device device() const {return device_;};
    std::shared_ptr<Allocator>  allocator() const {return allocator_;};

private:
    void* data_;
    size_t bytes_;
    std::shared_ptr<Allocator> allocator_;
    Device device_;
};

#endif