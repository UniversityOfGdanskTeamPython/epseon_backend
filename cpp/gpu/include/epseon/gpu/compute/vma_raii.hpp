#pragma once
#include "epseon/vulkan_headers.hpp"

namespace vma::raii {
    class Allocator : public vma::Allocator { // NOLINT: hicpp-special-member-functions
      public:
        using vma::Allocator::Allocator;

        ~Allocator() {
            this->destroy();
        }
    };
} // namespace vma::raii