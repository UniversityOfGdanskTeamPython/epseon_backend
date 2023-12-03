#include "epseon/gpu/common.hpp"
#include <sstream>

#include "epseon/vulkan_headers.hpp"

namespace epseon {
    namespace gpu {
        namespace common {
            std::string vulkan_version_to_string(uint32_t version) {
                std::stringstream ss;
                ss << vk::apiVersionVariant(version) << "." << vk::apiVersionMajor(version) << "."
                   << vk::apiVersionMinor(version) << "." << vk::apiVersionPatch(version);
                return ss.str();
            }
        } // namespace common
    }     // namespace gpu
} // namespace epseon
