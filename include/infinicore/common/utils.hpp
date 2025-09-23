#include <stdexcept>

#include <spdlog/spdlog.h>

#include "infinicore.h"

#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)

#define INFINICORE_CHECK_ERROR(call)                                                           \
    do {                                                                                       \
        spdlog::info("Entering `" #call "` at `" __FILE__ ":" STRINGIZE(__LINE__) "`.");       \
        int ret = (call);                                                                      \
        spdlog::info("Exiting `" #call "` at `" __FILE__ ":" STRINGIZE(__LINE__) "`.");        \
        if (ret != INFINI_STATUS_SUCCESS) {                                                    \
            throw std::runtime_error(#call " failed with error code: " + std::to_string(ret)); \
        }                                                                                      \
    } while (false)
