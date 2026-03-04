cmake_minimum_required(VERSION 3.22)
project(SplatCore_SilasHan)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_definitions(
    VK_USE_PLATFORM_WIN32_KHR
    NOMINMAX
)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(SplatCore_SilasHan main.cpp)

target_link_libraries(SplatCore_SilasHan PRIVATE
    Vulkan::Vulkan
    glfw
)

# Validation layers only in Debug
target_compile_definitions(SplatCore_SilasHan PRIVATE
    $<$<CONFIG:Debug>:VK_ENABLE_VALIDATION_LAYERS>
)

# Compiler warnings and optimisation flags
if(MSVC)
    target_compile_options(SplatCore_SilasHan PRIVATE
        /W4
        $<$<CONFIG:Debug>:/Od /Zi /RTC1>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
else()
    # GCC / Clang
    target_compile_options(SplatCore_SilasHan PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:Release>:-O2 -DNDEBUG>
    )
endif()
