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

# ── Shader compilation ───────────────────────────────────────────────────
find_program(GLSLC_EXECUTABLE glslc
    HINTS "${Vulkan_GLSLC_EXECUTABLE}" ENV VULKAN_SDK
    REQUIRED)

set(SHADER_SRC_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_BIN_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SHADER_BIN_DIR}")

set(SHADER_SOURCES
    "${SHADER_SRC_DIR}/triangle.vert"
    "${SHADER_SRC_DIR}/triangle.frag"
)

set(SPV_OUTPUTS "")
foreach(SHADER IN LISTS SHADER_SOURCES)
    get_filename_component(SHADER_NAME "${SHADER}" NAME)
    set(SPV_OUT "${SHADER_BIN_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT  "${SPV_OUT}"
        COMMAND "${GLSLC_EXECUTABLE}" "${SHADER}" -o "${SPV_OUT}"
        DEPENDS "${SHADER}"
        COMMENT "glslc: ${SHADER_NAME} -> ${SHADER_NAME}.spv"
        VERBATIM
    )
    list(APPEND SPV_OUTPUTS "${SPV_OUT}")
endforeach()

add_custom_target(SplatCore_Shaders ALL DEPENDS ${SPV_OUTPUTS})
# ─────────────────────────────────────────────────────────────────────────

add_executable(SplatCore_SilasHan main.cpp)

target_link_libraries(SplatCore_SilasHan PRIVATE
    Vulkan::Vulkan
    glfw
)

add_dependencies(SplatCore_SilasHan SplatCore_Shaders)

# Passes the absolute .spv directory as a string macro to main.cpp.
# CMAKE_CURRENT_BINARY_DIR is always an absolute path, so this is
# safe for both single-config (Ninja/Make) and multi-config (MSVC).
target_compile_definitions(SplatCore_SilasHan PRIVATE
    "SHADER_DIR=\"${SHADER_BIN_DIR}/\""
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
