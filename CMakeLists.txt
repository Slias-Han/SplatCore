cmake_minimum_required(VERSION 3.22)

# Platform guards added for Linux CI compatibility (ubuntu-latest / llvmpipe)
# Windows MSVC remains the primary build target.
# All Win32-specific constructs are wrapped in if(WIN32) blocks.
if(WIN32)
    set(CMAKE_VS_PLATFORM_NAME "x64")
    if(NOT DEFINED CMAKE_GENERATOR_PLATFORM)
        set(CMAKE_GENERATOR_PLATFORM "x64" CACHE STRING "" FORCE)
    endif()
endif()

project(SplatCore_SilasHan)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32)
    add_compile_definitions(
        VK_USE_PLATFORM_WIN32_KHR
        NOMINMAX
    )
elseif(UNIX AND NOT APPLE)
    add_compile_definitions(VK_USE_PLATFORM_XCB_KHR)
    # xcb is the correct Vulkan surface extension for Linux CI (llvmpipe supports it)
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm CONFIG REQUIRED)

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
    "${SHADER_SRC_DIR}/cube.vert"
    "${SHADER_SRC_DIR}/cube.frag"
    "${SHADER_SRC_DIR}/point.vert"
    "${SHADER_SRC_DIR}/point.frag"
)

set(SHADER_OUTPUTS "")
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
    list(APPEND SHADER_OUTPUTS "${SPV_OUT}")
endforeach()

add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/shaders/vram_poison_comp.spv"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
    COMMAND "${GLSLC_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/shaders/vram_poison.comp"
        -o "${CMAKE_BINARY_DIR}/shaders/vram_poison_comp.spv"
    DEPENDS "${CMAKE_SOURCE_DIR}/shaders/vram_poison.comp"
    VERBATIM
)
list(APPEND SHADER_OUTPUTS "${CMAKE_BINARY_DIR}/shaders/vram_poison_comp.spv")

add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/shaders/sha256_compute.spv"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
    COMMAND "${GLSLC_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/shaders/sha256_compute.comp"
        -o "${CMAKE_BINARY_DIR}/shaders/sha256_compute.spv"
    DEPENDS "${CMAKE_SOURCE_DIR}/shaders/sha256_compute.comp"
    VERBATIM
)
list(APPEND SHADER_OUTPUTS "${CMAKE_BINARY_DIR}/shaders/sha256_compute.spv")

add_custom_target(SplatCore_Shaders ALL DEPENDS ${SHADER_OUTPUTS})
# ─────────────────────────────────────────────────────────────────────────

add_executable(SplatCore_SilasHan
    main.cpp
    src/core/memory/MemorySystem.cpp
    src/core/memory/VramLogger.cpp
    src/tests/HashProbe.cpp
    src/tests/PoisonTestHarness.cpp
    third_party/vma/vma_impl.cpp
)

if(MSVC)
    set_source_files_properties(third_party/vma/vma_impl.cpp PROPERTIES
        COMPILE_OPTIONS "/wd4100;/wd4189;/wd4324")
else()
    set_source_files_properties(third_party/vma/vma_impl.cpp PROPERTIES COMPILE_OPTIONS "-w")
endif()

target_include_directories(SplatCore_SilasHan PRIVATE
    third_party/vma
)

target_link_libraries(SplatCore_SilasHan PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
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

enable_testing()

add_executable(SplatCore_MemoryLifecycleTests
    tests/memory/test_vram_lifecycle.cpp
    src/core/memory/MemorySystem.cpp
    third_party/vma/vma_impl.cpp
)

target_include_directories(SplatCore_MemoryLifecycleTests PRIVATE
    third_party/vma
    .
)

target_link_libraries(SplatCore_MemoryLifecycleTests PRIVATE
    Vulkan::Vulkan
    ${Vulkan_LIBRARIES}
)

if(MSVC)
    target_link_libraries(SplatCore_MemoryLifecycleTests PRIVATE
        $<$<CONFIG:Debug>:ucrtd.lib>
        $<$<CONFIG:Debug>:vcruntimed.lib>
        $<$<CONFIG:Debug>:msvcrtd.lib>
        $<$<CONFIG:Release>:ucrt.lib>
        $<$<CONFIG:Release>:vcruntime.lib>
        $<$<CONFIG:Release>:msvcrt.lib>
    )
endif()

if(WIN32)
    target_link_libraries(SplatCore_MemoryLifecycleTests PRIVATE
        kernel32.lib
        user32.lib
    )
endif()

if(MSVC)
    set(WINSDK_LIB "$ENV{WindowsSdkDir}Lib/$ENV{WindowsSDKVersion}um/x64")
    target_link_directories(SplatCore_MemoryLifecycleTests PRIVATE
        "${WINSDK_LIB}"
    )
endif()

if(MSVC)
    target_compile_options(SplatCore_MemoryLifecycleTests PRIVATE
        /W4
        $<$<CONFIG:Debug>:/Od /Zi /RTC1>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
else()
    target_compile_options(SplatCore_MemoryLifecycleTests PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:Release>:-O2 -DNDEBUG>
    )
endif()

add_test(NAME memory_vram_lifecycle
         COMMAND SplatCore_MemoryLifecycleTests)

add_executable(SplatCore_PoisonTests
    tests/poison/test_vram_poison.cpp
    src/tests/HashProbe.cpp
    src/tests/PoisonTestHarness.cpp
    src/core/memory/MemorySystem.cpp
    src/core/memory/VramLogger.cpp
    third_party/vma/vma_impl.cpp
)

target_include_directories(SplatCore_PoisonTests PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    third_party/vma
    src
    .
)

target_link_libraries(SplatCore_PoisonTests PRIVATE
    Vulkan::Vulkan
    ${Vulkan_LIBRARIES}
    glfw
    glm::glm
)

if(MSVC)
    target_link_libraries(SplatCore_PoisonTests PRIVATE
        $<$<CONFIG:Debug>:ucrtd.lib>
        $<$<CONFIG:Debug>:vcruntimed.lib>
        $<$<CONFIG:Debug>:msvcrtd.lib>
        $<$<CONFIG:Release>:ucrt.lib>
        $<$<CONFIG:Release>:vcruntime.lib>
        $<$<CONFIG:Release>:msvcrt.lib>
    )
endif()

if(WIN32)
    target_link_libraries(SplatCore_PoisonTests PRIVATE
        kernel32.lib
        user32.lib
    )
endif()

target_compile_definitions(SplatCore_PoisonTests PRIVATE
    "SHADER_DIR=\"${SHADER_BIN_DIR}/\""
    $<$<CONFIG:Debug>:VK_ENABLE_VALIDATION_LAYERS>
)

if(MSVC)
    target_compile_options(SplatCore_PoisonTests PRIVATE
        /W4
        $<$<CONFIG:Debug>:/Od /Zi /RTC1>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
else()
    target_compile_options(SplatCore_PoisonTests PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:Release>:-O2 -DNDEBUG>
    )
endif()

add_dependencies(SplatCore_PoisonTests SplatCore_Shaders)

add_test(NAME vram_poison_test
    COMMAND SplatCore_PoisonTests
            "${CMAKE_SOURCE_DIR}/tests/assets/poison_test_scene.ply"
            "${CMAKE_BINARY_DIR}/shaders"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_executable(SplatCore_SHA256Tests
    tests/sha256/test_sha256_consistency.cpp
    src/tests/HashProbe.cpp
    src/tests/PoisonTestHarness.cpp
    src/core/memory/MemorySystem.cpp
    src/core/memory/VramLogger.cpp
    third_party/vma/vma_impl.cpp
)

target_include_directories(SplatCore_SHA256Tests PRIVATE
    ${Vulkan_INCLUDE_DIRS}
    third_party/vma
    src
    .
)

target_link_libraries(SplatCore_SHA256Tests PRIVATE
    Vulkan::Vulkan
    ${Vulkan_LIBRARIES}
    glfw
    glm::glm
)

if(MSVC)
    target_link_libraries(SplatCore_SHA256Tests PRIVATE
        $<$<CONFIG:Debug>:ucrtd.lib>
        $<$<CONFIG:Debug>:vcruntimed.lib>
        $<$<CONFIG:Debug>:msvcrtd.lib>
        $<$<CONFIG:Release>:ucrt.lib>
        $<$<CONFIG:Release>:vcruntime.lib>
        $<$<CONFIG:Release>:msvcrt.lib>
    )
endif()

if(WIN32)
    target_link_libraries(SplatCore_SHA256Tests PRIVATE
        kernel32.lib
        user32.lib
    )
endif()

target_compile_definitions(SplatCore_SHA256Tests PRIVATE
    "SHADER_DIR=\"${SHADER_BIN_DIR}/\""
    $<$<CONFIG:Debug>:VK_ENABLE_VALIDATION_LAYERS>
)

if(MSVC)
    target_compile_options(SplatCore_SHA256Tests PRIVATE
        /W4
        $<$<CONFIG:Debug>:/Od /Zi /RTC1>
        $<$<CONFIG:Release>:/O2 /DNDEBUG>
    )
else()
    target_compile_options(SplatCore_SHA256Tests PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-O0 -g>
        $<$<CONFIG:Release>:-O2 -DNDEBUG>
    )
endif()

add_dependencies(SplatCore_SHA256Tests SplatCore_Shaders)

add_test(NAME sha256_consistency_test
    COMMAND SplatCore_SHA256Tests
            "${CMAKE_SOURCE_DIR}/tests/assets/poison_test_scene.ply"
            "${CMAKE_BINARY_DIR}/shaders"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
