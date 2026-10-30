cmake_minimum_required(VERSION 3.24)
project(AdvancePlayer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(APPLE)
  enable_language(OBJCXX)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 13.0 CACHE STRING "")
endif()

# Platform-independent logic: builds everywhere, including the Windows CI job.
add_library(advp_portable STATIC
  core/src/rate.cpp
  core/src/schedule.cpp
  core/src/governor.cpp
  core/src/bench.cpp
  core/src/device_common.cpp
)
target_include_directories(advp_portable PUBLIC core/include PRIVATE core/src)

add_executable(advp_tests tests/core_tests.cpp)
target_link_libraries(advp_tests PRIVATE advp_portable)
enable_testing()
add_test(NAME core_tests COMMAND advp_tests)

# Media pipeline: needs our LGPL FFmpeg build (tools/build_ffmpeg.sh) and a
# GPU backend. Only the Metal backend exists so far.
set(ADVP_FFMPEG_DIR "${CMAKE_SOURCE_DIR}/third_party/ffmpeg" CACHE PATH "FFmpeg install prefix")
if(APPLE AND EXISTS "${ADVP_FFMPEG_DIR}/include/libavcodec/avcodec.h")
  # Embed the Metal shader sources; they are compiled at runtime, so the
  # build does not need Xcode's offline Metal compiler.
  set(ADVP_SHADER_HEADER "${CMAKE_BINARY_DIR}/generated/advp_shaders.h")
  set(ADVP_SHADER_TEXT "// Generated from core/shaders. Do not edit.\n#pragma once\n")
  foreach(shader IN ITEMS interpolate:kInterpolateShaderSource present:kPresentShaderSource)
    string(REPLACE ":" ";" parts "${shader}")
    list(GET parts 0 file)
    list(GET parts 1 symbol)
    file(READ "core/shaders/${file}.metal" source)
    string(APPEND ADVP_SHADER_TEXT
      "static const char ${symbol}[] = R\"ADVPMETAL(${source})ADVPMETAL\";\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "core/shaders/${file}.metal")
  endforeach()
  file(WRITE "${ADVP_SHADER_HEADER}" "${ADVP_SHADER_TEXT}")

  add_library(advp_media STATIC
    core/src/video_decoder.cpp
    core/src/audio_decoder.cpp
    core/src/link.cpp
    core/src/convert.cpp
    core/src/apple/metal_backend.mm
    core/src/apple/device_apple.mm
    core/src/apple/player_mac.mm
    core/src/apple/audio_output.mm
  )
  target_include_directories(advp_media
    PUBLIC core/include
    PRIVATE core/src "${CMAKE_BINARY_DIR}/generated" "${ADVP_FFMPEG_DIR}/include")
  set_source_files_properties(core/src/apple/metal_backend.mm core/src/apple/device_apple.mm
    core/src/apple/player_mac.mm core/src/apple/audio_output.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_link_directories(advp_media PUBLIC "${ADVP_FFMPEG_DIR}/lib")
  target_link_libraries(advp_media PUBLIC
    advp_portable
    avformat avcodec avutil swscale swresample
    "-framework Foundation" "-framework AppKit" "-framework Metal"
    "-framework CoreVideo" "-framework CoreMedia" "-framework VideoToolbox"
    "-framework QuartzCore" "-framework ImageIO" "-framework UniformTypeIdentifiers"
    "-framework AudioToolbox" "-framework CoreAudio")

  add_executable(advp cli/main.cpp)
  target_link_libraries(advp PRIVATE advp_media)
  set_target_properties(advp PROPERTIES BUILD_RPATH "${ADVP_FFMPEG_DIR}/lib")
else()
  message(STATUS "Media pipeline skipped (needs macOS and ${ADVP_FFMPEG_DIR})")
endif()
