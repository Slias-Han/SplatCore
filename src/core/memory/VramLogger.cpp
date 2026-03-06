#include "VramLogger.h"

#include <cstdio>
#include <ctime>
#include <exception>

namespace SplatCore {

namespace {

static constexpr double BYTES_TO_MB = 1.0 / (1024.0 * 1024.0);
FILE* s_logFile = nullptr;

} // namespace

void VramLogger::init(const char* logPath)
{
    if (s_logFile != nullptr)
    {
        std::fflush(s_logFile);
        std::fclose(s_logFile);
        s_logFile = nullptr;
    }

    FILE* openedFile = nullptr;
#if defined(_WIN32)
    if (fopen_s(&openedFile, logPath, "a") != 0)
    {
        openedFile = nullptr;
    }
#else
    openedFile = std::fopen(logPath, "a");
#endif
    s_logFile = openedFile;
    if (s_logFile == nullptr)
    {
        std::terminate();
    }

    std::fprintf(s_logFile, "=== SplatCore Engine Log — VMA Memory Lifecycle v0.5 ===\n");

    const std::time_t now = std::time(nullptr);
    char timeBuffer[64] = {};
#if defined(_WIN32)
    std::tm localTm{};
    if (localtime_s(&localTm, &now) == 0)
    {
        std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", &localTm);
    }
#else
    std::tm localTm{};
    if (localtime_r(&now, &localTm) != nullptr)
    {
        std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", &localTm);
    }
#endif
    std::fprintf(s_logFile, "[Startup] %s\n", timeBuffer[0] == '\0' ? "unknown-time" : timeBuffer);
    std::fflush(s_logFile);
}

void VramLogger::shutdown()
{
    if (s_logFile != nullptr)
    {
        std::fflush(s_logFile);
        std::fclose(s_logFile);
        s_logFile = nullptr;
    }
}

void VramLogger::logFrame(const VramSnapshot& snapshot)
{
    if (s_logFile == nullptr)
    {
        return;
    }

    const double staticMb = static_cast<double>(snapshot.staticBytes) * BYTES_TO_MB;
    const double dynamicMb = static_cast<double>(snapshot.dynamicBytes) * BYTES_TO_MB;
    const double stagingMb = static_cast<double>(snapshot.stagingBytes) * BYTES_TO_MB;

    char lineBuffer[256] = {};
    std::snprintf(
        lineBuffer,
        sizeof(lineBuffer),
        "[Frame %06u] Static: %.2f MB | Dynamic: %.2f MB | Staging: %.2f MB\n",
        snapshot.frameIndex,
        staticMb,
        dynamicMb,
        stagingMb);
    std::fprintf(s_logFile, "%s", lineBuffer);
    std::fflush(s_logFile);
}

} // namespace SplatCore
