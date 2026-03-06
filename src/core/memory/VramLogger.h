#pragma once
#include "MemorySystem.h"

namespace SplatCore {

class VramLogger {
public:
    // 初始化：打开 engine.log（追加模式），写入启动时间戳
    static void init(const char* logPath = "engine.log");

    // 关闭：flush 并关闭文件句柄
    static void shutdown();

    // 每帧调用：将 snapshot 格式化写入 log
    // 输出格式（精确）：
    // [Frame 000042] Static: 12.34 MB | Dynamic: 0.00 MB | Staging: 0.00 MB
    static void logFrame(const VramSnapshot& snapshot);

private:
    VramLogger() = delete;
};

} // namespace SplatCore
