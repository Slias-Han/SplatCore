#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class TransferDirection { H2D, D2H, D2D };

enum class TransferPhase {
    INIT,
    RENDER_LOOP,
    TEST_ONLY,
    UNKNOWN
};

struct TransferEvent {
    std::string name;
    TransferDirection direction = TransferDirection::H2D;
    TransferPhase phase = TransferPhase::UNKNOWN;
    uint64_t bytes = 0;
    std::string sourceLocation;
    std::string notes;
    bool eliminatable = false;
};

class PcieTransferTracker {
public:
    static PcieTransferTracker& instance();

    void record(const TransferEvent& event);

    void setPhase(TransferPhase phase);
    TransferPhase currentPhase() const { return m_currentPhase; }

    void recordCopy(const char* name,
                    TransferDirection dir,
                    uint64_t bytes,
                    const char* file,
                    int line);

    void writeReport(const std::string& path) const;

    std::vector<TransferEvent> eliminatableCopies() const;

    void reset();

    void setFramesMeasured(uint32_t frames) { m_framesMeasured = frames; }
    uint32_t framesMeasured() const { return m_framesMeasured; }

    const std::vector<TransferEvent>& events() const { return m_events; }

private:
    PcieTransferTracker() = default;

    std::vector<TransferEvent> m_events;
    TransferPhase m_currentPhase = TransferPhase::UNKNOWN;
    uint32_t m_framesMeasured = 0;
};

#define PCIE_RECORD_H2D(name, bytes) \
    PcieTransferTracker::instance().recordCopy(name, TransferDirection::H2D, bytes, __FILE__, __LINE__)

#define PCIE_RECORD_D2H(name, bytes) \
    PcieTransferTracker::instance().recordCopy(name, TransferDirection::D2H, bytes, __FILE__, __LINE__)

#define PCIE_SET_PHASE(phase) \
    PcieTransferTracker::instance().setPhase(TransferPhase::phase)
