
#pragma once

#include <Arduino.h>
#include <Preferences.h>
#include <esp_system.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <stddef.h>
#include <string.h>

namespace EdgeStore {

static constexpr uint32_t MAGIC = 0x50454447UL;
static constexpr uint32_t VERSION = 1;
static constexpr uint8_t CAPACITY = 12;

enum State : uint8_t {
  STARTUP = 0,
  SAFE = 1,
  DANGER = 2
};

struct SensorData {
  float temperature;
  float current;
  float vibration;
  bool valid;
};

struct Alert {
  SensorData sensors;
  float risk;
  uint32_t deviceId;
  uint32_t bootId;
  uint32_t eventId;
  char mode[12];
  char date[11];
  char time[9];
};

struct Blob {
  uint32_t magic;
  uint32_t version;
  uint32_t deviceId;
  uint32_t nextEventId;
  uint8_t state;
  uint8_t count;
  uint8_t reserved[2];
  Alert events[CAPACITY];
  uint32_t checksum;
};

class PersistentOutbox {
 public:

  bool begin() {
    lock_ = xSemaphoreCreateMutex();
    if (!lock_) return false;

    if (!prefs_.begin("prededge", false)) {
      return false;
    }

    size_t length = prefs_.getBytesLength("outbox");

    if (length == 0 && !prefs_.isKey("outbox")) {
      Blob fresh = {};

      fresh.magic = MAGIC;
      fresh.version = VERSION;
      fresh.deviceId = esp_random();
      fresh.nextEventId = 1;
      fresh.state = STARTUP;
      fresh.count = 0;

      if (!save(fresh)) return false;

      data_ = fresh;

      Serial.printf(
        "[STORE] New NVS outbox | device=%lu\n",
        (unsigned long)data_.deviceId
      );

      return true;
    }

    if (length != sizeof(Blob)) {
      Serial.printf(
        "[STORE] ERROR: incompatible NVS blob (%u vs %u bytes)\n",
        (unsigned)length,
        (unsigned)sizeof(Blob)
      );

      return false;
    }

    Blob loaded = {};

    if (
      prefs_.getBytes(
        "outbox",
        &loaded,
        sizeof(Blob)
      ) != sizeof(Blob)
    ) {
      Serial.println("[STORE] ERROR: NVS read failed");
      return false;
    }

    if (
      loaded.magic != MAGIC ||
      loaded.version != VERSION ||
      loaded.count > CAPACITY ||
      loaded.state > DANGER ||
      loaded.nextEventId == 0 ||
      loaded.checksum != hash(loaded)
    ) {
      Serial.println(
        "[STORE] ERROR: NVS integrity check failed"
      );

      return false;
    }

    data_ = loaded;

    Serial.printf(
      "[STORE] RESTORED | state=%u | pending=%u | next=%lu\n",
      (unsigned)data_.state,
      (unsigned)data_.count,
      (unsigned long)data_.nextEventId
    );

    return true;
  }

  State state() {
    xSemaphoreTake(lock_, portMAX_DELAY);

    State result = (State)data_.state;

    xSemaphoreGive(lock_);
    return result;
  }

  uint8_t pendingCount() {
    xSemaphoreTake(lock_, portMAX_DELAY);

    uint8_t count = data_.count;

    xSemaphoreGive(lock_);
    return count;
  }

  // Save state and alert in one NVS commit.
  // No alert when state remains unchanged.
  bool transition(
    State target,
    const Alert &source,
    bool &enqueued
  ) {
    enqueued = false;

    if (
      xSemaphoreTake(
        lock_,
        pdMS_TO_TICKS(1000)
      ) != pdTRUE
    ) {
      return false;
    }

    if (data_.state == target) {
      xSemaphoreGive(lock_);
      return true;
    }

    // Initial SAFE is a silent baseline.
    bool notify = !(
      data_.state == STARTUP &&
      target == SAFE
    );

    if (notify && data_.count == CAPACITY) {
      xSemaphoreGive(lock_);

      Serial.println(
        "[STORE] FULL: state transition NOT committed"
      );

      return false;
    }

    Blob candidate = data_;
    candidate.state = target;

    if (notify) {
      Alert event = source;

      event.deviceId = candidate.deviceId;
      event.eventId = candidate.nextEventId++;

      memset(
        event.mode,
        0,
        sizeof(event.mode)
      );

      strncpy(
        event.mode,
        target == DANGER ? "DANGER" : "SAFE",
        sizeof(event.mode) - 1
      );

      candidate.events[candidate.count++] = event;
    }

    bool committed = save(candidate);

    if (committed) {
      data_ = candidate;
    }

    xSemaphoreGive(lock_);

    if (!committed) {
      Serial.println(
        "[STORE] NVS WRITE FAILED: state unchanged"
      );

      return false;
    }

    enqueued = notify;

    if (notify) {
      Serial.printf(
        "[STORE] SAVED %s | event=%lu | pending=%u\n",
        target == DANGER ? "DANGER" : "SAFE",
        (unsigned long)candidate.events[
          candidate.count - 1
        ].eventId,
        (unsigned)candidate.count
      );
    } else {
      Serial.println(
        "[STORE] Initial SAFE baseline persisted"
      );
    }

    return true;
  }

  // Get oldest pending alert without deleting it.
  bool peek(Alert &out) {
    if (
      xSemaphoreTake(
        lock_,
        pdMS_TO_TICKS(1000)
      ) != pdTRUE
    ) {
      return false;
    }

    bool found = data_.count > 0;

    if (found) {
      out = data_.events[0];
    }

    xSemaphoreGive(lock_);
    return found;
  }

  // Remove only after successful HTTP delivery.
  bool acknowledge(
    uint32_t deviceId,
    uint32_t eventId
  ) {
    if (
      xSemaphoreTake(
        lock_,
        pdMS_TO_TICKS(1000)
      ) != pdTRUE
    ) {
      return false;
    }

    if (
      data_.count == 0 ||
      data_.events[0].deviceId != deviceId ||
      data_.events[0].eventId != eventId
    ) {
      xSemaphoreGive(lock_);
      return false;
    }

    Blob candidate = data_;

    for (uint8_t i = 1; i < candidate.count; ++i) {
      candidate.events[i - 1] =
        candidate.events[i];
    }

    --candidate.count;

    memset(
      &candidate.events[candidate.count],
      0,
      sizeof(Alert)
    );

    bool committed = save(candidate);

    if (committed) {
      data_ = candidate;
    }

    xSemaphoreGive(lock_);

    if (!committed) {
      Serial.println(
        "[STORE] ACK SAVE FAILED; event retained"
      );

      return false;
    }

    Serial.printf(
      "[STORE] ACKED event=%lu | remaining=%u\n",
      (unsigned long)eventId,
      (unsigned)candidate.count
    );

    return true;
  }

 private:

  static uint32_t hash(const Blob &blob) {
    const uint8_t* bytes =
      reinterpret_cast<const uint8_t*>(&blob);

    uint32_t h = 2166136261UL;

    for (
      size_t i = 0;
      i < offsetof(Blob, checksum);
      ++i
    ) {
      h ^= bytes[i];
      h *= 16777619UL;
    }

    return h;
  }

  bool save(Blob &candidate) {
    candidate.checksum = hash(candidate);

    return prefs_.putBytes(
      "outbox",
      &candidate,
      sizeof(Blob)
    ) == sizeof(Blob);
  }

  Preferences prefs_;
  SemaphoreHandle_t lock_ = nullptr;
  Blob data_ = {};
};

}
