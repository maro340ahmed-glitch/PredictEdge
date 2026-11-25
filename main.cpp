#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <esp_system.h>
#include <time.h>
#include <math.h>
#include <string.h>

#include "PredictEdgeModel.h"
#include "PersistentOutbox.h"

// ============================================
// 1. PINS
// ============================================

constexpr uint8_t MAX6675_SO = 25;
constexpr uint8_t MAX6675_CS = 32;
constexpr uint8_t MAX6675_SCK = 33;

constexpr uint8_t CURRENT_PIN = 34;

constexpr uint8_t SDA_PIN = 21;
constexpr uint8_t SCL_PIN = 22;
constexpr uint8_t OLED_ADDR = 0x3C;
constexpr uint8_t MPU_ADDR = 0x68;

// Alert pins
constexpr uint8_t RED_LED_PIN = 14;
constexpr uint8_t BLUE_LED_PIN = 12;
constexpr uint8_t BUZZER_PIN = 27;

constexpr bool PASSIVE_BUZZER = true;

// ============================================
// 2. MELODIC ALERT CONFIG
// ============================================

constexpr unsigned int TONE_CHIME_LOW = 880;
constexpr unsigned int TONE_CHIME_HIGH = 1175;

constexpr int ALERT_CHIME_COUNT = 3;
constexpr uint32_t ALERT_CHIME_GAP_MS = 250;

// ============================================
// 3. SENSORS & GRADUAL RISK CONFIG
// ============================================

constexpr float ADC_REFERENCE_V = 3.3f;
constexpr float ADC_MAX_VALUE = 4095.0f;

float acs712ZeroV = 2.42f;
constexpr float ACS712_SENSITIVITY = 0.06667f;
constexpr float CURRENT_DEADBAND_A = 0.15f;

constexpr int VIB_SAMPLES = 50;
constexpr uint32_t VIB_SAMPLE_DELAY_MS = 10;

// نطاقات التدرج التدريجي (0% -> 100%)
constexpr float TEMP_SAFE_C = 25.0f;
constexpr float TEMP_DANGER_C = 90.0f;
constexpr float CURRENT_DANGER_A = 3.0f;
constexpr float VIB_SAFE_G = 0.02f;
constexpr float VIB_DANGER_G = 0.50f;

// عتبات التحول مع Hysteresis لمنع التذبذب
constexpr float RISK_ALERT_THRESHOLD = 0.80f; // يدخل DANGER عند 80%
constexpr float RISK_RESET_THRESHOLD = 0.60f; // يرجع SAFE تحت 60%
constexpr float RISK_SMOOTH_ALPHA = 0.35f;    // سرعة التدرج كل ثانية

// ============================================
// 4. NETWORK (WOKWI)
// ============================================

constexpr bool ENABLE_WEBHOOK = true;
constexpr bool INSECURE_TLS = true;

const char *WIFI_SSID = "Mi";
const char *WIFI_PASS = "123456789";

const char *WEBHOOK_URL =
    "https://n8n-production-6877272c.up.railway.app/webhook/PREDICT-EDGE";

const char *MACHINE_ID = "Motor_01";

// ===========================================  =
// 5. TIMING
// ============================================

constexpr uint32_t SENSOR_INTERVAL_MS = 1000;
constexpr uint32_t SUMMARY_INTERVAL_MS = 8000;

constexpr uint32_t WIFI_RETRY_MS = 15000;
constexpr uint32_t WEBHOOK_RETRY_MS = 10000;

constexpr uint32_t HTTP_CONNECT_TIMEOUT_MS = 5000;
constexpr uint32_t HTTP_TIMEOUT_MS = 8000;

constexpr int HISTORY_SIZE = 3;
constexpr int MIN_VALID_SAMPLES = 6;

constexpr uint32_t EXPECTED_TREND_MS =
    (HISTORY_SIZE - 1) * SUMMARY_INTERVAL_MS;

constexpr uint32_t TREND_TOLERANCE_MS = 5000;

static_assert(
    PredictEdgeML::FEATURES == 6,
    "Model expects six features");

static_assert(
    PredictEdgeML::SUMMARY_PERIOD_SECONDS * 1000UL == SUMMARY_INTERVAL_MS,
    "Summary period mismatch");

// ============================================
// 6. GLOBALS
// ============================================

using EdgeStore::Alert;
using EdgeStore::PersistentOutbox;
using EdgeStore::SensorData;
using EdgeStore::State;

struct SensorSummary
{
    SensorData values;
    uint32_t sequence;
    uint32_t timeMs;
};

Adafruit_SSD1306 display(128, 64, &Wire, -1);
SPIClass thermoSPI(VSPI);
PersistentOutbox outbox;

SemaphoreHandle_t i2cMutex = nullptr;
SemaphoreHandle_t historyMutex = nullptr;

TaskHandle_t inferenceHandle = nullptr;
TaskHandle_t alertHandle = nullptr;

SensorSummary history[HISTORY_SIZE] = {};

int historyCount = 0;
int historyNext = 0;

uint32_t summarySequence = 0;

bool latestValid = false;

SensorData latestData = {
    NAN, NAN, NAN, false};

bool oledReady = false;
bool mpuRawReady = false;
bool buzzerToneActive = false;

volatile State liveState = EdgeStore::STARTUP;
volatile float latestMlRisk = NAN;
volatile bool mlReady = false;

float smoothedRisk = 0.0f;
bool riskInitialized = false;

uint32_t bootId = 0;

// ============================================
// 7. HELPERS
// ============================================

const char *stateName(State state)
{
    switch (state)
    {
    case EdgeStore::SAFE:
        return "SAFE";
    case EdgeStore::DANGER:
        return "DANGER";
    default:
        return "STARTUP";
    }
}

float clamp01(float x)
{
    if (!isfinite(x) || x < 0.0f)
        return 0.0f;
    if (x > 1.0f)
        return 1.0f;
    return x;
}

float computeSensorRisk(const SensorData &s)
{
    float tempRisk = clamp01((s.temperature - TEMP_SAFE_C) / (TEMP_DANGER_C - TEMP_SAFE_C));
    float currRisk = clamp01(s.current / CURRENT_DANGER_A);
    float vibRisk = clamp01((s.vibration - VIB_SAFE_G) / (VIB_DANGER_G - VIB_SAFE_G));
    return fmaxf(tempRisk, fmaxf(currRisk, vibRisk));
}

float computeFusedTargetRisk(const SensorData &s)
{
    float sensorRisk = computeSensorRisk(s);
    if (mlReady && isfinite(latestMlRisk))
    {
        float mlRisk = clamp01(latestMlRisk);
        float blended = 0.55f * sensorRisk + 0.45f * mlRisk;
        return clamp01(fmaxf(sensorRisk, blended));
    }
    return sensorRisk;
}

void allIndicatorsOff()
{
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(BLUE_LED_PIN, LOW);
}

void buzzerOff()
{
    if (PASSIVE_BUZZER)
    {
        if (buzzerToneActive)
        {
            noTone(BUZZER_PIN);
            buzzerToneActive = false;
        }
        digitalWrite(BUZZER_PIN, LOW);
    }
    else
    {
        digitalWrite(BUZZER_PIN, LOW);
    }
}

void playSoftChime()
{
    if (PASSIVE_BUZZER)
    {
        tone(BUZZER_PIN, TONE_CHIME_LOW);
        buzzerToneActive = true;
        vTaskDelay(pdMS_TO_TICKS(50));

        tone(BUZZER_PIN, TONE_CHIME_HIGH);
        buzzerToneActive = true;
        vTaskDelay(pdMS_TO_TICKS(60));

        noTone(BUZZER_PIN);
        buzzerToneActive = false;
        digitalWrite(BUZZER_PIN, LOW);
    }
    else
    {
        digitalWrite(BUZZER_PIN, HIGH);
        vTaskDelay(pdMS_TO_TICKS(60));
        digitalWrite(BUZZER_PIN, LOW);
    }
}

void fatalError(const char *message)
{
    Serial.print("[FATAL] ");
    Serial.println(message);

    buzzerOff();
    allIndicatorsOff();

    for (;;)
    {
        delay(1000);
    }
}

bool initMPU6050Raw()
{
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B);
    Wire.write(0x00);
    if (Wire.endTransmission() != 0)
    {
        return false;
    }

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1C);
    Wire.write(0x10);
    if (Wire.endTransmission() != 0)
    {
        return false;
    }

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x1A);
    Wire.write(0x03);
    Wire.endTransmission();

    return true;
}

bool readMPU6050AccelG(float &gx, float &gy, float &gz)
{
    if (!mpuRawReady)
    {
        return false;
    }

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    if (Wire.endTransmission(false) != 0)
    {
        return false;
    }

    if (Wire.requestFrom(MPU_ADDR, (uint8_t)6, (uint8_t)true) != 6)
    {
        return false;
    }

    int16_t rx = (int16_t)((Wire.read() << 8) | Wire.read());
    int16_t ry = (int16_t)((Wire.read() << 8) | Wire.read());
    int16_t rz = (int16_t)((Wire.read() << 8) | Wire.read());

    constexpr float LSB_PER_G_8G = 4096.0f;
    gx = rx / LSB_PER_G_8G;
    gy = ry / LSB_PER_G_8G;
    gz = rz / LSB_PER_G_8G;
    return true;
}

// ============================================
// 8. ALERT TASK
// ============================================

void Task_Alert(void *pv)
{
    (void)pv;

    State lastState = EdgeStore::STARTUP;

    for (;;)
    {
        State current = liveState;

        if (current == EdgeStore::DANGER)
        {
            digitalWrite(BLUE_LED_PIN, LOW);
            digitalWrite(RED_LED_PIN, HIGH);

            if (lastState != EdgeStore::DANGER)
            {
                for (int i = 0; i < ALERT_CHIME_COUNT; ++i)
                {
                    if (liveState != EdgeStore::DANGER)
                    {
                        break;
                    }
                    playSoftChime();
                    if (i < ALERT_CHIME_COUNT - 1)
                    {
                        vTaskDelay(pdMS_TO_TICKS(ALERT_CHIME_GAP_MS));
                    }
                }
                buzzerOff();
            }
        }
        else if (current == EdgeStore::SAFE)
        {
            digitalWrite(RED_LED_PIN, LOW);
            digitalWrite(BLUE_LED_PIN, HIGH);

            // عند الرجوع من DANGER إلى SAFE: شغل البازر مع الليد الزرقاء
            if (lastState == EdgeStore::DANGER)
            {
                for (int i = 0; i < ALERT_CHIME_COUNT; ++i)
                {
                    if (liveState != EdgeStore::SAFE)
                    {
                        break;
                    }
                    playSoftChime();
                    if (i < ALERT_CHIME_COUNT - 1)
                    {
                        vTaskDelay(pdMS_TO_TICKS(ALERT_CHIME_GAP_MS));
                    }
                }
                buzzerOff();
            }
            else
            {
                buzzerOff();
            }
        }
        else
        {
            buzzerOff();
            allIndicatorsOff();
        }

        lastState = current;
        vTaskDelay(pdMS_TO_TICKS(150));
    }
}

// ============================================
// 9. TEMPERATURE - MAX6675
// ============================================

float readTemperature()
{
    thermoSPI.beginTransaction(
        SPISettings(500000, MSBFIRST, SPI_MODE0));

    digitalWrite(MAX6675_CS, LOW);
    delayMicroseconds(5);

    uint16_t raw = thermoSPI.transfer16(0x0000);

    digitalWrite(MAX6675_CS, HIGH);
    thermoSPI.endTransaction();

    if ((raw & 0x0004) || (raw & 0x8000))
    {
        Serial.printf(
            "[MAX6675] Invalid raw value: 0x%04X\n",
            raw);
        return NAN;
    }

    return ((raw >> 3) & 0x0FFF) * 0.25f;
}

// ============================================
// 10. CURRENT - ACS712
// ============================================

float readCurrent()
{
    int adc = analogRead(CURRENT_PIN);

    float voltage = (adc / ADC_MAX_VALUE) * ADC_REFERENCE_V;

    float signedCurrent =
        (voltage - acs712ZeroV) / ACS712_SENSITIVITY;

    float current = fabsf(signedCurrent);

    if (current < CURRENT_DEADBAND_A)
    {
        current = 0.0f;
    }

    return current;
}

// ============================================
// 11. VIBRATION - MPU6050 RMS
// ============================================

float readVibration()
{
    if (!mpuRawReady)
    {
        return 0.0f;
    }

    float ax[VIB_SAMPLES];
    float ay[VIB_SAMPLES];
    float az[VIB_SAMPLES];

    float mx = 0.0f;
    float my = 0.0f;
    float mz = 0.0f;

    int validSamples = 0;

    for (int i = 0; i < VIB_SAMPLES; ++i)
    {
        float gx = 0.0f, gy = 0.0f, gz = 0.0f;

        if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(100)) == pdTRUE)
        {
            bool ok = readMPU6050AccelG(gx, gy, gz);
            xSemaphoreGive(i2cMutex);

            if (ok && isfinite(gx) && isfinite(gy) && isfinite(gz))
            {
                ax[validSamples] = gx;
                ay[validSamples] = gy;
                az[validSamples] = gz;

                mx += gx;
                my += gy;
                mz += gz;
                validSamples++;
            }
        }

        vTaskDelay(pdMS_TO_TICKS(VIB_SAMPLE_DELAY_MS));
    }

    if (validSamples == 0)
    {
        return 0.0f;
    }

    mx /= validSamples;
    my /= validSamples;
    mz /= validSamples;

    float sum = 0.0f;
    for (int i = 0; i < validSamples; ++i)
    {
        float dx = ax[i] - mx;
        float dy = ay[i] - my;
        float dz = az[i] - mz;

        sum += dx * dx + dy * dy + dz * dz;
    }

    return sqrtf(sum / validSamples);
}

// ============================================
// 12. READ ALL SENSORS
// ============================================

SensorData readSensors()
{
    SensorData s;

    s.temperature = readTemperature();
    s.current = readCurrent();
    s.vibration = readVibration();

    s.valid =
        isfinite(s.temperature) &&
        isfinite(s.current) &&
        isfinite(s.vibration) &&
        s.temperature >= 0 &&
        s.current >= 0 &&
        s.vibration >= 0;

    return s;
}

// ============================================
// 13. DATE AND TIME
// ============================================

void captureDateTime(Alert &a)
{
    a.date[0] = '\0';
    a.time[0] = '\0';

    struct tm info;
    if (getLocalTime(&info, 10))
    {
        strftime(a.date, sizeof(a.date), "%Y-%m-%d", &info);
        strftime(a.time, sizeof(a.time), "%H:%M:%S", &info);
    }
}

// ============================================
// 14. MACHINE STATE TRANSITIONS (NVS PERSISTENT OUTBOX)
// ============================================

void changeMachineState(
    State target,
    const SensorData &s,
    float risk)
{
    if (target == liveState)
    {
        return;
    }

    Alert event = {};
    event.sensors = s;
    event.risk = risk;
    event.bootId = bootId;
    captureDateTime(event);

    bool enqueued = false;
    if (outbox.transition(target, event, enqueued))
    {
        liveState = target;
    }
}

// ============================================
// 15. OLED DASHBOARD (LIVE 1s GRADUAL RISK + BAR)
// ============================================

void updateDisplay(float risk, const char *state)
{
    if (!oledReady)
    {
        return;
    }

    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(250)) != pdTRUE)
    {
        return;
    }

    display.clearDisplay();
    display.setTextWrap(false);
    display.setTextColor(SSD1306_WHITE);

    // السطر الأول: Risk: XX%
    display.setTextSize(2);
    display.setCursor(0, 4);
    display.print("Risk: ");
    if (isfinite(risk))
    {
        int pct = (int)lroundf(clamp01(risk) * 100.0f);
        display.print(pct);
        display.print("%");
    }
    else
    {
        display.print("--%");
    }

    // السطر الثاني: Mode: SAFE / DANGER
    display.setTextSize(1);
    display.setCursor(0, 32);
    display.print("Mode:");
    display.setTextSize(2);
    display.setCursor(34, 28);
    display.print(state);

    // شريط تدرج بصري للخطر (0% -> 100%)
    display.drawRect(4, 52, 120, 9, SSD1306_WHITE);
    if (isfinite(risk))
    {
        int barW = (int)lroundf(clamp01(risk) * 116.0f);
        if (barW > 0)
        {
            display.fillRect(6, 54, barW, 5, SSD1306_WHITE);
        }
    }

    display.display();
    xSemaphoreGive(i2cMutex);
}

// ============================================
// 16. SAMPLING TASK (LIVE 1s RISK + OLED + 8s SUMMARY)
// ============================================

void Task_Sampling(void *pv)
{
    (void)pv;

    TickType_t wake = xTaskGetTickCount();
    TickType_t nextSummary = wake + pdMS_TO_TICKS(SUMMARY_INTERVAL_MS);

    uint32_t good = 0;
    double sumT = 0;
    double sumI = 0;
    double sumV = 0;

    for (;;)
    {
        SensorData s = readSensors();

        if (s.valid)
        {
            good++;
            sumT += s.temperature;
            sumI += s.current;
            sumV += s.vibration;

            // حساب التدرج اللحظي كل ثانية
            float targetRisk = computeFusedTargetRisk(s);
            if (!riskInitialized)
            {
                smoothedRisk = targetRisk;
                riskInitialized = true;
            }
            else
            {
                smoothedRisk += RISK_SMOOTH_ALPHA * (targetRisk - smoothedRisk);
            }
            smoothedRisk = clamp01(smoothedRisk);

            // تحديث الحالة وحفظها في NVS عند عبور العتبات
            if (smoothedRisk >= RISK_ALERT_THRESHOLD)
            {
                changeMachineState(EdgeStore::DANGER, s, smoothedRisk);
            }
            else if (smoothedRisk < RISK_RESET_THRESHOLD)
            {
                changeMachineState(EdgeStore::SAFE, s, smoothedRisk);
            }

            Serial.printf(
                "[LIVE] T=%.2f C | I=%.2f A | V=%.3f | Risk=%.1f%% | Mode=%s\n",
                s.temperature,
                s.current,
                s.vibration,
                smoothedRisk * 100.0f,
                stateName(liveState));

            updateDisplay(smoothedRisk, stateName(liveState));
        }
        else
        {
            Serial.println("[LIVE] SENSOR FAULT (invalid reading)");
            updateDisplay(NAN, "FAULT");
        }

        TickType_t now = xTaskGetTickCount();

        if ((int32_t)(now - nextSummary) >= 0)
        {
            SensorData avg = {NAN, NAN, NAN, false};

            if (good > 0)
            {
                avg.temperature = (float)(sumT / good);
                avg.current = (float)(sumI / good);
                avg.vibration = (float)(sumV / good);
            }

            avg.valid = good >= MIN_VALID_SAMPLES;
            uint32_t seq = summarySequence;

            if (xSemaphoreTake(historyMutex, pdMS_TO_TICKS(100)) == pdTRUE)
            {
                seq = ++summarySequence;
                latestData = avg;
                latestValid = avg.valid;

                if (avg.valid)
                {
                    history[historyNext] = {avg, seq, millis()};
                    historyNext = (historyNext + 1) % HISTORY_SIZE;

                    if (historyCount < HISTORY_SIZE)
                    {
                        historyCount++;
                    }
                }
                else
                {
                    historyCount = 0;
                    historyNext = 0;
                }

                xSemaphoreGive(historyMutex);
                xTaskNotifyGive(inferenceHandle);
            }
            else
            {
                Serial.println("[SUMMARY] History mutex timeout");
            }

            Serial.printf(
                "\n--- [SUMMARY #%lu] (Samples=%lu) | T_avg=%.2f C | I_avg=%.2f A | V_avg=%.3f | Risk=%.1f%% ---\n\n",
                (unsigned long)seq,
                (unsigned long)good,
                avg.temperature,
                avg.current,
                avg.vibration,
                smoothedRisk * 100.0f);

            good = 0;
            sumT = 0;
            sumI = 0;
            sumV = 0;

            nextSummary += pdMS_TO_TICKS(SUMMARY_INTERVAL_MS);

            if ((int32_t)(now - nextSummary) >= 0)
            {
                nextSummary = now + pdMS_TO_TICKS(SUMMARY_INTERVAL_MS);
            }
        }

        vTaskDelayUntil(&wake, pdMS_TO_TICKS(SENSOR_INTERVAL_MS));
    }
}

// ============================================
// 17. ML INFERENCE TASK (3-SUMMARY WINDOW)
// ============================================

void Task_Inference(void *pv)
{
    (void)pv;

    uint32_t processedSequence = 0;

    for (;;)
    {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        SensorSummary oldest = {};
        SensorSummary newest = {};
        int count = 0;
        bool valid = false;
        uint32_t seq = 0;

        if (xSemaphoreTake(historyMutex, pdMS_TO_TICKS(100)) != pdTRUE)
        {
            Serial.println("[ML] History mutex timeout");
            continue;
        }

        count = historyCount;
        valid = latestValid;
        seq = summarySequence;

        if (count == HISTORY_SIZE)
        {
            oldest = history[historyNext];
            newest = history[(historyNext + HISTORY_SIZE - 1) % HISTORY_SIZE];
        }

        xSemaphoreGive(historyMutex);

        if (seq == processedSequence)
        {
            continue;
        }

        processedSequence = seq;

        if (!valid)
        {
            mlReady = false;
            continue;
        }

        if (count < HISTORY_SIZE)
        {
            Serial.printf("[ML] Building window: %d/%d summaries...\n", count, HISTORY_SIZE);
            continue;
        }

        uint32_t dtMs = newest.timeMs - oldest.timeMs;

        if (
            dtMs == 0 ||
            dtMs < EXPECTED_TREND_MS - TREND_TOLERANCE_MS ||
            dtMs > EXPECTED_TREND_MS + TREND_TOLERANCE_MS)
        {
            Serial.printf("[ML] Trend time out of range: %lu ms\n", (unsigned long)dtMs);
            continue;
        }

        float dtSec = dtMs / 1000.0f;

        float features[PredictEdgeML::FEATURES] = {
            newest.values.temperature,
            newest.values.current,
            newest.values.vibration,
            (newest.values.temperature - oldest.values.temperature) / dtSec,
            (newest.values.current - oldest.values.current) / dtSec,
            (newest.values.vibration - oldest.values.vibration) / dtSec};

        float mlRisk = PredictEdgeML::predictRisk(features);

        if (!isfinite(mlRisk))
        {
            Serial.println("[ML] Invalid prediction");
            continue;
        }

        latestMlRisk = clamp01(mlRisk);
        mlReady = true;

        Serial.printf(
            "[ML] RawML=%.1f%% | LiveRisk=%.1f%% | Mode=%s | Window=%.1fs | NVS Pending=%u\n",
            latestMlRisk * 100.0f,
            smoothedRisk * 100.0f,
            stateName(liveState),
            dtSec,
            (unsigned)outbox.pendingCount());
    }
}

// ============================================
// 18. JSON HELPERS
// ============================================

String jsonNumber(float value, unsigned int digits)
{
    return isfinite(value) ? String(value, digits) : String("null");
}

String eventToJSON(const Alert &a)
{
    String json;
    json.reserve(320);

    json += "{\"machine\":\"";
    json += MACHINE_ID;
    json += "\",\"mode\":\"";
    json += a.mode;
    json += "\",\"device_id\":";
    json += String(a.deviceId);
    json += ",\"boot_id\":";
    json += String(a.bootId);
    json += ",\"event_id\":";
    json += String(a.eventId);
    json += ",\"risk_score\":";
    json += jsonNumber(a.risk, 4);
    json += ",\"temperature\":";
    json += jsonNumber(a.sensors.temperature, 2);
    json += ",\"current\":";
    json += jsonNumber(a.sensors.current, 2);
    json += ",\"vibration\":";
    json += jsonNumber(a.sensors.vibration, 4);
    json += ",\"date\":";

    if (a.date[0])
    {
        json += '"';
        json += a.date;
        json += '"';
    }
    else
    {
        json += "null";
    }

    json += ",\"time\":";
    if (a.time[0])
    {
        json += '"';
        json += a.time;
        json += '"';
    }
    else
    {
        json += "null";
    }

    json += '}';
    return json;
}

// ============================================
// 19. WIFI STATUS
// ============================================

const char *wifiStatusName(wl_status_t status)
{
    switch (status)
    {
    case WL_CONNECTED:
        return "CONNECTED";
    case WL_NO_SSID_AVAIL:
        return "SSID_NOT_FOUND";
    case WL_CONNECT_FAILED:
        return "CONNECT_FAILED";
    case WL_CONNECTION_LOST:
        return "CONNECTION_LOST";
    case WL_DISCONNECTED:
        return "DISCONNECTED";
    case WL_IDLE_STATUS:
        return "IDLE";
    default:
        return "OTHER";
    }
}

// ============================================
// 20. HTTP POST TO N8N
// ============================================

bool postWebhook(const Alert &event)
{
    WiFiClientSecure client;

    if (INSECURE_TLS)
    {
        client.setInsecure();
    }
    else
    {
        Serial.println("[CLOUD] CA certificate not configured");
        return false;
    }

    HTTPClient http;
    http.setConnectTimeout(HTTP_CONNECT_TIMEOUT_MS);
    http.setTimeout(HTTP_TIMEOUT_MS);

    if (!http.begin(client, WEBHOOK_URL))
    {
        Serial.println("[CLOUD] HTTP begin failed");
        return false;
    }

    http.addHeader("Content-Type", "application/json");

    String payload = eventToJSON(event);
    Serial.printf("[CLOUD] POST (%s #%lu):\n%s\n", event.mode, (unsigned long)event.eventId, payload.c_str());

    int code = http.POST(payload);

    if (code < 0)
    {
        Serial.printf("[CLOUD] HTTP ERROR %d: %s\n", code, http.errorToString(code).c_str());
    }
    else
    {
        Serial.printf("[CLOUD] HTTP STATUS: %d\n", code);
        String response = http.getString();
        if (response.length() > 0)
        {
            Serial.printf("[CLOUD] Server response: %s\n", response.c_str());
        }
    }

    http.end();
    return code >= 200 && code < 300;
}

// ============================================
// 21. NETWORK TASK (NVS OUTBOX PEEK -> POST -> ACK)
// ============================================

void Task_Network(void *pv)
{
    (void)pv;

    if (!ENABLE_WEBHOOK)
    {
        Serial.println("[NET] Webhook disabled");
        vTaskDelete(nullptr);
        return;
    }

    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    WiFi.setSleep(false);
    WiFi.begin(WIFI_SSID, WIFI_PASS, 6);

    Serial.println("[NET] Connecting to configured WiFi...");

    configTzTime("EET-2EEST,M4.5.5/0,M10.5.4/24", "pool.ntp.org", "time.google.com");

    bool connectedLogged = false;
    uint32_t lastWiFiRetry = millis();
    uint32_t lastAttempt = 0;

    for (;;)
    {
        wl_status_t status = WiFi.status();
        uint32_t now = millis();

        if (status == WL_CONNECTED)
        {
            if (!connectedLogged)
            {
                connectedLogged = true;
                Serial.print("[NET] CONNECTED | IP: ");
                Serial.println(WiFi.localIP());
            }
        }
        else
        {
            if (connectedLogged)
            {
                Serial.println("[NET] Connection lost");
            }
            connectedLogged = false;

            if ((uint32_t)(now - lastWiFiRetry) >= WIFI_RETRY_MS)
            {
                lastWiFiRetry = now;
                Serial.printf("[NET] WiFi status=%d (%s); reconnecting...\n", (int)status, wifiStatusName(status));
                WiFi.reconnect();
            }
        }

        if (WiFi.status() == WL_CONNECTED && (uint32_t)(now - lastAttempt) >= WEBHOOK_RETRY_MS)
        {
            lastAttempt = now;

            Alert pending = {};
            if (outbox.peek(pending))
            {
                Serial.printf("[NET] Attempting %s event #%lu...\n", pending.mode, (unsigned long)pending.eventId);
                if (postWebhook(pending))
                {
                    outbox.acknowledge(pending.deviceId, pending.eventId);
                }
                else
                {
                    Serial.printf("[CLOUD] %s event #%lu failed; retained in NVS\n", pending.mode, (unsigned long)pending.eventId);
                }
            }
        }

        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

// ============================================
// 22. SETUP
// ============================================

void setup()
{
    Serial.begin(115200);
    delay(250);

    Serial.println("\n[BOOT] PredictEdge - Live Gradual Risk + ML + NVS Outbox...");

    pinMode(RED_LED_PIN, OUTPUT);
    pinMode(BLUE_LED_PIN, OUTPUT);
    allIndicatorsOff();

    pinMode(BUZZER_PIN, OUTPUT);
    buzzerOff();

    pinMode(MAX6675_CS, OUTPUT);
    digitalWrite(MAX6675_CS, HIGH);

    thermoSPI.begin(MAX6675_SCK, MAX6675_SO, -1, MAX6675_CS);

    analogReadResolution(12);
    analogSetPinAttenuation(CURRENT_PIN, ADC_11db);
    pinMode(CURRENT_PIN, INPUT);

    // معايرة حساس التيار مع حماية ضد السلك المفصول (0.00V)
    double adcSum = 0;
    for (int i = 0; i < 100; ++i)
    {
        adcSum += analogRead(CURRENT_PIN);
        delay(2);
    }
    float calibratedV = ((adcSum / 100.0) / ADC_MAX_VALUE) * ADC_REFERENCE_V;
    if (calibratedV >= 1.0f && calibratedV <= 2.9f)
    {
        acs712ZeroV = calibratedV;
        Serial.printf("[ACS712] Calibrated Zero V = %.4f V\n", acs712ZeroV);
    }
    else
    {
        acs712ZeroV = 2.42f;
        Serial.printf("[ACS712] Invalid Zero V (%.4f V) - using fallback %.2f V\n",
                      calibratedV, acs712ZeroV);
    }

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(100000);

    i2cMutex = xSemaphoreCreateMutex();
    historyMutex = xSemaphoreCreateMutex();

    if (!i2cMutex || !historyMutex)
    {
        fatalError("FreeRTOS mutex allocation failed");
    }

    if (!outbox.begin())
    {
        fatalError("PersistentOutbox NVS init failed");
    }
    liveState = outbox.state();

    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(250)) == pdTRUE)
    {
        oledReady = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
        if (oledReady)
        {
            display.clearDisplay();
            display.display();
        }
        xSemaphoreGive(i2cMutex);
    }

    if (xSemaphoreTake(i2cMutex, pdMS_TO_TICKS(250)) == pdTRUE)
    {
        mpuRawReady = initMPU6050Raw();
        xSemaphoreGive(i2cMutex);
    }
    if (!mpuRawReady)
    {
        Serial.println("[WARN] MPU6050 raw init failed; using fallback vibration=0.0");
    }
    else
    {
        Serial.println("[MPU6050] Raw init OK");
    }

    bootId = esp_random();

    updateDisplay(NAN, stateName(liveState));

    xTaskCreatePinnedToCore(Task_Alert, "Alert", 3072, nullptr, 2, &alertHandle, 1);
    xTaskCreatePinnedToCore(Task_Inference, "Inference", 8192, nullptr, 2, &inferenceHandle, 0);
    xTaskCreatePinnedToCore(Task_Network, "Network", 12288, nullptr, 1, nullptr, 1);
    xTaskCreatePinnedToCore(Task_Sampling, "Sampling", 8192, nullptr, 3, nullptr, 1);

    Serial.println("[BOOT] PredictEdge Node Ready.");
}

// ============================================
// 23. LOOP
// ============================================

void loop()
{
    vTaskDelay(pdMS_TO_TICKS(1000));
}
