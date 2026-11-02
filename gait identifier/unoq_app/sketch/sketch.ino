// MPU6050 -> Python over Arduino_RouterBridge at a fixed rate.
// Wiring: MPU6050 on the dedicated SDA/SCL header pins (Wire), verified with the
// imu_test sketch. Use Wire1 for the Qwiic connector instead.
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>

#define IMU_WIRE Wire
static const uint8_t MPU_ADDR = 0x68;   // AD0 low; use 0x69 if AD0 is high
static const uint32_t SAMPLE_PERIOD_MS = 10;  // 100 Hz

// Ranges set below: accel +/-16 g (2048 LSB/g), gyro +/-1000 dps (32.8 LSB/dps).
// Python converts raw counts, so changing these means changing SCALE in main.py.

// ---- LED matrix status (set from Python via Bridge) ----
// code 0..99 = show that number (countdown), 100 = recording, 101 = done / safe to unplug,
// 102 = shutting down
ArduinoLEDMatrix matrix;
static uint8_t frame[104];  // 8 rows x 13 cols, brightness 0..7
static volatile int statusCode = -1;
static int drawnCode = -2;

static const uint8_t FONT[10][5] = {  // 3x5 digits, 3 bits per row (MSB = left)
  {7,5,5,5,7},{2,6,2,2,7},{7,1,7,4,7},{7,1,7,1,7},{5,5,7,1,1},
  {7,4,7,1,7},{7,4,7,5,7},{7,1,1,1,1},{7,5,7,5,7},{7,5,7,1,7}};

static void px(int r, int c) { if (r >= 0 && r < 8 && c >= 0 && c < 13) frame[r * 13 + c] = 7; }

static void glyph(int digit, int col) {
  for (int r = 0; r < 5; r++)
    for (int b = 0; b < 3; b++)
      if (FONT[digit][r] & (4 >> b)) px(1 + r, col + b);
}

static void drawStatus(int code) {
  memset(frame, 0, sizeof(frame));
  if (code >= 0 && code <= 99) {
    if (code >= 10) { glyph(code / 10, 3); glyph(code % 10, 7); }
    else glyph(code, 5);
  } else if (code == 100) {          // recording: outline border
    for (int c = 0; c < 13; c++) { px(0, c); px(7, c); }
    for (int r = 0; r < 8; r++) { px(r, 0); px(r, 12); }
  } else if (code == 101) {          // done: check mark
    px(4,3); px(5,4); px(6,5); px(5,6); px(4,7); px(3,8); px(2,9); px(1,10);
  } else if (code == 102) {          // shutting down: X
    for (int i = 0; i < 8; i++) { px(i, 3 + i); px(i, 10 - i); }
  }
  matrix.draw(frame);
}

static void set_status(int code) { statusCode = code; }

static void writeReg(uint8_t reg, uint8_t val) {
  IMU_WIRE.beginTransmission(MPU_ADDR);
  IMU_WIRE.write(reg);
  IMU_WIRE.write(val);
  IMU_WIRE.endTransmission();
}

static bool readBurst(int16_t *out) {
  IMU_WIRE.beginTransmission(MPU_ADDR);
  IMU_WIRE.write(0x3B);  // ACCEL_XOUT_H
  if (IMU_WIRE.endTransmission(false) != 0) return false;
  if (IMU_WIRE.requestFrom((int)MPU_ADDR, 14) != 14) return false;
  for (int i = 0; i < 7; i++) {
    int16_t hi = IMU_WIRE.read();
    int16_t lo = IMU_WIRE.read();
    out[i] = (hi << 8) | lo;
  }
  return true;  // ax ay az temp gx gy gz
}

static uint8_t readReg(uint8_t reg) {
  IMU_WIRE.beginTransmission(MPU_ADDR);
  IMU_WIRE.write(reg);
  IMU_WIRE.endTransmission(false);
  IMU_WIRE.requestFrom((int)MPU_ADDR, 1);
  return IMU_WIRE.available() ? IMU_WIRE.read() : 0xFF;
}

// Wake and configure the MPU6050, then read the settings back. A write that gets
// lost (sensor not powered yet, loose wire) otherwise leaves it asleep, and a
// sleeping chip reads back all zeros.
static bool initImu() {
  for (int attempt = 0; attempt < 10; attempt++) {
    writeReg(0x6B, 0x80);  // device reset
    delay(100);
    writeReg(0x6B, 0x01);  // wake, clock = gyro X PLL
    delay(50);
    writeReg(0x1A, 0x03);  // DLPF ~44 Hz
    writeReg(0x1B, 0x10);  // gyro +/-1000 dps
    writeReg(0x1C, 0x18);  // accel +/-16 g
    if (readReg(0x6B) == 0x01 && readReg(0x1C) == 0x18 && readReg(0x1B) == 0x10) return true;
    delay(200);
  }
  return false;
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  drawStatus(100);   // boot test: border for 1.5 s proves the matrix works independent of Python
  delay(1500);
  drawStatus(88);    // then "88" until Python takes over (proves the sketch is running)
  drawnCode = 88;
  Bridge.begin();
  Bridge.provide("set_status", set_status);
  IMU_WIRE.begin();
  IMU_WIRE.setClock(400000);
  if (!initImu()) { statusCode = 102; }  // X on the matrix = IMU did not initialise
}

void loop() {
  int sc = statusCode;
  if (sc != drawnCode) { drawStatus(sc); drawnCode = sc; }

  static uint32_t next = 0;
  uint32_t now = millis();
  if ((int32_t)(now - next) < 0) return;
  next = now + SAMPLE_PERIOD_MS;

  int16_t r[7];
  if (!readBurst(r)) return;  // dropped sample shows up as a gap in timestamps

  // All six channels exactly zero never happens on a working chip (gravity alone
  // makes accel non-zero): the sensor is asleep or reset. Re-initialise, send nothing.
  static int zeroCount = 0;
  if (r[0] == 0 && r[1] == 0 && r[2] == 0 && r[4] == 0 && r[5] == 0 && r[6] == 0) {
    if (++zeroCount >= 20) { initImu(); zeroCount = 0; }
    return;
  }
  zeroCount = 0;
  Bridge.notify("imu_sample", (uint32_t)now, r[0], r[1], r[2], r[4], r[5], r[6]);
}
