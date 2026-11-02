// IMU sanity test: shows an arrow on the Uno Q LED matrix (8 rows x 13 cols)
// pointing in the direction the board is accelerating.
//
// Hold the board flat, matrix facing up. Push it: right/left/forward/back.
//   dot   = still
//   arrow = moving that way (gravity is filtered out, so tilting mostly won't trigger it)
//   X     = IMU not found (check wiring / IMU_WIRE / MPU_ADDR)
// Serial prints raw g values at ~10 Hz, so you can also check the readings there.
#include <Arduino_LED_Matrix.h>
#include <Wire.h>

#define IMU_WIRE Wire1                // Qwiic port; use Wire for the header SDA/SCL pins
static const uint8_t MPU_ADDR = 0x68; // 0x69 if AD0 is high

// Flip these if an arrow points the wrong way for your mounting.
static const int SIGN_RIGHT = +1;  // +1: positive ax -> right arrow
static const int SIGN_UP    = +1;  // +1: positive ay -> up arrow

static const float THRESHOLD_G = 0.20f;  // acceleration needed to show an arrow
static const uint32_t HOLD_MS  = 300;    // keep the arrow up briefly so it's visible

ArduinoLEDMatrix matrix;
static uint8_t frame[104];  // 8 x 13, row-major, brightness 0..7

static void px(int r, int c) {
  if (r >= 0 && r < 8 && c >= 0 && c < 13) frame[r * 13 + c] = 7;
}

static void drawArrow(char dir) {
  memset(frame, 0, sizeof(frame));
  // Build an "up" arrow, then mirror/transpose as needed.
  if (dir == 'U' || dir == 'D') {
    int pts[][2] = {{0,6},{1,5},{1,6},{1,7},{2,4},{2,5},{2,6},{2,7},{2,8},
                    {3,6},{4,6},{5,6},{6,6},{7,6}};
    for (auto &p : pts) px(dir == 'U' ? p[0] : 7 - p[0], p[1]);
  } else if (dir == 'R' || dir == 'L') {
    for (int c = 0; c <= 12; c++) {
      int cc = (dir == 'R') ? c : 12 - c;
      if (c <= 9) { px(3, cc); px(4, cc); }
      if (c == 10) for (int r = 1; r <= 6; r++) px(r, cc);
      if (c == 11) for (int r = 2; r <= 5; r++) px(r, cc);
      if (c == 12) { px(3, cc); px(4, cc); }
    }
  } else if (dir == 'X') {
    for (int i = 0; i < 8; i++) { px(i, 3 + i); px(i, 9 - i); }
  } else {  // still: small centre block
    px(3, 6); px(3, 7); px(4, 6); px(4, 7);
  }
  matrix.draw(frame);
}

static bool readAccel(float &ax, float &ay, float &az) {
  IMU_WIRE.beginTransmission(MPU_ADDR);
  IMU_WIRE.write(0x3B);
  if (IMU_WIRE.endTransmission(false) != 0) return false;
  if (IMU_WIRE.requestFrom((int)MPU_ADDR, 6) != 6) return false;
  int16_t v[3];
  for (int i = 0; i < 3; i++) {
    int16_t hi = IMU_WIRE.read();
    int16_t lo = IMU_WIRE.read();
    v[i] = (hi << 8) | lo;
  }
  const float LSB_PER_G = 8192.0f;  // +/-4 g range
  ax = v[0] / LSB_PER_G; ay = v[1] / LSB_PER_G; az = v[2] / LSB_PER_G;
  return true;
}

void setup() {
  Serial.begin(115200);
  matrix.begin();
  matrix.setGrayscaleBits(3);
  IMU_WIRE.begin();

  IMU_WIRE.beginTransmission(MPU_ADDR);
  IMU_WIRE.write(0x6B); IMU_WIRE.write(0x00);  // wake
  bool ok = (IMU_WIRE.endTransmission() == 0);
  if (ok) {
    IMU_WIRE.beginTransmission(MPU_ADDR); IMU_WIRE.write(0x1C); IMU_WIRE.write(0x08);  // +/-4 g
    IMU_WIRE.endTransmission();
    IMU_WIRE.beginTransmission(MPU_ADDR); IMU_WIRE.write(0x1A); IMU_WIRE.write(0x03);  // ~44 Hz filter
    IMU_WIRE.endTransmission();
  }
  if (!ok) {
    Serial.println("MPU6050 not found at that address/bus");
    while (true) { drawArrow('X'); delay(500); }
  }
  Serial.println("MPU6050 OK");
}

void loop() {
  static float gx = 0, gy = 0, gz = 1;  // gravity estimate (low-pass)
  static bool init = false;
  static char shown = 'S';
  static uint32_t shownAt = 0, lastPrint = 0;

  float ax, ay, az;
  if (!readAccel(ax, ay, az)) { delay(10); return; }
  if (!init) { gx = ax; gy = ay; gz = az; init = true; }
  const float a = 0.02f;  // ~0.3 s time constant at 100 Hz
  gx += a * (ax - gx); gy += a * (ay - gy); gz += a * (az - gz);

  float lx = (ax - gx) * SIGN_RIGHT;  // motion minus gravity
  float ly = (ay - gy) * SIGN_UP;

  char want = 'S';
  float mag = max(fabsf(lx), fabsf(ly));
  if (mag > THRESHOLD_G) {
    if (fabsf(lx) >= fabsf(ly)) want = lx > 0 ? 'R' : 'L';
    else                        want = ly > 0 ? 'U' : 'D';
  }

  uint32_t now = millis();
  if (want != 'S') { if (want != shown) drawArrow(want); shown = want; shownAt = now; }
  else if (shown != 'S' && now - shownAt > HOLD_MS) { drawArrow('S'); shown = 'S'; }
  else if (shown == 'S' && now - shownAt > 1000) { drawArrow('S'); shownAt = now; }

  if (now - lastPrint > 100) {
    lastPrint = now;
    Serial.print("ax="); Serial.print(ax, 2);
    Serial.print(" ay="); Serial.print(ay, 2);
    Serial.print(" az="); Serial.println(az, 2);
  }
  delay(10);
}
