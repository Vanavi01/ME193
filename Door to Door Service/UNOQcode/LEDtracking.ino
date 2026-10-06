// UNO Q microcontroller side. Exposes show/clear/drive to the Python side.
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

Arduino_LED_Matrix matrix;
const int COLS = 13, ROWS = 8;
uint8_t frame[ROWS * COLS];

// Motor driver pins (TB6612 / L298N style: one PWM + two direction pins)
const int PWM_PIN = 9;
const int IN1 = 7;
const int IN2 = 8;

void show(int col, int row) {
  memset(frame, 0, sizeof(frame));
  if (col >= 0 && col < COLS && row >= 0 && row < ROWS) {
    frame[row * COLS + col] = 7;   // full brightness at 3-bit grayscale
  }
  matrix.draw(frame);
}

void clearDisplay() {
  memset(frame, 0, sizeof(frame));
  matrix.draw(frame);
}

void drive(int speed) {
  speed = constrain(speed, -255, 255);
  if (speed > 0)      { digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW); }
  else if (speed < 0) { digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH); }
  else                { digitalWrite(IN1, LOW);  digitalWrite(IN2, LOW); }
  analogWrite(PWM_PIN, abs(speed));
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  pinMode(PWM_PIN, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  drive(0);

  Bridge.begin();
  Bridge.provide("show", show);
  Bridge.provide("clear", clearDisplay);
  Bridge.provide("drive", drive);
}

void loop() {}
