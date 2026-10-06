// MCU side of motor_test.py / center_car.py (UNO Q). Drives both Cytron Maker
// Drive channels together so the car goes straight forward/back.
//   M1A <- pin 10, M1B <- pin 11
//   M2A <- pin 6,  M2B <- pin 5
//   A=PWM, B=LOW -> one direction;  A=LOW, B=PWM -> other;  both LOW -> brake
#include <Arduino_RouterBridge.h>

const int M1A = 10;
const int M1B = 11;
const int M2A = 6;
const int M2B = 5;

// Motors on opposite sides of a car are often mounted mirrored, so one has to
// spin "backwards" for the car to go straight. If the car spins in place
// instead of driving straight, flip M2_REVERSED.
const bool M2_REVERSED = false;

// Straightness trim (percent of commanded speed). Both motors always get the
// same command, but no two motors spin at exactly the same rate. If the car
// drifts, lower the trim of the faster motor (e.g. 95) until it tracks straight.
const int M1_TRIM = 100;
const int M2_TRIM = 100;

// Both-or-neither: at low PWM one motor can start while the other stays stuck,
// which pivots the car. Any nonzero command is raised to at least MIN_DRIVE
// (set it just above the PWM where the stiffer motor reliably starts), and
// every start/direction change begins with a short full-power KICK so both
// wheels break free together.
const int MIN_DRIVE = 120;
const int KICK_MS = 40;
int last_speed = 0;

// Drive one channel. speed: -255..255, 0 = brake.
void drive(int pinA, int pinB, int speed) {
  if (speed > 0) {
    analogWrite(pinA, speed);
    analogWrite(pinB, 0);
  } else if (speed < 0) {
    analogWrite(pinA, 0);
    analogWrite(pinB, -speed);
  } else {
    analogWrite(pinA, 0);
    analogWrite(pinB, 0);
  }
}

// The car's only command: one speed for both wheels, so it can go straight
// forward/back but never turn.
// speed: -255..255. Positive = forward, negative = backward, 0 = stop.
void set_motor(int speed) {
  speed = constrain(speed, -255, 255);
  if (speed != 0 && abs(speed) < MIN_DRIVE) {
    speed = speed > 0 ? MIN_DRIVE : -MIN_DRIVE;
  }

  // Starting from rest or reversing: kick both motors together first
  bool starting = speed != 0 && (last_speed == 0 || (speed > 0) != (last_speed > 0));
  last_speed = speed;
  if (starting) {
    int kick = speed > 0 ? 255 : -255;
    drive(M1A, M1B, kick);
    drive(M2A, M2B, M2_REVERSED ? -kick : kick);
    delay(KICK_MS);
  }

  int m1 = speed * M1_TRIM / 100;
  int m2 = speed * M2_TRIM / 100;
  drive(M1A, M1B, m1);
  drive(M2A, M2B, M2_REVERSED ? -m2 : m2);
}

void setup() {
  pinMode(M1A, OUTPUT);
  pinMode(M1B, OUTPUT);
  pinMode(M2A, OUTPUT);
  pinMode(M2B, OUTPUT);
  set_motor(0);

  Bridge.begin();
  Bridge.provide("set_motor", set_motor);
}

void loop() {}
