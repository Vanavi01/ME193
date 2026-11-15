// Impenetrable Safe - UnoQ microcontroller sketch
// Drives the servo and 3 LEDs, counts button presses. Controlled from Python through the
// Bridge (see python/main.py). Change the pin numbers below to match your wiring.
//
// LED modes: 0 = off, 1 = on, 2 = fast blink.
// Servo: M5Stack 360 = CONTINUOUS rotation. The pulse width sets speed + direction (1500 us =
// stop), so locking = spin one way for lock_ms, unlocking = spin the other way for unlock_ms.
// Pulses are generated in software (50 Hz); the UnoQ core has no <Servo.h>. The timings come
// from the PC (calibrated with `python -m src.hw_test --servo`) every time it connects.

#include <Arduino_RouterBridge.h>

// ---- pins (EDIT to match your wiring) ----
const int PIN_SERVO  = 9;   // servo signal (orange/yellow wire)
const int PIN_RED    = 5;   // LEDs left to right like on screen: red, yellow, green
const int PIN_YELLOW = 6;   // each: pin -> resistor -> LED long leg; short leg -> GND
const int PIN_GREEN  = 7;
const int PIN_BUTTON = 2;   // button between D2 and GND (internal pull-up, no resistor)
const int PIN_HALL   = 3;   // Hall sensor output (not installed yet)
const bool HALL_INSTALLED = false;

const int STOP_US = 1500;
const unsigned long STOP_PULSES_MS = 200;   // send "stop" briefly after a spin, then go quiet
int lockUs = 1300, unlockUs = 1700;         // defaults (speed 200); the PC sends its values
unsigned long lockMs = 500, unlockMs = 500;   // 500 ms each way, ends against the stops

const int ledPins[3] = {PIN_YELLOW, PIN_GREEN, PIN_RED};
volatile int ledMode[3] = {0, 0, 0};
volatile int lockedState = -1;              // -1 = unknown until the first command
volatile int runUs = STOP_US;
volatile unsigned long runStart = 0, runMs = 0;
volatile int pendingPresses = 0;

int lastReading = HIGH, stableState = HIGH;
unsigned long lastChange = 0, lastPulse = 0;

// ---- functions Python calls through the Bridge ----
void servo_run(int us, int ms) {             // spin at `us` for `ms`, then stop
  runUs = constrain(us, 500, 2500);
  runMs = (unsigned long)max(ms, 0);
  runStart = millis();
}

void servo_config(int lu, int uu, int lm, int um) {
  lockUs = lu; unlockUs = uu; lockMs = lm; unlockMs = um;
}

void servo_state(int locked) {               // where the latch is now (no movement)
  lockedState = locked ? 1 : 0;
}

void set_outputs(int locked, int y, int g, int r, int alarm) {
  locked = locked ? 1 : 0;
  if (lockedState == -1) {
    lockedState = locked;                    // first command after boot: trust it, don't move
  } else if (locked != lockedState) {
    lockedState = locked;
    if (locked) servo_run(lockUs, lockMs);
    else servo_run(unlockUs, unlockMs);
  }
  ledMode[0] = y;
  ledMode[1] = g;
  ledMode[2] = r;
  (void)alarm;  // reserved for a buzzer
}

int get_button() {           // presses since the last call
  int n = pendingPresses;
  pendingPresses = 0;
  return n;
}

int get_button_raw() {       // 1 while the button is held down (wiring check)
  return digitalRead(PIN_BUTTON) == LOW ? 1 : 0;
}

int get_door() {             // 1 = open (magnet away), 0 = closed / no sensor
  if (!HALL_INSTALLED) return 0;
  return digitalRead(PIN_HALL) == HIGH ? 1 : 0;
}

void setup() {
  for (int i = 0; i < 3; i++) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], LOW);
  }
  pinMode(PIN_BUTTON, INPUT_PULLUP);
  pinMode(PIN_HALL, INPUT_PULLUP);
  pinMode(PIN_SERVO, OUTPUT);
  digitalWrite(PIN_SERVO, LOW);      // no pulses = servo idle

  Bridge.begin();
  Bridge.provide("set_outputs", set_outputs);
  Bridge.provide("get_button", get_button);
  Bridge.provide("get_button_raw", get_button_raw);
  Bridge.provide("get_door", get_door);
  Bridge.provide("servo_run", servo_run);
  Bridge.provide("servo_config", servo_config);
  Bridge.provide("servo_state", servo_state);
}

void loop() {
  unsigned long now = millis();

  // button: count a press on each debounced HIGH -> LOW edge
  int reading = digitalRead(PIN_BUTTON);
  if (reading != lastReading) {
    lastChange = now;
    lastReading = reading;
  }
  if (now - lastChange > 30 && reading != stableState) {
    stableState = reading;
    if (stableState == LOW) pendingPresses++;
  }

  // LEDs: on / off / fast blink (4 Hz)
  bool blinkOn = (now / 125) % 2;
  for (int i = 0; i < 3; i++) {
    digitalWrite(ledPins[i], (ledMode[i] == 1 || (ledMode[i] == 2 && blinkOn)) ? HIGH : LOW);
  }

  // servo: 50 Hz pulses while spinning, then "stop" pulses briefly, then nothing (idle).
  unsigned long since = now - runStart;
  int width = 0;
  if (since < runMs) width = runUs;
  else if (since < runMs + STOP_PULSES_MS) width = STOP_US;
  if (width && now - lastPulse >= 20) {
    lastPulse = now;
    unsigned long t0 = micros();
    digitalWrite(PIN_SERVO, HIGH);
    while (micros() - t0 < (unsigned long)width) { }
    digitalWrite(PIN_SERVO, LOW);
  }
}
