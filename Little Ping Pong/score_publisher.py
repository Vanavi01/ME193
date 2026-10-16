'''
Publish the forehand drill's results over MQTT when a round is completed.

    ME193/Rogers/Victor          final score, as a float      e.g. 900.0
    ME193/Rogers/Victor/hits     balls returned (any stroke)  e.g. 18
    ME193/Rogers/Victor/misses   balls missed                 e.g. 2

Uses mqttlib.py (the course's paho-mqtt wrapper, test.mosquitto.org by default).
Connecting and publishing happen on a background thread, so a slow or offline
network never freezes the game; `status` says how it went (shown on the results screen).
'''

import queue
import threading

TOPIC = 'ME193/Rogers/Victor'
BROKER = 'test.mosquitto.org'


def round_summary(drill):
    hits = sum(o in ('correct', 'wrong') for _, o in drill.results)
    misses = sum(o == 'miss' for _, o in drill.results)
    return {'score': float(drill.score), 'hits': hits, 'misses': misses}


def round_messages(drill, topic=TOPIC):
    '''[(topic, payload), ...] for a finished round.'''
    s = round_summary(drill)
    return [(topic, f"{s['score']:.1f}"),
            (f'{topic}/hits', str(s['hits'])),
            (f'{topic}/misses', str(s['misses']))]


class ScorePublisher:

    def __init__(self, topic=TOPIC, broker=BROKER, client_factory=None, enabled=True):
        self.topic = topic
        self.broker = broker
        self.enabled = enabled
        self.status = 'off' if not enabled else 'idle'      # idle / sending / sent / failed: <why> / off
        self._factory = client_factory or self._default_client
        self._client = None
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _default_client(self):
        from mqttlib import MQTTClient
        return MQTTClient(broker=self.broker)

    def publish_round(self, drill):
        '''Queue a finished round's results; returns at once.'''
        if not self.enabled:
            return
        self.status = 'sending'
        self._queue.put(round_messages(drill, self.topic))

    def close(self, timeout=3.0):
        '''Wait (briefly) for anything still being sent, then disconnect.'''
        self._queue.put(None)
        self._thread.join(timeout)
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass

    def _worker(self):
        while True:
            messages = self._queue.get()
            if messages is None:
                return
            try:
                if self._client is None:
                    client = self._factory()
                    client.connect()
                    connected = getattr(client, '_connected', None)     # mqttlib waits but doesn't raise
                    if connected is not None and not connected.is_set():
                        raise ConnectionError(f'no answer from {self.broker}')
                    self._client = client
                for topic, payload in messages:
                    self._client.publish(topic, payload, qos=1)
                self.status = 'sent'
            except Exception as e:                      # never crash the game over the network
                self._client = None
                self.status = f'failed: {e}'
                print(f'MQTT: could not publish ({e})')
