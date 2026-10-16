'''Publishing a finished drill round over MQTT (fake client: no network needed).'''

import threading
import time
import unittest

from drill import Drill
from score_publisher import TOPIC, ScorePublisher, round_messages, round_summary


def finished_round():
    d = Drill(balls=5)
    d.score = 540
    d.results = [('smash', 'correct'), ('hit', 'wrong'), ('chop', 'miss'), ('hit', 'correct'), ('chop', 'miss')]
    return d


class FakeClient:
    def __init__(self, connect_ok=True, connect_error=None, delay=0.0):
        self.published = []
        self.connected = False
        self._connected = threading.Event()
        self.connect_ok, self.connect_error, self.delay = connect_ok, connect_error, delay

    def connect(self):
        time.sleep(self.delay)
        if self.connect_error:
            raise self.connect_error
        if self.connect_ok:
            self._connected.set()

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, payload, qos))

    def disconnect(self):
        self.connected = False


def wait_status(p, timeout=2.0):
    end = time.perf_counter() + timeout
    while p.status == 'sending' and time.perf_counter() < end:
        time.sleep(0.01)
    return p.status


class TestMessages(unittest.TestCase):

    def test_summary_counts_hits_and_misses(self):
        self.assertEqual(round_summary(finished_round()), {'score': 540.0, 'hits': 3, 'misses': 2})

    def test_topics_and_payloads(self):
        self.assertEqual(round_messages(finished_round()), [
            ('ME193/Rogers/Victor', '540.0'),          # the assignment's float score
            ('ME193/Rogers/Victor/hits', '3'),
            ('ME193/Rogers/Victor/misses', '2'),
        ])
        self.assertEqual(TOPIC, 'ME193/Rogers/Victor')

    def test_score_parses_as_a_float(self):
        self.assertIsInstance(float(round_messages(finished_round())[0][1]), float)


class TestPublisher(unittest.TestCase):

    def test_publishes_all_three_messages_in_the_background(self):
        client = FakeClient(delay=0.2)                  # slow network
        p = ScorePublisher(client_factory=lambda: client)
        t = time.perf_counter()
        p.publish_round(finished_round())
        self.assertLess(time.perf_counter() - t, 0.02)  # the game is not held up
        self.assertEqual(wait_status(p), 'sent')
        self.assertEqual([m[:2] for m in client.published], round_messages(finished_round()))
        self.assertTrue(all(m[2] == 1 for m in client.published))      # qos 1
        p.close()

    def test_reuses_the_connection_for_the_next_round(self):
        made = []
        p = ScorePublisher(client_factory=lambda: made.append(FakeClient()) or made[-1])
        for _ in range(2):
            p.publish_round(finished_round())
            wait_status(p)
        p.close()
        self.assertEqual(len(made), 1)
        self.assertEqual(len(made[0].published), 6)

    def test_offline_is_reported_not_raised(self):
        p = ScorePublisher(client_factory=lambda: FakeClient(connect_error=OSError('getaddrinfo failed')))
        p.publish_round(finished_round())
        self.assertTrue(wait_status(p).startswith('failed'))
        p.close()

    def test_broker_that_never_answers_is_a_failure_not_sent(self):
        p = ScorePublisher(client_factory=lambda: FakeClient(connect_ok=False))
        p.publish_round(finished_round())
        self.assertTrue(wait_status(p).startswith('failed'))
        p.close()

    def test_retries_after_a_failure(self):
        clients = [FakeClient(connect_error=OSError('down')), FakeClient()]
        p = ScorePublisher(client_factory=lambda: clients.pop(0))
        p.publish_round(finished_round())
        wait_status(p)
        p.publish_round(finished_round())
        self.assertEqual(wait_status(p), 'sent')
        p.close()

    def test_disabled_sends_nothing(self):
        p = ScorePublisher(client_factory=lambda: self.fail('should not connect'), enabled=False)
        p.publish_round(finished_round())
        self.assertEqual(p.status, 'off')
        p.close()

    def test_close_lets_a_pending_round_go_out(self):
        client = FakeClient(delay=0.1)
        p = ScorePublisher(client_factory=lambda: client)
        p.publish_round(finished_round())
        p.close()                                       # quitting right after the round ends
        self.assertEqual(len(client.published), 3)


if __name__ == '__main__':
    unittest.main()
