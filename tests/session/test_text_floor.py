"""Local text rounds must unlock human input after the closing audio finishes."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from agent.show_graph.actuators import LiveActuators
from agent.config import load_scenario


class TextFloorTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_returns_floor_in_text_mode(self):
        data = SimpleNamespace(scenario=load_scenario(), floor_next_speaker='host', turn_log=None,
                               touch_activity=Mock(), hand_raise_queue=Mock())
        data.hand_raise_queue.roles.return_value = []
        with patch.dict(os.environ, {'TALKSHOW_TEXT_INPUT': '1'}), \
             patch('agent.ui.ui_events.emit_floor_grant', new_callable=AsyncMock) as grant, \
             patch('agent.floor.hand_raise_ui.dequeue_hand_raise', new_callable=AsyncMock):
            await LiveActuators(None, data, None).after_line({'kind': 'close'})
            grant.assert_awaited_once_with('human', reason='round_closed')
            self.assertEqual(data.floor_next_speaker, 'human')

    async def test_upstream_audio_close_is_unchanged(self):
        data = SimpleNamespace(floor_next_speaker='host', touch_activity=Mock())
        with patch.dict(os.environ, {'TALKSHOW_TEXT_INPUT': '0'}), \
             patch('agent.ui.ui_events.emit_floor_grant', new_callable=AsyncMock) as grant:
            await LiveActuators(None, data, None).after_line({'kind': 'close'})
            grant.assert_not_awaited()
            self.assertEqual(data.floor_next_speaker, 'host')
