"""Verify the optional Ark text backend without making live API requests."""
import unittest
import httpx
from agent.config import LocaleLLMConfig
from agent.adapters.gemma_mm_client import GemmaMMClient


class TextBackendTests(unittest.IsolatedAsyncioTestCase):
    async def _complete(self, model):
        requests = []

        def handler(request):
            import json
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'message': {'content': 'Hello panel.'}}]})

        config = LocaleLLMConfig(model=model, base_url='https://example.test/v1',
                                 api_key='test', max_tokens=128,
                                 reply_language='English', include_heard_summary=True)
        client = GemmaMMClient(config, system_prompt='Be a host.')
        await client._http.aclose()
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            self.assertEqual(await client.complete_text('Open.'), 'Hello panel.')
        finally:
            await client.aclose()
        return requests[0]

    async def test_deepseek_disables_thinking(self):
        payload = await self._complete('deepseek-v4-flash')
        self.assertEqual(payload['thinking'], {'type': 'disabled'})

    async def test_gemma_payload_unchanged(self):
        payload = await self._complete('google/gemma-4-E4B-it')
        self.assertNotIn('thinking', payload)
