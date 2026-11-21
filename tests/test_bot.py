import asyncio
import base64
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace as Obj
from unittest.mock import AsyncMock, MagicMock, patch

import bot
import gemini_provider
import cover_provider


class CoreTests(unittest.TestCase):
    def test_private_and_allowlisted(self):
        settings = Obj(allowed={123})
        self.assertTrue(bot.authorized(Obj(effective_chat=Obj(type="private"), effective_user=Obj(id=123)), settings))
        self.assertFalse(bot.authorized(Obj(effective_chat=Obj(type="group"), effective_user=Obj(id=123)), settings))
        self.assertFalse(bot.authorized(Obj(effective_chat=Obj(type="private"), effective_user=Obj(id=456)), settings))

    def test_duration_boundaries(self):
        self.assertEqual(bot.parse_duration("Duration: 00:00:13.10"), 13.1)
        for value in ("Duration: 00:00:00.00", "Duration: 00:15:00.01", "not video"):
            with self.assertRaises(ValueError):
                bot.parse_duration(value)

    def test_package_contains_only_video_bio_and_generated_cover(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            video = folder / "video.mp4"
            video.write_bytes(b"the-video")
            cover = folder / "generated-cover.png"
            cover.write_bytes(b"generated-image")
            data = dict(caption="کپشن\n#هشتگ", cover_prompt="image description")
            package = bot.make_package(folder, video, cover, data)
            with zipfile.ZipFile(package) as result:
                self.assertEqual(result.read("cover.png"), b"generated-image")
                self.assertEqual(result.read("video.mp4"), b"the-video")
                self.assertEqual(result.read("bio.txt").decode(), "کپشن\n#هشتگ")
                self.assertEqual(set(result.namelist()), {"video.mp4", "cover.png", "bio.txt"})

    def test_citation_rendering(self):
        citation = Obj(type="url_citation", end_index=4, title="Example", url="https://example.com")
        response = Obj(output=[Obj(type="message", content=[Obj(type="output_text", text="test text", annotations=[citation])])])
        self.assertEqual(bot.render_research(response), "test [Example: https://example.com] text")


class AsyncCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_package_sent_to_both_or_once_when_owner_is_sender(self):
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp) / "package.zip"
            package.write_bytes(b"package")
            fake = Obj(send_document=AsyncMock())
            self.assertEqual(await bot.send_package(fake, package, 123, 456), [])
            self.assertEqual([x.kwargs["chat_id"] for x in fake.send_document.call_args_list], [123, 456])
            fake.send_document.reset_mock()
            self.assertEqual(await bot.send_package(fake, package, 456, 456), [])
            self.assertEqual(fake.send_document.await_count, 1)
            fake.send_document = AsyncMock(side_effect=[RuntimeError("blocked"), None])
            self.assertEqual(await bot.send_package(fake, package, 123, 456), [123])
            self.assertEqual(fake.send_document.await_count, 2)

    async def test_cover_requires_config(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(cover_provider.CoverConfigurationError):
                await cover_provider.generate_cover(Path(temp), "landscape")

    async def test_cover_uses_generated_image_response(self):
        image = b"\x89PNG\r\n\x1a\n" + b"generated-image"
        payload = json.dumps({"success": True, "result": {"image": base64.b64encode(image).decode()}}).encode()
        class Stream:
            async def iter_chunked(self, size):
                yield payload
        response = Obj(status=200, content=Stream())
        request = MagicMock()
        request.__aenter__ = AsyncMock(return_value=response)
        request.__aexit__ = AsyncMock(return_value=None)
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.post.return_value = request
        env = {"CLOUDFLARE_ACCOUNT_ID": "a" * 32, "CLOUDFLARE_API_TOKEN": "test-token"}
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", env), \
             patch.object(cover_provider, "ClientSession", return_value=session):
            cover = await cover_provider.generate_cover(Path(temp), "a landscape")
            self.assertEqual(cover.read_bytes(), image)
            self.assertEqual(cover.suffix, ".png")
            self.assertIn("flux-2-klein-4b", session.post.call_args.args[0])

    async def test_gemini_retries_service_unavailable_and_uses_fallback(self):
        def response(status, data=None):
            manager = MagicMock()
            manager.__aenter__ = AsyncMock(return_value=Obj(status=status, json=AsyncMock(return_value=data)))
            manager.__aexit__ = AsyncMock(return_value=None)
            return manager
        answer = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
        session = Obj(post=MagicMock(side_effect=[response(503), response(503), response(200, answer)]))
        with patch.object(gemini_provider.asyncio, "sleep", new_callable=AsyncMock):
            result = await gemini_provider._generate(session, "secret", "gemini-3.8-flash", [{"text": "test"}])
        self.assertEqual(result, {"ok": True})
        self.assertEqual(session.post.call_count, 3)
        self.assertIn("gemini-3-flash-preview", session.post.call_args.args[0])

    async def test_gemini_packages_audio_frames_and_headlines(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            frames = []
            for i in range(6):
                frame = folder / f"{i}.jpg"
                frame.write_bytes(b"image")
                frames.append(frame)
            audio = folder / "audio.mp3"
            audio.write_bytes(b"audio")
            content = dict(caption="caption #tag", cover_prompt="vertical creative cover")
            answers = [dict(analysis="analysis", transcript="speech", search_terms="topic"), content]
            with patch.object(gemini_provider, "_generate", new_callable=AsyncMock, side_effect=answers) as generate, \
                 patch.object(gemini_provider, "_headlines", new_callable=AsyncMock, return_value="current headlines"):
                result = await gemini_provider.prepare_content_gemini(Obj(api_key="key", model="model"), frames, audio)
            self.assertEqual(result, (content, "speech", "analysis", "current headlines"))
            self.assertEqual(len(generate.call_args_list[0].args[3]), 8)
            self.assertIn("current headlines", generate.call_args_list[1].args[3][0]["text"])

    async def test_extract_real_video_frames_and_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            video = folder / "sample.mp4"
            await bot.ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-f", "lavfi", "-i", "sine=frequency=1000", "-t", "2", "-pix_fmt", "yuv420p", "-y", video)
            frames, audio = await bot.extract_media(video, folder)
            self.assertEqual(len(frames), 6)
            self.assertTrue(all(f.stat().st_size > 0 for f in frames))
            self.assertGreater(audio.stat().st_size, 0)

    async def test_prepare_content_uses_audio_frames_and_search(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            frames = []
            for i in range(6):
                frame = folder / f"{i}.jpg"
                frame.write_bytes(b"image")
                frames.append(frame)
            audio = folder / "audio.mp3"
            audio.write_bytes(b"audio")
            data = dict(caption="caption #tag", cover_prompt="vertical cover")
            research = Obj(output=[Obj(type="message", content=[Obj(type="output_text", text="research", annotations=[])])])
            create = AsyncMock(side_effect=[Obj(output_text="visual analysis"), research, Obj(output_text=json.dumps(data))])
            transcribe = AsyncMock(return_value=Obj(text="spoken text"))
            client = Obj(responses=Obj(create=create), audio=Obj(transcriptions=Obj(create=transcribe)))
            result = await bot.prepare_content(client, Obj(model="model", transcription_model="transcribe"), frames, audio)
            self.assertEqual(result[0], data)
            self.assertEqual(result[1], "spoken text")
            self.assertEqual(create.call_args_list[1].kwargs["tool_choice"], "required")
            self.assertEqual(len(create.call_args_list[0].kwargs["input"][0]["content"]), 7)

    async def test_unauthorized_upload_does_not_call_services(self):
        instance = bot.Nexify.__new__(bot.Nexify)
        instance.settings = Obj(allowed={123})
        message = Obj(reply_text=AsyncMock())
        update = Obj(effective_chat=Obj(type="private"), effective_user=Obj(id=999), message=message)
        await instance.video(update, Obj())
        message.reply_text.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
