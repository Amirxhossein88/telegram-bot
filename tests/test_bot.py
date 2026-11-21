import asyncio
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace as Obj
from unittest.mock import AsyncMock

import bot


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

    def test_package_and_frame_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            video = folder / "video.mp4"
            video.write_bytes(b"the-video")
            frames = []
            for i in range(6):
                frame = folder / f"f{i}.jpg"
                frame.write_bytes(f"frame-{i}".encode())
                frames.append(frame)
            data = dict(cover_frame=3, caption="کپشن", bio="بیو", cover_text="کاور", notes="یادداشت")
            package = bot.make_package(folder, video, frames, data, "متن", "analysis", "trends")
            with zipfile.ZipFile(package) as result:
                self.assertEqual(result.read("cover.jpg"), b"frame-2")
                self.assertEqual(result.read("video.mp4"), b"the-video")
                self.assertEqual(result.read("caption.txt").decode(), "کپشن")
                self.assertIn("trend-research.txt", result.namelist())

    def test_citation_rendering(self):
        citation = Obj(type="url_citation", end_index=4, title="Example", url="https://example.com")
        response = Obj(output=[Obj(type="message", content=[Obj(type="output_text", text="test text", annotations=[citation])])])
        self.assertEqual(bot.render_research(response), "test [Example: https://example.com] text")


class AsyncCoreTests(unittest.IsolatedAsyncioTestCase):
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
            data = dict(cover_frame=1, caption="caption", bio="bio", cover_text="cover", notes="notes")
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
