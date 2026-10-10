from pathlib import Path
from contextlib import ExitStack
import hashlib
import json
from unittest.mock import patch
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.src import data
from shared.src.data import clean,universe,validate_ohlcv
from shared.src.render import render_chart


def bars(n=20):
    close=100+np.arange(n)*.2
    return pd.DataFrame({'Open':close-.1,'High':close+.3,'Low':close-.4,'Close':close,'Volume':np.arange(n)+1000},index=pd.bdate_range('2018-01-01',periods=n))


class RenderingDataTests(unittest.TestCase):
    def test_universe_and_strict_cleaning(self):
        cfg=universe();self.assertEqual(len(cfg['tickers']),50);self.assertEqual(len(set(cfg['tickers'])),50)
        for ticker in ['TSLA','NVDA','PLTR','COIN','AAPL','MSFT','AMZN','META','KO','JNJ','PG','SPY']:self.assertIn(ticker,cfg['tickers'])
        frame=bars();out,dropped=clean(frame);pd.testing.assert_frame_equal(out,frame);self.assertEqual(dropped,0)
        for column,value in [('Open',np.nan),('High',1),('Low',300),('Volume',-1)]:
            bad=frame.copy();bad.iloc[-1,bad.columns.get_loc(column)]=value
            with self.assertRaises(ValueError):clean(bad)

    def test_styles_sizes_no_metadata_and_exact_pixel_mirror(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for style in ['candle','ohlc','line']:
                p=root/(style+'.png');m=root/(style+'_mirror.png')
                render_chart(bars(),p,20,(640,480),style);render_chart(bars(),m,20,(640,480),style,True)
                with Image.open(p) as image,Image.open(m) as flipped:
                    self.assertEqual(image.size,(640,480));self.assertEqual(image.mode,'RGB');self.assertEqual(image.info,{})
                    np.testing.assert_array_equal(np.asarray(ImageOps.mirror(image)),np.asarray(flipped))
            self.assertEqual(len({(root/(s+'.png')).read_bytes() for s in ['candle','ohlc','line']}),3)
            render_chart(bars(100),root/'wide.png',100,(1024,640))
            with Image.open(root/'wide.png') as image:
                self.assertEqual(image.size,(1024,640))

    def test_only_visible_window_controls_pixels_and_length_is_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frame=bars(40)
            render_chart(frame.iloc[:20],root/'before.png')
            frame.iloc[20:,:4]*=2
            render_chart(frame.iloc[:20],root/'after.png')
            self.assertEqual((root/'before.png').read_bytes(),(root/'after.png').read_bytes())
            with self.assertRaises(ValueError):render_chart(frame,root/'bad.png')


class DataProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.raw_dir = self.root / "data"
        self.raw_dir.mkdir()
        (self.root / "results").mkdir()
        self.config = self.root / "universe.json"
        self.cfg = {"exp03_tickers": ["AAA"], "additional_tickers": ["BBB"],
                    "start": "2018-01-01", "end": "2018-12-31"}
        self.config.write_text(json.dumps(self.cfg))
        self.frame = bars()
        self.frame.index.name = "Date"
        entries = {}
        for ticker in ["AAA", "BBB"]:
            target = self.raw_dir / f"{ticker}.parquet"
            self.frame.to_parquet(target)
            entries[ticker] = {"rows": len(self.frame), "first_session": "2018-01-01",
                               "last_session": "2018-01-26", "dropped_incomplete_rows": 0,
                               "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        self.manifest = {"source": "Yahoo Finance via yfinance", "yfinance_version": "original-version",
                         "downloaded_at_utc": "2020-01-01T00:00:00+00:00", "auto_adjust": True,
                         "start": self.cfg["start"], "end": self.cfg["end"],
                         "config_sha256": hashlib.sha256(self.config.read_bytes()).hexdigest(),
                         "tickers": entries}
        # Deliberately noncanonical spacing makes byte preservation test meaningful.
        self.original = (json.dumps(self.manifest, indent=4) + "\n\n").encode()
        self.local_manifest = self.raw_dir / "manifest.json"
        self.public_manifest = self.root / "results/data_manifest.json"
        self.local_manifest.write_bytes(self.original)
        self.public_manifest.write_bytes(self.original)
        for name, value in [("SHARED", self.root), ("DATA_DIR", self.raw_dir), ("CONFIG", self.config)]:
            self.stack.enter_context(patch.object(data, name, value))
        self.stack.enter_context(patch("yfinance.set_tz_cache_location"))

    def run_cli(self, *args):
        with patch.object(sys, "argv", ["data.py", *args]):
            data.main()

    def test_cached_verify_preserves_exact_manifest_bytes_and_acquisition(self):
        before_times = [p.stat().st_mtime_ns for p in [self.local_manifest, self.public_manifest]]
        with patch.object(data, "download_ticker", side_effect=AssertionError("No download allowed")):
            self.run_cli()
        for p, timestamp in zip([self.local_manifest, self.public_manifest], before_times):
            self.assertEqual(p.read_bytes(), self.original)
            self.assertEqual(p.stat().st_mtime_ns, timestamp)
        pd.testing.assert_frame_equal(data.load("AAA"), self.frame, check_freq=False)

    def test_load_checks_config_checksum_and_published_identity_before_parsing(self):
        with patch.object(data.pd, "read_parquet", side_effect=AssertionError("Must fail before parsing")) as reader:
            self.config.write_text(json.dumps({**self.cfg, "end": "2018-06-01"}))
            with self.assertRaisesRegex(ValueError, "configuration"):
                data.load("AAA")
            self.config.write_text(json.dumps(self.cfg))
            self.public_manifest.write_bytes(self.original + b" ")
            with self.assertRaisesRegex(ValueError, "manifests differ"):
                data.load("AAA")
            self.public_manifest.write_bytes(self.original)
            target = self.raw_dir / "AAA.parquet"
            target.write_bytes(target.read_bytes() + b"modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                data.load("AAA")
            self.assertEqual(reader.call_count, 0)

    def test_configuration_mismatch_requires_refresh_and_drops_stale_entries(self):
        self.config.write_text(json.dumps({**self.cfg, "additional_tickers": [], "end": "2018-06-01"}))
        with patch.object(data, "download_ticker", return_value=(self.frame, 0)) as download:
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.run_cli("--tickers", "AAA")
            download.assert_not_called()
            self.assertEqual(self.local_manifest.read_bytes(), self.original)
            self.run_cli("--refresh", "--tickers", "AAA")
            download.assert_called_once_with("AAA", "2018-01-01", "2018-06-01")
        latest = json.loads(self.local_manifest.read_bytes())
        self.assertEqual(set(latest["tickers"]), {"AAA"})
        self.assertIn("downloaded_at_utc", latest["tickers"]["AAA"])
        self.assertIn("yfinance_version", latest["tickers"]["AAA"])
        self.assertEqual(latest["config_sha256"], hashlib.sha256(self.config.read_bytes()).hexdigest())
        self.assertEqual(self.local_manifest.read_bytes(), self.public_manifest.read_bytes())
        with self.assertRaisesRegex(ValueError, "manifest entry"):
            data.load("BBB")

    def test_missing_snapshot_download_records_only_its_new_acquisition(self):
        (self.raw_dir / "BBB.parquet").unlink()
        with patch.object(data, "download_ticker", return_value=(self.frame, 0)) as download:
            self.run_cli()
            download.assert_called_once_with("BBB", "2018-01-01", "2018-12-31")
        latest = json.loads(self.local_manifest.read_bytes())
        self.assertEqual(latest["tickers"]["AAA"], self.manifest["tickers"]["AAA"])
        self.assertEqual(latest["downloaded_at_utc"], self.manifest["downloaded_at_utc"])
        self.assertEqual(latest["yfinance_version"], self.manifest["yfinance_version"])
        self.assertIn("downloaded_at_utc", latest["tickers"]["BBB"])
        self.assertNotEqual(latest["tickers"]["BBB"]["yfinance_version"], "original-version")
        pd.testing.assert_frame_equal(data.load("BBB"), self.frame, check_freq=False)


if __name__=='__main__':unittest.main()
