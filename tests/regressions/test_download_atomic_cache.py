"""Regression test for BUG-006 -- poisoned OBO/GAF download cache.

Pre-fix symptom: `_download_file` wrote the response body straight to the
cache path:

    with urllib.request.urlopen(req) as response, open(dest, "wb") as out:
        out.write(response.read())

If the transfer died mid-body, `response.read()` raised
`http.client.IncompleteRead` -- which subclasses HTTPException/ValueError, NOT
URLError or OSError -- so it escaped the `except (URLError, OSError)` retry
guard entirely. Worse, `open(dest, "wb")` had already created the file, so a
zero-byte `go-basic.obo` was left in the cache. Every later run then hit
`if cached_path.exists(): return cached_path` and returned the empty file
without ever retrying. `_parse_obo` on an empty file returns {} with no error,
so the tool silently emitted an empty ontology: degenerate clustering and empty
GO-tree figures, with no warning.

A plain connection-refused does NOT trigger this (urlopen raises before open()),
which is why the defect survived a fully green suite. It requires failure
*mid-body*, which these tests reproduce with a local socket server.

Post-fix: the body streams to a sibling `.part` file, is validated, and is
promoted with os.replace() only on success. The cache therefore only ever holds
complete files. Cache reads reject zero-byte entries, healing caches already
poisoned by the old code.
"""

import gzip
import socket
import threading

import pytest

from gsea_tool.go_clustering import (
    CorruptDownloadError,
    _download_file,
    _validate_download,
    download_or_load_gaf,
    download_or_load_obo,
)

VALID_OBO = (
    b"format-version: 1.2\n"
    b"ontology: go\n"
    b"\n"
    b"[Term]\n"
    b"id: GO:0008150\n"
    b"name: biological_process\n"
    b"namespace: biological_process\n"
)


class _Server:
    """Minimal HTTP server that can truncate its response body mid-transfer.

    Counts requests so tests can assert that a cache hit performs no I/O.
    """

    def __init__(self, body: bytes, *, truncate: bool = False):
        self.body = body
        self.truncate = truncate
        self.requests = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            try:
                conn.recv(65536)
                self.requests += 1
                if self.truncate:
                    # Promise more than we send, then hang up: IncompleteRead.
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n"
                        + self.body[:16]
                    )
                else:
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Length: "
                        + str(len(self.body)).encode()
                        + b"\r\n\r\n"
                        + self.body
                    )
            finally:
                conn.close()

    def url(self, name):
        return f"http://127.0.0.1:{self.port}/{name}"

    def close(self):
        self._sock.close()


@pytest.fixture
def truncating_server():
    srv = _Server(VALID_OBO, truncate=True)
    yield srv
    srv.close()


@pytest.fixture
def good_server():
    srv = _Server(VALID_OBO)
    yield srv
    srv.close()


class TestTruncatedTransfer:
    """A transfer that dies mid-body must not poison the cache."""

    def test_truncated_download_raises_connection_error(self, truncating_server, tmp_path):
        # Pre-fix: http.client.IncompleteRead escaped the retry guard.
        with pytest.raises(ConnectionError):
            download_or_load_obo(truncating_server.url("go-basic.obo"), tmp_path)

    def test_truncated_download_leaves_no_cache_file(self, truncating_server, tmp_path):
        with pytest.raises(ConnectionError):
            download_or_load_obo(truncating_server.url("go-basic.obo"), tmp_path)
        # Pre-fix: a zero-byte go-basic.obo was left behind here.
        assert not (tmp_path / "go-basic.obo").exists()

    def test_truncated_download_leaves_no_part_file(self, truncating_server, tmp_path):
        with pytest.raises(ConnectionError):
            download_or_load_obo(truncating_server.url("go-basic.obo"), tmp_path)
        assert list(tmp_path.glob("*.part")) == []

    def test_truncation_is_retried(self, truncating_server, tmp_path):
        """IncompleteRead must be treated as transient: two attempts, not one."""
        with pytest.raises(ConnectionError):
            download_or_load_obo(truncating_server.url("go-basic.obo"), tmp_path)
        assert truncating_server.requests == 2


class TestPoisonedCacheHealing:
    """A zero-byte cache entry from the old code must be discarded, not returned."""

    def test_zero_byte_obo_cache_is_replaced(self, good_server, tmp_path):
        poisoned = tmp_path / "go-basic.obo"
        poisoned.write_bytes(b"")
        result = download_or_load_obo(good_server.url("go-basic.obo"), tmp_path)
        # Pre-fix: returned the empty file and never contacted the server.
        assert good_server.requests == 1
        assert result.read_bytes() == VALID_OBO

    def test_zero_byte_gaf_cache_is_replaced(self, tmp_path):
        gaf_bytes = gzip.compress(b"!gaf-version: 2.2\nFB\tFBgn1\tg\t\tGO:0008150\n")
        srv = _Server(gaf_bytes)
        try:
            poisoned = tmp_path / "fb.gaf.gz"
            poisoned.write_bytes(b"")
            result = download_or_load_gaf(srv.url("fb.gaf.gz"), tmp_path)
            assert srv.requests == 1
            assert result.read_bytes() == gaf_bytes
        finally:
            srv.close()

    def test_nonempty_cache_is_reused_without_network(self, good_server, tmp_path):
        cached = tmp_path / "go-basic.obo"
        cached.write_bytes(VALID_OBO)
        result = download_or_load_obo(good_server.url("go-basic.obo"), tmp_path)
        assert good_server.requests == 0
        assert result == cached


class TestAtomicity:
    """The cache path is only ever created by an atomic rename."""

    def test_successful_download_leaves_no_part_file(self, good_server, tmp_path):
        download_or_load_obo(good_server.url("go-basic.obo"), tmp_path)
        assert list(tmp_path.glob("*.part")) == []

    def test_part_file_preserves_gz_extension(self, tmp_path):
        """with_name, not with_suffix: 'fb.gaf.gz' must not become 'fb.gaf.part'."""
        gaf_bytes = gzip.compress(b"!gaf-version: 2.2\n")
        srv = _Server(gaf_bytes)
        try:
            dest = tmp_path / "fb.gaf.gz"
            _download_file(srv.url("fb.gaf.gz"), dest, kind="gaf")
            assert dest.exists()
            assert dest.read_bytes() == gaf_bytes
        finally:
            srv.close()


class TestValidation:
    """A 200 response carrying the wrong payload is not a valid download."""

    def test_html_error_page_rejected_as_obo(self, tmp_path):
        srv = _Server(b"<!DOCTYPE html><html><body>404 Not Found</body></html>")
        try:
            with pytest.raises(ConnectionError):
                download_or_load_obo(srv.url("go-basic.obo"), tmp_path)
            assert not (tmp_path / "go-basic.obo").exists()
        finally:
            srv.close()

    def test_non_gzip_rejected_as_gaf_gz(self, tmp_path):
        srv = _Server(b"not gzipped at all")
        try:
            with pytest.raises(ConnectionError):
                download_or_load_gaf(srv.url("fb.gaf.gz"), tmp_path)
        finally:
            srv.close()

    def test_empty_file_rejected(self, tmp_path):
        empty = tmp_path / "go-basic.obo"
        empty.write_bytes(b"")
        with pytest.raises(CorruptDownloadError):
            _validate_download(empty, "obo")

    def test_valid_obo_accepted(self, tmp_path):
        good = tmp_path / "go-basic.obo"
        good.write_bytes(VALID_OBO)
        _validate_download(good, "obo")  # must not raise


class TestLocalPathOverride:
    """clustering.go_obo_path -- the supported offline workflow."""

    def test_local_obo_used_without_network(self, good_server, tmp_path):
        local = tmp_path / "my-go.obo"
        local.write_bytes(VALID_OBO)
        cache = tmp_path / "cache"
        result = download_or_load_obo(good_server.url("go-basic.obo"), cache, local_path=local)
        assert result == local
        assert good_server.requests == 0
        assert not cache.exists()  # no cache dir created either

    def test_missing_local_obo_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            download_or_load_obo("http://unused.invalid/go.obo", tmp_path,
                                 local_path=tmp_path / "absent.obo")

    def test_corrupt_local_obo_raises_rather_than_downloading(self, good_server, tmp_path):
        """A bad override is reported, never silently swapped for a download."""
        local = tmp_path / "bad.obo"
        local.write_bytes(b"<html>nope</html>")
        with pytest.raises(CorruptDownloadError):
            download_or_load_obo(good_server.url("go-basic.obo"), tmp_path, local_path=local)
        assert good_server.requests == 0

    def test_empty_string_local_path_falls_back_to_download(self, good_server, tmp_path):
        """Config default is "" -- must behave as 'not set'."""
        result = download_or_load_obo(good_server.url("go-basic.obo"), tmp_path, local_path="")
        assert good_server.requests == 1
        assert result.read_bytes() == VALID_OBO
