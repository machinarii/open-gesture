import sys


def test_python_version_in_supported_range():
    assert (3, 10) <= sys.version_info[:2] < (3, 15)


def test_hashlib_is_not_broken():
    import hashlib

    hashlib.blake2b(b"open-gesture")


def test_package_imports():
    import open_gesture_annotate

    assert open_gesture_annotate.__version__ == "0.1.0"
